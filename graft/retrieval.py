"""Time-aware prefix search, complete-leaf selection, and leaf-local resolution."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .config import SearchConfig
from .scorer import conditional_query


@torch.inference_mode()
def decode_paths(query, times, catalog, lookup, profiles, conditioner, leaf_training_support, config: SearchConfig):
    batch = len(query)
    paths = torch.empty((batch, 1, 0), device=query.device, dtype=torch.long)
    scores = torch.zeros((batch, 1), device=query.device)
    stages = []
    sizes = [len(book) for book in catalog.books]
    children_by_level = catalog._csr_children
    for level in range(3):
        width, size = paths.shape[1], sizes[level]
        prefix = paths.reshape(batch * width, level)
        key = torch.zeros(len(prefix), device=query.device, dtype=torch.long)
        for digit in range(level):
            key = key * sizes[digit] + prefix[:, digit]
        children = children_by_level[level]
        lengths = children.row_ptr[key + 1] - children.row_ptr[key]
        # Match the paper's sparse decoder, including top-k tie ordering.
        slots = max(1, int(lengths.max()))
        position = children.row_ptr[key, None] + torch.arange(slots, device=key.device)[None]
        valid = torch.arange(slots, device=key.device)[None] < lengths[:, None]
        if len(children.digits):
            position = position.clamp(max=len(children.digits) - 1)
            digits = children.digits[position].masked_fill(~valid, 0)
            legal = valid & (children.birth[position] <= times.repeat_interleave(width)[:, None])
        else:
            digits = torch.zeros_like(position)
            legal = torch.zeros_like(valid)
        keys = key[:, None] * size + digits
        descriptors = profiles[level][lookup[level][keys]]
        level_query = query.repeat_interleave(width, 0)
        if level == 2:
            level_query = conditional_query(conditioner, level_query, prefix, lookup, profiles, sizes)
        logits = torch.einsum("bd,bkd->bk", level_query, descriptors) / config.temperature
        if level == 2 and config.new_leaf_bias:
            logits = logits + (~leaf_training_support[lookup[level][keys]]).to(logits.dtype) * config.new_leaf_bias
        active = legal.any(-1)
        logits.masked_fill_(~legal, -torch.inf)
        logits = torch.where(active[:, None], logits, torch.zeros_like(logits))
        if level == 2:
            probabilities = logits.masked_fill(~legal, -torch.inf)
            scores = torch.where(scores.isfinite(), scores * config.coarse_weight, -torch.inf)
        else:
            probabilities = logits.log_softmax(-1).masked_fill(~legal, -torch.inf)
        combined = scores[..., None] + probabilities.reshape(batch, width, slots)
        budget = config.coarse_beam if level < 2 else config.leaf_beam
        scores, chosen = combined.flatten(1).topk(min(budget, width * slots), -1)
        parent = chosen // slots
        digit = digits.reshape(batch, width * slots).gather(1, chosen)
        paths = torch.cat([paths.gather(1, parent[..., None].expand(-1, -1, level)), digit[..., None]], -1)
        stages.append({"paths": paths, "scores": scores})
    return paths, scores, stages


@torch.inference_mode()
def resolve_items(query, times, paths, scores, catalog, vectors, config: SearchConfig):
    buckets = catalog.buckets[catalog.bucket_for_paths(paths)]
    legal = (buckets != 0) & (catalog.first[buckets] <= times[:, None, None]) & scores.isfinite()[..., None]
    raw = torch.einsum("bd,bnkd->bnk", query, vectors[buckets]) / config.temperature
    raw.masked_fill_(~legal, -torch.inf)
    energy = (raw - raw.logsumexp(-1, keepdim=True) + scores[..., None]).masked_fill(~legal, -torch.inf)
    flat = energy.flatten(1)
    count = min(config.top_k, flat.shape[-1])
    top_scores, positions = flat.topk(count, -1)
    items = buckets.flatten(1).gather(1, positions).masked_fill(~top_scores.isfinite(), 0)
    if count < config.top_k:
        items = F.pad(items, (0, config.top_k - count))
    return items
