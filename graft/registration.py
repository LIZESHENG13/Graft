"""Chronological fresh-path admission under fixed, existing two-digit prefixes."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from .catalog import Catalog
from .descriptors import prototype_tree


@torch.inference_mode()
def register_arrivals(
    base: Catalog, item_vectors: np.ndarray, cutoff: int, stable_order: np.ndarray, budget: int, device="cpu"
) -> Catalog:
    if budget < 0:
        raise ValueError("New-leaf budget must be nonnegative")
    arrays = np.asarray(item_vectors)
    if len(arrays) < len(base.codes):
        arrays = np.pad(arrays, ((0, len(base.codes) - len(arrays)), (0, 0)))
    vectors = F.normalize(torch.as_tensor(arrays, device=device), dim=-1)
    old = base.codes[:, 0] >= 0
    old_ids = np.flatnonzero(old)
    if not len(old_ids):
        raise ValueError("Registration requires an existing semantic catalog")
    lookup, profiles = prototype_tree(base, vectors, old, device)
    sizes = base.base_sizes
    pairs = np.unique(base.codes[old_ids, :2], axis=0)
    pair_keys = pairs[:, 0].astype(np.int64) * sizes[1] + pairs[:, 1]
    pair_vectors = profiles[1][lookup[1][torch.as_tensor(pair_keys, device=device)]]
    roots = profiles[0][lookup[0]]
    arrivals = np.flatnonzero((base.first <= cutoff) & ~old)
    arrivals = arrivals[np.lexsort((stable_order[arrivals], base.first[arrivals]))]
    coarse = np.full((len(base.codes), 2), -1, np.int32)
    for begin in range(0, len(arrivals), 256):
        ids = arrivals[begin : begin + 256]
        query = vectors[torch.as_tensor(ids, device=device)]
        scores = query @ pair_vectors.T / 0.07
        parents = torch.as_tensor(pairs[:, 0], device=device)[None].expand(len(ids), -1)
        scores.masked_fill_(parents != (query @ roots.T).argmax(-1)[:, None], -torch.inf)
        coarse[ids] = pairs[scores.argmax(-1).cpu().numpy()]
    leaves = {}
    old_codes = base.codes[old_ids]
    for pair in pairs:
        third = np.unique(old_codes[(old_codes[:, :2] == pair).all(-1), 2])
        keys = (int(pair[0]) * sizes[1] + int(pair[1])) * sizes[2] + third
        leaves[tuple(pair)] = {int(token): profiles[2][lookup[2][int(key)]] for token, key in zip(third, keys)}
    codes, trace, admitted = base.codes.copy(), [], 0
    for iid in arrivals:
        pair = coarse[iid]
        branch = leaves[tuple(pair)]
        tokens = list(branch)
        error = (vectors[iid] - torch.stack(list(branch.values()))).square().sum(-1)
        nearest = int(error.argmin())
        token, novelty = tokens[nearest], float(error[nearest])
        fresh = admitted < budget
        if fresh:
            unused = [digit for digit in range(sizes[2]) if digit not in branch]
            if unused:
                residual = arrays[iid] - base.books[0][pair[0]] - base.books[1][pair[1]]
                selected = ((base.books[2][unused] - residual) ** 2).sum(-1).argmin()
                token = unused[int(selected)]
                branch[token] = vectors[iid].clone()
                admitted += 1
            else:
                fresh = False
        codes[iid] = [*pair, token]
        trace.append(
            {
                "item_id": int(iid),
                "timestamp": int(base.first[iid]),
                "sid": codes[iid].tolist(),
                "appended": False,
                "fresh_path_admitted": fresh,
                "novelty": novelty,
                "codebook_size": sizes[2],
            }
        )
    np.testing.assert_array_equal(codes[old], base.codes[old])
    return Catalog(
        [book.copy() for book in base.books],
        codes,
        base.first.copy(),
        base.code_birth.copy(),
        sizes,
        trace,
        base.training_seen.copy(),
    )
