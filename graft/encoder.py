"""Trainable-backbone checkpoints and frozen content/history caches."""

from __future__ import annotations

from pathlib import Path
import json
import time
import numpy as np
import torch
from torch.nn import functional as F

from .backbone.model import UniSRec
from .data import load_domain, load_split
from .io import atomic_json, parameter_hash, sha256


class FrozenContentEncoder:
    def __init__(self, checkpoint, content, train_limit, device="cpu"):
        self.device = torch.device(device)
        self.content = torch.as_tensor(np.array(content, copy=True), device=self.device)
        saved = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.model = UniSRec(saved["model_config"], self.content[:train_limit]).to(self.device)
        missing, unexpected = self.model.load_state_dict(saved["model"], strict=False)
        if missing != ["plm_embedding.weight"] or unexpected:
            raise ValueError(f"Unexpected encoder state: {missing}, {unexpected}")
        self.model.eval().requires_grad_(False)
        self.selected_epoch = saved["epoch"]

    @torch.inference_mode()
    def encode_items(self, features):
        raw = self.model.moe_adaptor(torch.as_tensor(features, device=self.device, dtype=self.content.dtype))
        return raw, F.normalize(raw, dim=-1)

    @torch.inference_mode()
    def encode_history(self, right_history, lengths):
        history = torch.as_tensor(right_history, device=self.device, dtype=torch.long)
        lengths = torch.as_tensor(lengths, device=self.device, dtype=torch.long)
        raw = self.model.moe_adaptor(self.content[history])
        return F.normalize(self.model.forward(history, raw, lengths), dim=-1)


def cache_features(config, output, device="cpu", batch_size=2048, encoder_checkpoint=None, splits=("train", "valid")):
    if batch_size < 1 or not splits or set(splits) - {"train", "valid", "test"}:
        raise ValueError("Supply positive batch_size and train/valid/test splits")
    output = Path(output)
    checkpoint = encoder_checkpoint or config.path("encoder_checkpoint")
    checkpoint_hash = sha256(checkpoint)
    metadata = output / "result.json"
    previous = json.loads(metadata.read_text()) if metadata.exists() else {}
    if output.exists() and any(output.iterdir()) and not previous:
        raise FileExistsError(f"Incomplete cache: {output}")
    if previous and previous["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("Existing cache belongs to a different encoder checkpoint")
    train, valid, content, _, _, train_limit, _ = load_domain(config)
    encoder = FrozenContentEncoder(checkpoint, content, train_limit, device)
    before, started = parameter_hash(encoder.model), time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    vectors_path = output / "item_vectors.npy"
    if not vectors_path.exists():
        # The original item adaptor used one batch; retain that operation ordering.
        _, vectors = encoder.encode_items(encoder.content)
        np.save(vectors_path, vectors.cpu().numpy())
    completed = set(previous.get("splits", []))
    for split in splits:
        if split in completed:
            if not (output / f"{split}_queries.npy").exists():
                raise FileNotFoundError(f"Cache metadata lists a missing {split} file")
            continue
        arrays = {"train": train, "valid": valid}.get(split)
        if arrays is None:
            arrays = load_split(config, split=split)
        temporary = output / f"{split}_queries.tmp.npy"
        values = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=np.float16, shape=(len(arrays["item"]), encoder.model.hidden_size)
        )
        for start in range(0, len(values), batch_size):
            stop = min(start + batch_size, len(values))
            query = encoder.encode_history(arrays["right_history"][start:stop], arrays["length"][start:stop])
            values[start:stop] = query.cpu().numpy()
        values.flush()
        del values
        temporary.replace(output / f"{split}_queries.npy")
        completed.add(split)
    if parameter_hash(encoder.model) != before:
        raise AssertionError("Caching changed encoder parameters")
    result = dict(
        status="completed",
        dataset=config.dataset,
        seed=config.seed,
        checkpoint_sha256=checkpoint_hash,
        checkpoint_epoch=encoder.selected_epoch,
        encoder_parameters_frozen=True,
        splits=sorted(completed),
        item_vectors_sha256=sha256(vectors_path),
        query_precision="float16 storage; normalized float32 scoring",
        seconds=round(time.monotonic() - started, 2),
    )
    atomic_json(metadata, result)
    return result
