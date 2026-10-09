"""Append-only semantic IDs, catalog visibility, and per-prefix trie timestamps."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .io import atomic_json

INF = np.iinfo(np.int64).max


class SparseLookup:
    """An immutable integer map without allocating the Cartesian SID space."""

    def __init__(self, keys, values, fallback):
        self.keys, self.values, self.fallback = keys, values, fallback

    def __getitem__(self, query):
        query = torch.as_tensor(query, device=self.keys.device, dtype=torch.long)
        if not len(self.keys):
            return torch.full_like(query, self.fallback)
        position = torch.searchsorted(self.keys, query.contiguous()).clamp(max=len(self.keys) - 1)
        return torch.where(self.keys[position] == query, self.values[position], torch.full_like(query, self.fallback))


@dataclass
class CSRChildren:
    row_ptr: torch.Tensor
    digits: torch.Tensor
    birth: torch.Tensor


def csr_children(parent, digits, birth, rows, device):
    counts = np.bincount(parent, minlength=rows)
    offsets = np.r_[0, counts.cumsum()]
    return CSRChildren(
        torch.as_tensor(offsets, device=device),
        torch.as_tensor(digits, device=device),
        torch.as_tensor(birth, device=device),
    )


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


@dataclass
class Catalog:
    books: list[np.ndarray]
    codes: np.ndarray
    first: np.ndarray
    code_birth: np.ndarray
    base_sizes: tuple[int, int, int]
    append_trace: list[dict]
    training_seen: np.ndarray

    @classmethod
    def load(cls, path):
        saved = np.load(path)
        trace_path = Path(path).with_suffix(".trace.json")
        trace = json.loads(trace_path.read_text()) if trace_path.exists() else []
        return cls(
            [saved[key] for key in ["c1", "c2", "c3"]],
            saved["codes"],
            saved["first"],
            saved["code_birth"],
            tuple(saved["base_sizes"].tolist()),
            trace,
            saved["training_seen"],
        )

    def torch(self, device, sparse=False):
        n1, n2, n3 = [len(book) for book in self.books]
        available = np.flatnonzero(self.codes[:, 0] >= 0)
        c = self.codes[available].astype(np.int64)
        paths = None
        if not sparse:
            paths = np.full((n1, n2, n3), INF, np.int64)
            np.minimum.at(paths, (c[:, 0], c[:, 1], c[:, 2]), self.first[available])
        keys = (c[:, 0] * n2 + c[:, 1]) * n3 + c[:, 2]
        unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        buckets = np.zeros((len(unique) + 1, max(1, int(counts.max()))), np.int64)
        positions = np.zeros(len(unique), np.int32)
        item_bucket = np.zeros(len(self.codes), np.int64)
        for iid, bid in zip(available, inverse):
            buckets[bid + 1, positions[bid]] = iid
            positions[bid] += 1
            item_bucket[iid] = bid + 1
        if sparse:
            root_birth = np.full(n1, INF, np.int64)
            pair_birth = np.full((n1, n2), INF, np.int64)
            leaf_birth = np.full(len(unique), INF, np.int64)
            np.minimum.at(root_birth, c[:, 0], self.first[available])
            np.minimum.at(pair_birth, (c[:, 0], c[:, 1]), self.first[available])
            np.minimum.at(leaf_birth, inverse, self.first[available])
            sparse_keys = torch.as_tensor(unique, device=device)
            flat_bucket = SparseLookup(sparse_keys, torch.arange(1, len(unique) + 1, device=device), 0)
        else:
            root_birth, pair_birth = paths.min(axis=(1, 2)), paths.min(axis=2)
            flat = np.zeros(n1 * n2 * n3, np.int64)
            flat[unique] = np.arange(1, len(unique) + 1)
            flat_bucket = torch.as_tensor(flat, device=device)
        result = TorchCatalog(
            books=[torch.as_tensor(book, device=device) for book in self.books],
            codes=torch.as_tensor(self.codes, device=device, dtype=torch.long),
            first=torch.as_tensor(self.first, device=device),
            code_birth=torch.as_tensor(self.code_birth, device=device),
            first_layer_birth=torch.as_tensor(root_birth, device=device),
            second_layer_birth=torch.as_tensor(pair_birth, device=device),
            third_layer_birth=None if sparse else torch.as_tensor(paths, device=device),
            buckets=torch.as_tensor(buckets, device=device),
            item_bucket=torch.as_tensor(item_bucket, device=device),
            flat_bucket=flat_bucket,
            base_sizes=self.base_sizes,
        )
        if sparse:
            result._sparse_leaf_birth = SparseLookup(sparse_keys, torch.as_tensor(leaf_birth, device=device), INF)
            roots = np.flatnonzero(root_birth < INF)
            pairs = np.flatnonzero(pair_birth.reshape(-1) < INF)
            result._csr_children = [
                csr_children(np.zeros(len(roots), np.int64), roots, root_birth[roots], 1, device),
                csr_children(pairs // n2, pairs % n2, pair_birth.reshape(-1)[pairs], n1, device),
                csr_children(unique // n3, unique % n3, leaf_birth, n1 * n2, device),
            ]
        return result

    def save(self, path):
        np.savez(
            path,
            c1=self.books[0],
            c2=self.books[1],
            c3=self.books[2],
            codes=self.codes,
            first=self.first,
            code_birth=self.code_birth,
            base_sizes=np.asarray(self.base_sizes),
            training_seen=self.training_seen,
        )
        atomic_json(Path(path).with_suffix(".trace.json"), self.append_trace)


@dataclass
class TorchCatalog:
    books: list[torch.Tensor]
    codes: torch.Tensor
    first: torch.Tensor
    code_birth: torch.Tensor
    first_layer_birth: torch.Tensor
    second_layer_birth: torch.Tensor
    third_layer_birth: torch.Tensor
    buckets: torch.Tensor
    item_bucket: torch.Tensor
    flat_bucket: torch.Tensor
    base_sizes: tuple[int, int, int]

    def valid_vocab(self, level, times):
        if level < 2:
            return torch.ones((len(times), len(self.books[level])), dtype=torch.bool, device=times.device)
        return self.code_birth[None] <= times[:, None]

    def valid_children(self, level, prefix, times):
        if level == 0:
            births = self.first_layer_birth[None].expand(len(times), -1)
        elif level == 1:
            births = self.second_layer_birth[prefix[:, 0]]
        else:
            if self.third_layer_birth is None:
                n2, n3 = len(self.books[1]), len(self.books[2])
                parent = (prefix[:, 0] * n2 + prefix[:, 1]) * n3
                keys = parent[:, None] + torch.arange(n3, device=times.device)[None]
                births = self._sparse_leaf_birth[keys]
            else:
                births = self.third_layer_birth[prefix[:, 0], prefix[:, 1]]
        return births <= times[:, None]

    def bucket_for_paths(self, paths):
        _, n2, n3 = [len(book) for book in self.books]
        keys = (paths[..., 0] * n2 + paths[..., 1]) * n3 + paths[..., 2]
        return self.flat_bucket[keys]
