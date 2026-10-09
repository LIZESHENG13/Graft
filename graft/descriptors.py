"""Frozen training-node means and registration-time new-leaf exemplars."""

import numpy as np
import torch
from torch.nn import functional as F
from .catalog import SparseLookup


def prototype_tree(catalog, vectors, train_registered, device, return_counts=False):
    ids = np.flatnonzero(catalog.codes[:, 0] >= 0)
    sizes = [len(book) for book in catalog.books]
    lookup, profiles, node_counts = ([], [], [])
    for level in range(3):
        prefix = catalog.codes[ids, : level + 1]
        key = prefix[:, 0].astype(np.int64)
        for l in range(1, level + 1):
            key = key * sizes[l] + prefix[:, l]
        unique, inverse = np.unique(key, return_inverse=True)
        old = train_registered[ids]
        assignment = torch.as_tensor(inverse[old], device=device)
        profile = torch.zeros((len(unique) + 1, vectors.shape[-1]), device=device)
        profile[:-1].index_add_(0, assignment, vectors[torch.as_tensor(ids[old], device=device)])
        counts = torch.zeros(len(unique), device=device)
        counts.index_add_(0, assignment, torch.ones(len(assignment), device=device))
        node_counts.append(torch.cat([counts.clamp(min=1), torch.ones(1, device=device)]))
        profile[:-1] /= counts.clamp(min=1)[:, None]
        new_nodes = torch.nonzero(counts == 0, as_tuple=True)[0].cpu().numpy()
        first_id = {}
        for iid, node in sorted(zip(ids, inverse), key=lambda value: (catalog.first[value[0]], int(value[0]))):
            first_id.setdefault(int(node), int(iid))
        if len(new_nodes):
            exemplar = torch.as_tensor([first_id[int(node)] for node in new_nodes], device=device)
            profile[torch.as_tensor(new_nodes, device=device)] = vectors[exemplar]
        profile = F.normalize(profile, dim=-1)
        if level == 2 and hasattr(catalog, "_prefer_sparse_lookup"):
            table = SparseLookup(
                torch.as_tensor(unique, device=device), torch.arange(len(unique), device=device), len(unique)
            )
        else:
            table = torch.full((int(np.prod(sizes[: level + 1])),), len(unique), device=device, dtype=torch.long)
            table[torch.as_tensor(unique, device=device)] = torch.arange(len(unique), device=device)
        lookup.append(table)
        profiles.append(profile)
    return (lookup, profiles, node_counts) if return_counts else (lookup, profiles)
