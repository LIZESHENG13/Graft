"""One shared residual scorer for every leaf, including newly registered paths."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PrefixConditioner(nn.Module):
    def __init__(self, dimension: int = 300, hidden: int = 256):
        super().__init__()
        self.dimension, self.hidden = dimension, hidden
        self.net = nn.Sequential(nn.Linear(3 * dimension + 3, hidden), nn.GELU(), nn.Linear(hidden, dimension))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, query, prefix_vectors, level: int = 2):
        level_id = torch.full(query.shape[:-1], level, device=query.device, dtype=torch.long)
        depth = F.one_hot(level_id, 3).to(query.dtype)
        inputs = torch.cat([query, prefix_vectors, query * prefix_vectors, depth], -1)
        return F.normalize(query + self.net(inputs), dim=-1)


def conditional_query(conditioner, query, prefix, lookup, profiles, sizes, level: int = 2):
    if level == 0:
        prefix_vectors = torch.zeros_like(query)
    else:
        keys = prefix[:, 0]
        for digit in range(1, level):
            keys = keys * sizes[digit] + prefix[:, digit]
        prefix_vectors = profiles[level - 1][lookup[level - 1][keys]]
    return conditioner(query, prefix_vectors, level)


def load_scorer(path, device):
    saved = torch.load(path, map_location=device, weights_only=False)
    arguments = saved.get("model_args", {})
    if arguments.get("coarse_sizes") is not None:
        raise ValueError("The paper scorer has no learned coarse-node biases")
    model = PrefixConditioner(dimension=arguments.get("dimension", 300), hidden=arguments.get("hidden", 256)).to(device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval().requires_grad_(False)
    return model, saved
