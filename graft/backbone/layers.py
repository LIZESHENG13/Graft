"""UniSRec adaptor and SASRec sequence layers, with checkpoint-compatible names.

Adapted from RUCAIBox/UniSRec and RUCAIBox/RecBole (MIT).
See licenses/UniSRec-MIT.txt and licenses/RecBole-MIT.txt.
"""

import torch
from torch import nn
from torch.nn import functional as F
from recbole.model.layers import TransformerEncoder


class PWLayer(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.zeros(input_size))
        self.lin = nn.Linear(input_size, output_size, bias=False)
        nn.init.normal_(self.lin.weight, std=0.02)

    def forward(self, x):
        return self.lin(self.dropout(x) - self.bias)


class MoEAdaptorLayer(nn.Module):
    def __init__(self, n_exps, layers, dropout=0.0, noise=True):
        super().__init__()
        self.n_exps, self.noisy_gating = n_exps, noise
        self.experts = nn.ModuleList(PWLayer(layers[0], layers[1], dropout) for _ in range(n_exps))
        self.w_gate = nn.Parameter(torch.zeros(layers[0], n_exps))
        self.w_noise = nn.Parameter(torch.zeros(layers[0], n_exps))

    def forward(self, x):
        logits = x @ self.w_gate
        if self.noisy_gating and self.training:
            logits = logits + torch.randn_like(logits) * (F.softplus(x @ self.w_noise) + 0.01)
        gates = logits.softmax(dim=-1)
        experts = torch.stack([expert(x) for expert in self.experts], dim=-2)
        return (gates.unsqueeze(-1) * experts).sum(-2)


class SASRec(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.max_seq_length = config["max_seq_length"]
        self.initializer_range = config["initializer_range"]
        self.item_embedding = nn.Embedding(config["n_items"], self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=config["n_layers"],
            n_heads=config["n_heads"],
            hidden_size=self.hidden_size,
            inner_size=config["inner_size"],
            hidden_dropout_prob=config["hidden_dropout_prob"],
            attn_dropout_prob=config["attn_dropout_prob"],
            hidden_act=config["hidden_act"],
            layer_norm_eps=config["layer_norm_eps"],
        )
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=config["layer_norm_eps"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
        self.loss_fct = nn.CrossEntropyLoss()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def get_attention_mask(self, item_seq):
        valid = (item_seq > 0).unsqueeze(1).unsqueeze(2)
        causal = torch.tril(torch.ones(item_seq.shape[-1], item_seq.shape[-1], device=item_seq.device)).bool()
        valid = valid & causal[None, None]
        return (1.0 - valid.to(next(self.parameters()).dtype)) * -10000.0

    @staticmethod
    def gather_indexes(output, positions):
        index = positions[:, None, None].expand(-1, 1, output.shape[-1])
        return output.gather(1, index).squeeze(1)
