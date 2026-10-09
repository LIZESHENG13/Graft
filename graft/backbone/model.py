"""Content-only UniSRec history encoder (RUCAIBox/UniSRec, MIT)."""

import torch
from torch import nn
from torch.nn import functional as F

from .layers import SASRec, MoEAdaptorLayer


class UniSRec(SASRec):
    def __init__(self, config, item_embeddings):
        config = dict(config, n_items=len(item_embeddings) - 1)
        super().__init__(config)
        if config["train_stage"] != "inductive_ft":
            raise ValueError("Graft uses the content-only inductive_ft backbone")
        self.item_embedding = None
        self.temperature = config["temperature"]
        self.plm_embedding = nn.Embedding(len(item_embeddings), item_embeddings.shape[1], padding_idx=0)
        self.plm_embedding.weight.requires_grad_(False)
        with torch.no_grad():
            self.plm_embedding.weight.copy_(item_embeddings)
        self.moe_adaptor = MoEAdaptorLayer(config["n_exps"], config["adaptor_layers"], config["adaptor_dropout_prob"])

    def forward(self, item_seq, item_emb, item_seq_len):
        positions = torch.arange(item_seq.shape[1], device=item_seq.device)[None].expand_as(item_seq)
        values = self.dropout(self.LayerNorm(item_emb + self.position_embedding(positions)))
        outputs = self.trm_encoder(values, self.get_attention_mask(item_seq), output_all_encoded_layers=True)
        return self.gather_indexes(outputs[-1], item_seq_len - 1)

    def calculate_loss(self, input_ids, length, labels):
        history = self.moe_adaptor(self.plm_embedding(input_ids))
        query = F.normalize(self.forward(input_ids, history, length), dim=-1)
        items = F.normalize(self.moe_adaptor(self.plm_embedding.weight), dim=-1)
        return self.loss_fct(query @ items.T / self.temperature, labels)
