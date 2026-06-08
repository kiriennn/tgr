"""SASRec (Kang & McAuley, 2018) -- the main sequential baseline.

Self-attention over the item sequence with a causal mask; the representation at
each position predicts the next item via dot product with the (shared) item
embedding table. Trained with full-vocabulary cross-entropy (no sampled
negatives), and evaluated with **full ranking**, so it is directly comparable to
TIGER's reported numbers.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SASRec(nn.Module):
    def __init__(self, num_items, max_len=50, d_model=64, num_layers=2,
                 num_heads=1, dropout=0.2):
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.item_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, seq):
        # seq: (B, L) padded-left item ids
        B, L = seq.shape
        positions = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        h = self.item_emb(seq) + self.pos_emb(positions)
        h = self.drop(h)
        causal = torch.triu(torch.ones(L, L, device=seq.device, dtype=torch.bool), diagonal=1)
        pad_mask = seq == 0
        h = self.encoder(h, mask=causal, src_key_padding_mask=pad_mask)
        h = self.ln(h)
        logits = h @ self.item_emb.weight.t()  # (B, L, num_items+1)
        return logits

    def loss(self, seq, target):
        logits = self.forward(seq)  # (B, L, V)
        V = logits.size(-1)
        loss = nn.functional.cross_entropy(
            logits.view(-1, V), target.view(-1), ignore_index=0)
        return loss

    @torch.no_grad()
    def score_last(self, seq):
        """Scores over all items for the last position of each sequence."""
        logits = self.forward(seq)          # (B, L, V)
        last = logits[:, -1, :]             # (B, V)
        last[:, 0] = float("-inf")          # never recommend padding
        return last
