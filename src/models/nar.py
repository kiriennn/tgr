"""Non-autoregressive (masked-denoising) Semantic ID prediction -- RQ4.

Tests whether generative retrieval *needs* a left-to-right seq2seq
factorisation. Instead of generating c1 -> c2 -> ... -> cL autoregressively, we
train a BERT/MaskGIT-style model that, given the user history and a *corrupted*
target code (some positions replaced by [MASK]), reconstructs all positions
jointly. At inference we start from an all-masked code and read out a
distribution per position in one shot. The headline evaluation enumerates the
highest-probability tuples without a catalog trie, drops invalid codes from the
candidate list, and counts them as a failure mode. Trie-constrained decoding is
kept as a separate engineering ablation.

Architecture: a Transformer encoder over the history tokens + a Transformer
decoder with L learnable position slots that cross-attend to the history; no
causal mask on the decoder, so all positions are predicted in parallel.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from semantic_ids.build import EOS, PAD


class NARDenoiser(nn.Module):
    def __init__(self, sp, d_model=128, num_layers=4, num_heads=4, dropout=0.1,
                 max_items=20):
        super().__init__()
        self.sp = sp
        self.code_len = sp.code_len
        V = sp.vocab_size
        self.MASK = V                      # extra mask token id
        self.tok_emb = nn.Embedding(V + 1, d_model, padding_idx=PAD)
        self.enc_pos = nn.Embedding(max_items * (sp.code_len + 2) + 8, d_model)
        self.slot_emb = nn.Embedding(sp.code_len, d_model)  # target position slots
        enc_layer = nn.TransformerEncoderLayer(d_model, num_heads, d_model * 4,
                                               dropout, batch_first=True,
                                               activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        dec_layer = nn.TransformerDecoderLayer(d_model, num_heads, d_model * 4,
                                               dropout, batch_first=True,
                                               activation="gelu", norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers)
        self.head = nn.Linear(d_model, V)
        # per-position allowed token-id ranges (for masking logits)
        self.pos_lo = sp.pos_offset
        self.pos_hi = [sp.pos_offset[p] + sp.pos_sizes[p] for p in range(sp.code_len)]

    def encode_history(self, enc_ids, enc_mask):
        L = enc_ids.size(1)
        pos = torch.arange(L, device=enc_ids.device).unsqueeze(0)
        h = self.tok_emb(enc_ids) + self.enc_pos(pos)
        return self.encoder(h, src_key_padding_mask=(enc_mask == 0))

    def forward(self, enc_ids, enc_mask, corrupted):
        """corrupted: (B, code_len) target token ids with some == MASK."""
        mem = self.encode_history(enc_ids, enc_mask)
        B = enc_ids.size(0)
        slots = self.slot_emb(torch.arange(self.code_len, device=enc_ids.device)).unsqueeze(0).expand(B, -1, -1)
        q = slots + self.tok_emb(corrupted)
        dec = self.decoder(q, mem, memory_key_padding_mask=(enc_mask == 0))
        logits = self.head(dec)            # (B, code_len, V)
        # restrict each position to its valid token range
        neg = torch.full_like(logits, float("-inf"))
        for p in range(self.code_len):
            neg[:, p, self.pos_lo[p]:self.pos_hi[p]] = logits[:, p, self.pos_lo[p]:self.pos_hi[p]]
        return neg


def nar_loss(model, enc_ids, enc_mask, target, mask_prob=0.6):
    """Masked-denoising loss: randomly mask target positions and reconstruct."""
    B, L = target.shape
    rand = torch.rand(B, L, device=target.device)
    mask = rand < mask_prob
    mask[:, 0] |= (~mask.any(1))           # ensure at least one masked position
    corrupted = target.clone()
    corrupted[mask] = model.MASK
    logits = model(enc_ids, enc_mask, corrupted)
    loss = nn.functional.cross_entropy(
        logits[mask], target[mask], reduction="mean")
    return loss


def _code_from_tokens(sp, toks):
    try:
        return sp._tokens_to_code(toks)
    except Exception:
        return None


@torch.no_grad()
def nar_generate_candidates(model, enc_ids, enc_mask, sp, num_beams=20,
                            constrained=False, device="cpu"):
    """Single-shot all-masked prediction + beam over Semantic-ID positions.

    Works for any model whose ``forward(enc_ids, enc_mask, corrupted)`` returns
    per-position logits of shape (B, code_len, V); the parallel-head baseline
    below ignores ``corrupted``.

    ``constrained=False`` is the NAR analogue of TIGER's paper-style free
    decoding: each position is restricted only to its legal token range, while
    the full tuple may still be off-catalogue. Those invalid tuples are returned
    in ``raw_codes_per_example`` and counted by ``invalid_code_rate``. With
    ``constrained=True`` the catalog trie restricts every prefix to valid item
    IDs, so invalid codes are impossible by construction.
    """
    model.eval()
    B = enc_ids.size(0)
    corrupted = torch.full((B, model.code_len), model.MASK, device=device)
    logprobs = torch.log_softmax(model(enc_ids.to(device), enc_mask.to(device), corrupted), -1)
    items_per_example, raw_codes_per_example = [], []
    for b in range(B):
        beams = [([], 0.0)]                 # (token prefix, logprob)
        for p in range(model.code_len):
            new = []
            if constrained:
                for toks, lp in beams:
                    for t in sp.allowed_tokens(toks):
                        if t == EOS:
                            continue
                        new.append((toks + [t], lp + float(logprobs[b, p, t])))
            else:
                lo, hi = model.pos_lo[p], model.pos_hi[p]
                k = min(num_beams, hi - lo)
                vals, rel_idx = torch.topk(logprobs[b, p, lo:hi], k=k)
                choices = [(int(lo + rel_idx[j]), float(vals[j])) for j in range(k)]
                for toks, lp in beams:
                    for t, tlp in choices:
                        new.append((toks + [t], lp + tlp))
            if not new:
                break
            new.sort(key=lambda x: -x[1])
            beams = new[:num_beams]
        items, seen, raw = [], set(), []
        for toks, _ in beams:
            raw.append(_code_from_tokens(sp, toks))
            it = sp.tokens_to_item(toks)
            if it is not None and it not in seen:
                seen.add(it); items.append(it)
        items_per_example.append(items)
        raw_codes_per_example.append(raw)
    return items_per_example, raw_codes_per_example


@torch.no_grad()
def nar_generate(model, enc_ids, enc_mask, sp, num_beams=20, device="cpu",
                 constrained=False):
    """Backward-compatible wrapper returning only candidate items."""
    items, _ = nar_generate_candidates(
        model, enc_ids, enc_mask, sp, num_beams=num_beams,
        constrained=constrained, device=device)
    return items


class ParallelHeadNAR(nn.Module):
    """Simplest non-autoregressive baseline (instructor's suggestion for RQ4).

    Encode the history, mean-pool it into a single context vector, then predict
    *all* Semantic-ID positions in parallel with ``code_len`` independent linear
    heads. There is no cross-position modelling at all -- this is the natural
    lower bound that isolates how much the masked-denoising decoder's joint
    modelling (and the autoregressive decoder's left-to-right factorisation)
    actually contribute.
    """

    def __init__(self, sp, d_model=128, num_layers=4, num_heads=4, dropout=0.1,
                 max_items=20):
        super().__init__()
        self.sp = sp
        self.code_len = sp.code_len
        V = sp.vocab_size
        self.MASK = V                       # unused, kept for nar_generate API parity
        self.tok_emb = nn.Embedding(V + 1, d_model, padding_idx=PAD)
        self.enc_pos = nn.Embedding(max_items * (sp.code_len + 2) + 8, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, num_heads, d_model * 4,
                                               dropout, batch_first=True,
                                               activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.heads = nn.ModuleList([nn.Linear(d_model, V) for _ in range(sp.code_len)])
        self.pos_lo = sp.pos_offset
        self.pos_hi = [sp.pos_offset[p] + sp.pos_sizes[p] for p in range(sp.code_len)]

    def forward(self, enc_ids, enc_mask, corrupted=None):
        L = enc_ids.size(1)
        pos = torch.arange(L, device=enc_ids.device).unsqueeze(0)
        h = self.tok_emb(enc_ids) + self.enc_pos(pos)
        h = self.encoder(h, src_key_padding_mask=(enc_mask == 0))
        m = enc_mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)   # (B, d_model)
        logits = torch.stack([self.heads[p](pooled) for p in range(self.code_len)], 1)
        neg = torch.full_like(logits, float("-inf"))
        for p in range(self.code_len):
            neg[:, p, self.pos_lo[p]:self.pos_hi[p]] = logits[:, p, self.pos_lo[p]:self.pos_hi[p]]
        return neg


def parallel_loss(model, enc_ids, enc_mask, target):
    """Plain per-position cross-entropy (predict the whole code from history)."""
    logits = model(enc_ids, enc_mask)                       # (B, code_len, V)
    B, Lc, V = logits.shape
    return nn.functional.cross_entropy(logits.reshape(B * Lc, V), target.reshape(B * Lc))
