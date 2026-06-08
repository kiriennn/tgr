"""TIGER-style generative retrieval model.

A T5 encoder-decoder, **trained from scratch** (random init -- TIGER does not
use pretrained T5 weights; only the *content embeddings* come from Sentence-T5).
The encoder reads the flattened Semantic-ID tokens of the user's history; the
decoder generates the next item's Semantic ID, one code token at a time.

Decoding uses beam search. With ``constrained=True`` a trie restricts generation
to valid Semantic IDs (no invalid codes); with ``constrained=False`` we can
measure the invalid-code rate as a diagnostic.
"""
from __future__ import annotations

import torch
from transformers import T5Config, T5ForConditionalGeneration

from semantic_ids.build import EOS, PAD


def build_tiger_model(vocab_size, d_model=128, d_ff=1024, num_layers=4,
                      num_heads=6, d_kv=64, dropout=0.1):
    cfg = T5Config(
        vocab_size=vocab_size,
        d_model=d_model,
        d_ff=d_ff,
        num_layers=num_layers,
        num_decoder_layers=num_layers,
        num_heads=num_heads,
        d_kv=d_kv,
        dropout_rate=dropout,
        pad_token_id=PAD,
        eos_token_id=EOS,
        decoder_start_token_id=PAD,
        feed_forward_proj="relu",
    )
    return T5ForConditionalGeneration(cfg)


def tiger_loss(model, enc_ids, enc_mask, dec_ids):
    """Cross-entropy next-Semantic-ID loss. Appends EOS to the target."""
    eos = torch.full((dec_ids.size(0), 1), EOS, device=dec_ids.device, dtype=torch.long)
    labels = torch.cat([dec_ids, eos], dim=1)
    out = model(input_ids=enc_ids, attention_mask=enc_mask, labels=labels)
    return out.loss


def _parse_generated_sequence(sequence, sp, code_len):
    """Parse one generated T5 sequence into (code_tokens, code_tuple, item_id).

    HuggingFace returns the decoder-start token first (PAD for this model). Only
    that leading start token may be removed. Generated PAD/EOS inside the code is
    a malformed output and must remain visible to the invalid-code diagnostic.
    """
    toks = [int(t) for t in sequence]
    if toks and toks[0] == PAD:
        toks = toks[1:]
    if EOS in toks:
        toks = toks[:toks.index(EOS)]
    else:
        toks = toks[:code_len]
    if len(toks) != code_len:
        return toks, None, None
    try:
        code = sp._tokens_to_code(toks)
    except Exception:
        code = None
    return toks, code, sp.tokens_to_item(toks)


@torch.no_grad()
def generate_candidates(model, enc_ids, enc_mask, sp, num_beams=20,
                        constrained=False, device="cpu"):
    """Return (items_per_example, raw_codes_per_example).

    items_per_example[b] is an ordered, de-duplicated list of item ids decoded
    from the beams (best first). raw_codes_per_example[b] is the list of decoded
    code tuples (including invalid ones) used for the invalid-code diagnostic.
    """
    model.eval()
    code_len = sp.code_len

    prefix_fn = None
    if constrained:
        def prefix_fn(batch_id, input_ids):
            prefix = input_ids.tolist()[1:]  # drop decoder_start (PAD)
            return sp.allowed_tokens(prefix)

    gen = model.generate(
        input_ids=enc_ids.to(device),
        attention_mask=enc_mask.to(device),
        num_beams=num_beams,
        num_return_sequences=num_beams,
        max_new_tokens=code_len + 1,
        prefix_allowed_tokens_fn=prefix_fn,
        early_stopping=True,
    )
    # gen: (B*num_beams, T) ordered best-first within each example
    gen = gen.view(enc_ids.size(0), num_beams, -1)
    items_per_example, raw_per_example = [], []
    for b in range(gen.size(0)):
        items, seen, raw = [], set(), []
        for beam in gen[b]:
            _, code, item = _parse_generated_sequence(beam.tolist(), sp, code_len)
            raw.append(code)
            if item is not None and item not in seen:
                seen.add(item)
                items.append(item)
        items_per_example.append(items)
        raw_per_example.append(raw)
    return items_per_example, raw_per_example


# --------------------------------------------------------------------------- #
# Time-aware variant (instructor rec #1): add a learned gap-bucket embedding to
# each event's token embeddings, analogous to absolute positional embeddings in
# SASRec, instead of inserting a separate gap token (RQ3).
# --------------------------------------------------------------------------- #
import torch.nn as nn


class TimeAwareTiger(nn.Module):
    def __init__(self, vocab_size, num_gap_buckets=5, d_model=128, **kw):
        super().__init__()
        self.t5 = build_tiger_model(vocab_size, d_model=d_model, **kw)
        # +1 row for the "none" bucket (user token / first item); init to zero so
        # the model starts identical to plain TIGER and learns the gap signal.
        self.gap_emb = nn.Embedding(num_gap_buckets + 1, d_model)
        nn.init.zeros_(self.gap_emb.weight)
        self.code_len_hint = None

    def _embeds(self, enc_ids, gap_ids):
        return self.t5.shared(enc_ids) + self.gap_emb(gap_ids)

    def forward_loss(self, enc_ids, enc_mask, gap_ids, dec_ids):
        eos = torch.full((dec_ids.size(0), 1), EOS, device=dec_ids.device, dtype=torch.long)
        labels = torch.cat([dec_ids, eos], dim=1)
        out = self.t5(inputs_embeds=self._embeds(enc_ids, gap_ids),
                      attention_mask=enc_mask, labels=labels)
        return out.loss


def time_aware_loss(model, enc_ids, enc_mask, gap_ids, dec_ids):
    return model.forward_loss(enc_ids, enc_mask, gap_ids, dec_ids)


@torch.no_grad()
def generate_candidates_timeaware(model, enc_ids, enc_mask, gap_ids, sp,
                                  num_beams=20, constrained=False, device="cpu"):
    """Same as generate_candidates but feeds time-aware encoder embeddings.

    We pre-compute encoder_outputs from inputs_embeds (robust across HF versions)
    and hand them to generate(), so decoding/constraints are identical to the
    plain model.
    """
    model.eval()
    t5 = model.t5
    code_len = sp.code_len
    embeds = model._embeds(enc_ids.to(device), gap_ids.to(device))
    enc_out = t5.get_encoder()(inputs_embeds=embeds,
                               attention_mask=enc_mask.to(device), return_dict=True)

    prefix_fn = None
    if constrained:
        def prefix_fn(batch_id, input_ids):
            return sp.allowed_tokens(input_ids.tolist()[1:])

    gen = t5.generate(
        encoder_outputs=enc_out,
        attention_mask=enc_mask.to(device),
        num_beams=num_beams, num_return_sequences=num_beams,
        max_new_tokens=code_len + 1,
        prefix_allowed_tokens_fn=prefix_fn, early_stopping=True,
    )
    gen = gen.view(enc_ids.size(0), num_beams, -1)
    items_per_example, raw_per_example = [], []
    for b in range(gen.size(0)):
        items, seen, raw = [], set(), []
        for beam in gen[b]:
            _, code, item = _parse_generated_sequence(beam.tolist(), sp, code_len)
            raw.append(code)
            if item is not None and item not in seen:
                seen.add(item); items.append(item)
        items_per_example.append(items); raw_per_example.append(raw)
    return items_per_example, raw_per_example
