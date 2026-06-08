"""Datasets and leave-one-out splitting shared by all models.

A user sequence ``s = [i_1, ..., i_n]`` is split as:
  test target  = i_n      , history = i_1..i_{n-1}
  val  target  = i_{n-1}  , history = i_1..i_{n-2}
  train        = all next-item examples within i_1..i_{n-2}

For training we use the standard "all prefixes" augmentation: every position t
in the training portion yields one example (history up to t -> i_{t+1}).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def bucket_gap(seconds: float) -> int:
    """Discretise an inter-event gap into a coarse bucket (RQ3)."""
    day = 86400
    if seconds < day:           return 0   # same day / < 1 day
    if seconds < 7 * day:       return 1   # < 1 week
    if seconds < 30 * day:      return 2   # < 1 month
    if seconds < 182 * day:     return 3   # < 6 months
    return 4                                # > 6 months


def split_sequences(sequences: dict, timestamps: dict | None = None):
    """Return (train_examples, val, test).

    Without timestamps, examples are (user, history_items, target).
    With timestamps, examples carry parallel times: (user, items, times, target).
    """
    train, val, test = [], {}, {}
    for u, seq in sequences.items():
        u = int(u)
        if len(seq) < 3:
            continue
        ts = timestamps[str(u)] if timestamps else None
        train_part = seq[:-2]
        if timestamps:
            for t in range(1, len(train_part)):
                train.append((u, train_part[:t], ts[:t], train_part[t]))
            val[u] = (seq[:-2], ts[:-2], seq[-2])
            test[u] = (seq[:-1], ts[:-1], seq[-1])
        else:
            for t in range(1, len(train_part)):
                train.append((u, train_part[:t], train_part[t]))
            val[u] = (seq[:-2], seq[-2])
            test[u] = (seq[:-1], seq[-1])
    return train, val, test


def temporal_cutoffs(timestamps: dict, val_frac: float = 0.1, test_frac: float = 0.1):
    """Return the global validation/test timestamp cutoffs for temporal split."""
    all_ts = sorted(t for ts in timestamps.values() for t in ts)
    if not all_ts:
        raise ValueError("temporal split requires timestamps")
    val_idx = int(len(all_ts) * (1 - val_frac - test_frac))
    test_idx = int(len(all_ts) * (1 - test_frac))
    val_idx = min(max(val_idx, 0), len(all_ts) - 1)
    test_idx = min(max(test_idx, 0), len(all_ts) - 1)
    return all_ts[val_idx], all_ts[test_idx]


def temporal_kwargs_from_meta(meta: dict | None) -> dict:
    """Return fixed temporal cutoffs stored by preprocessing, if present."""
    if not meta:
        return {}
    if "temporal_val_cut" not in meta or "temporal_test_cut" not in meta:
        return {}
    return {
        "val_cut": int(meta["temporal_val_cut"]),
        "test_cut": int(meta["temporal_test_cut"]),
    }


def split_sequences_temporal(sequences: dict, timestamps: dict,
                             val_frac: float = 0.1, test_frac: float = 0.1,
                             val_cut: int | None = None,
                             test_cut: int | None = None):
    """Global **time-based** split (instructor's recommended evaluation).

    Unlike leave-one-out -- which holds out each user's last interaction
    regardless of when it happened and can leak future information across users
    -- this splits by absolute time. Two global timestamp cutoffs are chosen at
    the ``1-val_frac-test_frac`` and ``1-test_frac`` quantiles of all
    interactions:

      * training        : per-user next-item examples among items before the
                          validation cutoff;
      * validation tgt  : the user's first interaction in [val_cut, test_cut),
                          predicted from everything they did before it;
      * test target     : the user's first interaction at/after test_cut,
                          predicted from everything they did before it.

    History always precedes the target in wall-clock time, so there is no
    future leakage. Returns the same structure as ``split_sequences`` (examples
    carry parallel timestamps).
    """
    if val_cut is None or test_cut is None:
        val_cut, test_cut = temporal_cutoffs(timestamps, val_frac, test_frac)
    else:
        val_cut, test_cut = int(val_cut), int(test_cut)

    train, val, test = [], {}, {}
    for u, seq in sequences.items():
        u = int(u)
        ts = timestamps[str(u)]
        if len(seq) < 2:
            continue
        # training next-item examples among the pre-validation items
        tr_items = [it for it, t in zip(seq, ts) if t < val_cut]
        tr_times = [t for t in ts if t < val_cut]
        for k in range(1, len(tr_items)):
            train.append((u, tr_items[:k], tr_times[:k], tr_items[k]))
        # first validation / test interaction (history = everything earlier)
        for j, (it, t) in enumerate(zip(seq, ts)):
            if val_cut <= t < test_cut and j >= 1 and u not in val:
                val[u] = (seq[:j], ts[:j], it)
            elif t >= test_cut and j >= 1 and u not in test:
                test[u] = (seq[:j], ts[:j], it)
    return train, val, test


def items_in_training_window(sequences: dict, timestamps: dict | None = None,
                             split: str = "leave_one_out",
                             val_frac: float = 0.1, test_frac: float = 0.1,
                             val_cut: int | None = None,
                             test_cut: int | None = None) -> set[int]:
    """Items that are actually visible to the training split.

    This differs from ``items_in_training_examples``: a single pre-cutoff event
    cannot form a next-item training pair, but the item is still observed in the
    train window and should not be counted as cold.
    """
    seen = set()
    if split == "leave_one_out":
        for seq in sequences.values():
            if len(seq) >= 3:
                seen.update(int(i) for i in seq[:-2])
        return seen
    if split == "temporal":
        if timestamps is None:
            raise ValueError("temporal training-window items require timestamps")
        if val_cut is None:
            val_cut, _ = temporal_cutoffs(timestamps, val_frac, test_frac)
        else:
            val_cut = int(val_cut)
        for u, seq in sequences.items():
            key = str(u)
            ts = timestamps[key] if key in timestamps else timestamps[u]
            for item, t in zip(seq, ts):
                if t < val_cut:
                    seen.add(int(item))
        return seen
    raise ValueError(f"unknown split={split}")


def items_in_training_examples(train_examples) -> set[int]:
    """Items observed in the training window.

    Useful for temporal-split diagnostics: if a validation/test target never
    appears before the cutoff, SASRec has no interaction-trained signal for that
    item, while content-code models may still be able to generate its Semantic
    ID. Reporting this rate keeps quality comparisons interpretable.
    """
    seen = set()
    for ex in train_examples:
        hist, tgt = ex[1], ex[-1]
        seen.update(int(i) for i in hist)
        seen.add(int(tgt))
    return seen


def target_coverage(eval_dict: dict, seen_items: set[int]) -> dict:
    """Return seen/cold target diagnostics for an eval split."""
    targets = [int(v[-1]) for v in eval_dict.values()]
    if not targets:
        return {"num_targets": 0, "seen_target_rate": 0.0, "cold_target_rate": 0.0}
    seen = sum(t in seen_items for t in targets)
    return {
        "num_targets": len(targets),
        "seen_target_rate": seen / len(targets),
        "cold_target_rate": 1.0 - seen / len(targets),
    }


class TigerDataset(Dataset):
    """Yields (encoder_input_ids, decoder_target_ids) of token ids.

    Each example is (user, history_items, target) or, when time gaps are used,
    (user, history_items, history_times, target). With ``use_time_gaps`` the
    encoder input interleaves a gap token between consecutive items (RQ3).
    """

    def __init__(self, examples, id_space, max_items=20, add_user_token=True,
                 use_time_gaps=False, use_time_embed=False, num_gap_buckets=5):
        self.examples = examples
        self.sp = id_space
        self.max_items = max_items
        self.add_user_token = add_user_token and id_space.num_user_tokens > 0
        self.use_time_gaps = use_time_gaps and id_space.num_gap_tokens > 0
        # time-aware embedding (instructor rec #1): emit a parallel per-token
        # gap-bucket id so the model can ADD a learned gap embedding to each
        # event's token embeddings, instead of inserting a separate gap token.
        self.use_time_embed = use_time_embed
        self.num_gap_buckets = num_gap_buckets
        self.none_bucket = num_gap_buckets          # id for user token / first item
        self.code_len = id_space.code_len

    def __len__(self):
        return len(self.examples)

    def _encode_history(self, user, history, times=None):
        history = history[-self.max_items:]
        if times is not None:
            times = times[-self.max_items:]
        toks, gaps = [], []
        if self.add_user_token:
            toks.append(self.sp.user_token(user)); gaps.append(self.none_bucket)
        for k, item in enumerate(history):
            code = self.sp.item_tokens[item - 1].tolist()  # 1-based ids
            if self.use_time_embed and times is not None and k > 0:
                b = bucket_gap(times[k] - times[k - 1])     # gap BEFORE this item
            else:
                b = self.none_bucket
            toks.extend(code); gaps.extend([b] * len(code))
            if self.use_time_gaps and times is not None and k < len(history) - 1:
                gap = bucket_gap(times[k + 1] - times[k])
                toks.append(self.sp.gap_token(gap)); gaps.append(self.none_bucket)
        return toks, gaps

    def __getitem__(self, idx):
        ex = self.examples[idx]
        if len(ex) == 4:
            user, history, times, target = ex
        else:
            user, history, target = ex
            times = None
        enc, gaps = self._encode_history(user, history, times)
        dec = self.sp.item_tokens[target - 1].tolist()
        if self.use_time_embed:
            return (torch.tensor(enc, dtype=torch.long),
                    torch.tensor(dec, dtype=torch.long),
                    torch.tensor(gaps, dtype=torch.long))
        return torch.tensor(enc, dtype=torch.long), torch.tensor(dec, dtype=torch.long)


def tiger_collate(batch, pad_id=0):
    has_gaps = len(batch[0]) == 3
    if has_gaps:
        encs, decs, gaps = zip(*batch)
    else:
        encs, decs = zip(*batch)
    maxlen = max(len(e) for e in encs)
    enc_ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
    enc_mask = torch.zeros((len(batch), maxlen), dtype=torch.long)
    for i, e in enumerate(encs):
        enc_ids[i, : len(e)] = e
        enc_mask[i, : len(e)] = 1
    dec_ids = torch.stack(decs)  # all same length (code_len)
    if has_gaps:
        none_b = int(max(g.max().item() for g in gaps)) if gaps else 0
        gap_ids = torch.full((len(batch), maxlen), none_b, dtype=torch.long)
        for i, g in enumerate(gaps):
            gap_ids[i, : len(g)] = g
        return enc_ids, enc_mask, dec_ids, gap_ids
    return enc_ids, enc_mask, dec_ids


class SasrecDataset(Dataset):
    """Causal next-item training. Returns (input_seq, target_seq) padded left."""

    def __init__(self, examples_by_user: dict, num_items, max_len=50):
        # build one training sequence per user from the augmented prefixes is wasteful;
        # SASRec trains on the whole train sequence with shifted targets instead.
        self.data = []
        self.num_items = num_items
        self.max_len = max_len
        for u, seq in examples_by_user.items():
            if len(seq) < 2:
                continue
            self.data.append(seq[-(max_len + 1):])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        inp = seq[:-1]
        tgt = seq[1:]
        pad = self.max_len - len(inp)
        inp = [0] * pad + inp
        tgt = [0] * pad + tgt
        return (torch.tensor(inp, dtype=torch.long), torch.tensor(tgt, dtype=torch.long))
