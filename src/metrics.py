"""Evaluation metrics.

All ranking metrics are computed with **full ranking** over the entire item
catalog (no sampled negatives). This is the single most important correctness
point for sequential-recommendation evaluation: sampled metrics (e.g. ranking
the target against 100 random negatives) are known to be inconsistent and
inflate results (Krichene & Rendle, 2020). TIGER reports full-ranking metrics,
so every model here is evaluated the same way.

Two evaluation styles are supported:

* ``rank_from_scores`` -- for models that produce a score for *every* item
  (e.g. SASRec). The target's rank is its position in the full score ordering.
* ``rank_from_candidates`` -- for generative retrieval (TIGER), which produces
  an *ordered candidate list* via beam search rather than a full score vector.
  The target's rank is its position in that list, or "miss" if absent.

For leave-one-out evaluation there is exactly one ground-truth item per user,
so Recall@K == HitRate@K.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def dcg_at_k(rank: int, k: int) -> float:
    """DCG for a single relevant item at 0-indexed ``rank`` (or -1 if not in list)."""
    if rank < 0 or rank >= k:
        return 0.0
    return 1.0 / math.log2(rank + 2)  # +2 because rank is 0-indexed


def metrics_from_ranks(ranks: Sequence[int], ks=(5, 10, 20)) -> dict:
    """Aggregate Recall@K and NDCG@K from per-user 0-indexed target ranks.

    ``ranks[u]`` is the 0-indexed position of the ground-truth item in user u's
    ranked list, or -1 if the item was not retrieved at all.
    With a single ground-truth item, ideal DCG == 1, so NDCG == DCG.
    """
    ranks = np.asarray(ranks)
    n = len(ranks)
    if n == 0:
        raise ValueError("cannot compute ranking metrics for an empty eval split")
    out = {}
    for k in ks:
        hit = (ranks >= 0) & (ranks < k)
        out[f"recall@{k}"] = float(hit.mean())
        dcg = np.array([dcg_at_k(int(r), k) for r in ranks])
        out[f"ndcg@{k}"] = float(dcg.mean())
    return out


def rank_from_scores(scores: np.ndarray, target: int, exclude: Sequence[int] | None = None) -> int:
    """0-indexed rank of ``target`` given a full score vector (higher = better).

    ``exclude`` optionally removes already-seen items from the ranking. The
    default TIGER/SASRec benchmark protocol does NOT exclude seen items, so the
    caller should pass ``exclude=None`` to match published numbers.
    Ties are broken pessimistically: all non-target items with the same score are
    ranked ahead of the target. This avoids optimistic scores for untrained or
    degenerate models whose item scores are all equal.
    """
    if exclude:
        scores = scores.copy()
        scores[list(exclude)] = -np.inf
    target_score = scores[target]
    # rank = higher-scoring items + tied non-target items
    return int(np.sum(scores > target_score) + np.sum(scores == target_score) - 1)


def rank_from_candidates(candidates: Sequence[int], target: int) -> int:
    """0-indexed rank of ``target`` in an ordered candidate list, else -1."""
    for i, c in enumerate(candidates):
        if c == target:
            return i
    return -1


# --------------------------------------------------------------------------- #
# Generative-retrieval diagnostics
# --------------------------------------------------------------------------- #
def invalid_code_rate(raw_beams: Sequence[Sequence[tuple]], valid_codes: set,
                      limit: int | None = None) -> float:
    """Fraction of generated code tuples that do not map to any catalog item.

    ``raw_beams`` is a list (per user) of lists of generated semantic-ID tuples
    (before mapping to items). Measured under *unconstrained* decoding; with
    trie-constrained decoding this is ~0 by construction. If ``limit`` is set,
    only the first ``limit`` beams for each user are counted; this gives
    ``invalid_code_rate@K`` diagnostics comparable to top-K retrieval plots.

    A ``None`` entry denotes a malformed/short generation (e.g. the model emitted
    fewer than ``code_len`` tokens or an out-of-range token). These MUST count as
    invalid -- dropping them silently deflates the rate. They are therefore
    treated as invalid here, so callers should pass the raw beams unfiltered.
    """
    total = invalid = 0
    for beams in raw_beams:
        for code in beams[:limit]:
            total += 1
            if code is None or tuple(code) not in valid_codes:
                invalid += 1
    return invalid / max(total, 1)


def collision_rate(code_assignments: Sequence[tuple], prefix_len: int | None = None) -> float:
    """Fraction of items whose semantic-ID *prefix* collides with another item.

    With ``prefix_len=None`` the full code (including the disambiguation token)
    is used and the rate is ~0 by design; pass ``prefix_len=L-1`` to measure
    pre-disambiguation collisions, i.e. how often the extra token is actually
    needed.
    """
    from collections import Counter

    keys = [tuple(c[:prefix_len]) if prefix_len else tuple(c) for c in code_assignments]
    counts = Counter(keys)
    colliding = sum(cnt for cnt in counts.values() if cnt > 1)
    return colliding / max(len(keys), 1)
