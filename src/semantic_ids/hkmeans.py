"""Hierarchical k-means Semantic IDs (alternative to RQ-VAE).

Used to answer RQ1: "Is residual quantization necessary, or does the benefit
come simply from assigning *content-based discrete codes* to items?"

We recursively k-means the content embeddings: at each level the points in a
cluster are split into ``branching`` sub-clusters, giving every item a path of
cluster indices of length ``num_levels``. This produces the same code shape as
RQ-VAE (L codes, each in [0, branching)) but with no learned quantizer, so a
head-to-head comparison isolates the effect of residual quantization.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def hierarchical_kmeans(
    embeddings: np.ndarray,
    num_levels: int = 3,
    branching: int = 256,
    seed: int = 42,
    fit_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Return integer codes of shape (N, num_levels).

    If ``fit_indices`` is passed, k-means trees are fitted only on that subset
    and then used to route/code every item. This is needed for temporal
    experiments: the codebook should be learned from the training window, while
    cold/future items may still be assigned content-derived codes.
    """
    n = embeddings.shape[0]
    fit_indices = np.arange(n) if fit_indices is None else np.asarray(fit_indices, dtype=np.int64)
    if len(fit_indices) == 0:
        raise ValueError("fit_indices must contain at least one item")
    if fit_indices.min() < 0 or fit_indices.max() >= n:
        raise ValueError("fit_indices contains out-of-range item indices")

    codes = np.zeros((n, num_levels), dtype=np.int64)
    # groups maps a code-prefix tuple -> (fit item indices, all item indices).
    # The tree is trained on fit indices but predicts/routes all items.
    groups = {(): (fit_indices, np.arange(n))}
    for level in range(num_levels):
        new_groups = {}
        for prefix, (fit_idx, all_idx) in groups.items():
            if len(all_idx) == 0:
                continue
            k = min(branching, len(fit_idx)) if len(fit_idx) else 1
            if k == 1:
                labels_fit = np.zeros(len(fit_idx), dtype=np.int64)
                labels_all = np.zeros(len(all_idx), dtype=np.int64)
            else:
                km = KMeans(n_clusters=k, n_init=4, random_state=seed)
                labels_fit = km.fit_predict(embeddings[fit_idx])
                labels_all = km.predict(embeddings[all_idx])
            codes[all_idx, level] = labels_all
            for lab in range(k):
                sub_fit = fit_idx[labels_fit == lab]
                sub_all = all_idx[labels_all == lab]
                new_groups[prefix + (lab,)] = (sub_fit, sub_all)
        groups = new_groups
    return codes
