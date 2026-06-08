"""Regression tests for the correctness bugs found in the code audit.

Run:  python3 tests/test_correctness.py   (plain asserts, no pytest needed)
"""
import os, sys
import tempfile
import inspect
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from semantic_ids.build import SemanticIDSpace
from semantic_ids.hkmeans import hierarchical_kmeans
from semantic_ids.io import (integer_set_fingerprint, item_text_fingerprint,
                             load_codes_validated, save_json, sidecar_path)
from metrics import (invalid_code_rate, collision_rate, rank_from_scores,
                     metrics_from_ranks)
from models.nar import nar_generate_candidates
from models.nar import nar_generate
from models.tiger import (_parse_generated_sequence, generate_candidates,
                          generate_candidates_timeaware)
from data.dataset import (items_in_training_examples, items_in_training_window,
                          split_sequences_temporal, target_coverage,
                          temporal_kwargs_from_meta)
from data.preprocess import apply_train_temporal_kcore
from semantic_ids.content import build_item_text


def test_tokens_to_item_roundtrip():
    """Every item id must round-trip through its tokens (1-based ids)."""
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 8, size=(50, 3))
    codes[10] = codes[3]            # force a collision
    codes[20] = codes[3]
    for collision_first in (False, True):
        sp = SemanticIDSpace(codes, codebook_size=8, num_user_tokens=16,
                             collision_first=collision_first)
        for item in range(1, len(codes) + 1):
            toks = sp.item_tokens[item - 1].tolist()
            assert sp.tokens_to_item(toks) == item, (
                f"round-trip failed for item {item} (collision_first={collision_first})")
    print("OK  tokens_to_item round-trips to 1-based item ids")


def test_invalid_code_rate_counts_none():
    """None / malformed generations must count as invalid, not be dropped."""
    valid = {(0, 0, 0, 0)}
    beams = [[(0, 0, 0, 0), None, (9, 9, 9, 9)]]  # 1 valid, 1 short, 1 off-catalogue
    rate = invalid_code_rate(beams, valid)
    assert abs(rate - 2 / 3) < 1e-9, f"expected 2/3, got {rate}"
    assert abs(invalid_code_rate(beams, valid, limit=2) - 0.5) < 1e-9
    print("OK  invalid_code_rate counts None as invalid")


def test_raw_codes_are_validated():
    try:
        SemanticIDSpace(np.array([[0, 2]]), codebook_size=2, num_user_tokens=0)
    except ValueError:
        print("OK  SemanticIDSpace rejects out-of-range raw codes")
        return
    raise AssertionError("out-of-range raw code should raise ValueError")


def test_pre_disambiguation_collision_metric():
    """Pre-disambiguation collision rate is computed on content codes and is
    invariant to collision-token placement."""
    codes = np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6]])  # 2/3 items collide
    sp0 = SemanticIDSpace(codes, codebook_size=8, num_user_tokens=4, collision_first=False)
    sp1 = SemanticIDSpace(codes, codebook_size=8, num_user_tokens=4, collision_first=True)
    r0 = collision_rate(sp0.content_codes)
    r1 = collision_rate(sp1.content_codes)
    assert abs(r0 - 2 / 3) < 1e-9 and abs(r1 - 2 / 3) < 1e-9, (r0, r1)
    print("OK  pre-disambiguation collision metric correct under --collision-first")


def test_disambiguation_position_uses_full_codebook_slice():
    """The appended collision token should get a full K-sized token slice.

    Otherwise TIGER/NAR are evaluated with a smaller-than-paper output space and
    invalid-code rates become artificially low.
    """
    codes = np.array([[0, 0], [1, 1], [2, 2]])
    sp = SemanticIDSpace(codes, codebook_size=8, num_user_tokens=4)
    assert sp.max_collisions == 1
    assert sp.disambiguation_size == 8
    assert sp.pos_sizes == [8, 8, 8]
    assert sp.vocab_size == 3 + 4 + 8 * 3
    print("OK  disambiguation token uses a full codebook-sized vocabulary slice")


def test_temporal_disambiguation_priority_prevents_future_code_shift():
    """Future/cold catalogue items must not change train-window target codes.

    Item 1 is treated as future/cold, item 2 as train-window. They share the same
    content code. Without priority, item 2 would get disambiguation token 1 just
    because future item 1 appears earlier in catalogue order.
    """
    codes = np.array([[1, 2], [1, 2], [3, 0]])
    sp = SemanticIDSpace(codes, codebook_size=4, num_user_tokens=0,
                         disambiguation_priority=[2, 3])
    assert sp.full_codes[1].tolist() == [1, 2, 0]
    assert sp.full_codes[0].tolist() == [1, 2, 1]
    assert sp.tokens_to_item(sp.item_tokens[1].tolist()) == 2

    sp_first = SemanticIDSpace(codes, codebook_size=4, num_user_tokens=0,
                               collision_first=True,
                               disambiguation_priority=[2, 3])
    assert sp_first.full_codes[1].tolist() == [0, 1, 2]
    assert sp_first.full_codes[0].tolist() == [1, 1, 2]
    assert sp_first.tokens_to_item(sp_first.item_tokens[1].tolist()) == 2
    print("OK  temporal disambiguation priority prevents future code shifts")


def test_rank_from_scores_breaks_ties_pessimistically():
    scores = np.ones(6)
    scores[0] = -np.inf
    assert rank_from_scores(scores, target=5) == 4
    assert metrics_from_ranks([4], ks=(5, 10))["recall@5"] == 1.0
    assert metrics_from_ranks([4], ks=(4,))["recall@4"] == 0.0
    print("OK  rank_from_scores uses pessimistic tie handling")


def test_empty_rank_metrics_raise():
    try:
        metrics_from_ranks([], ks=(5,))
    except ValueError:
        print("OK  empty eval splits fail explicitly")
        return
    raise AssertionError("metrics_from_ranks([]) should raise ValueError")


def test_generated_pad_inside_code_is_invalid():
    """Only the decoder-start PAD may be stripped; generated PAD is malformed."""
    codes = np.array([[0, 0], [0, 1]])
    sp = SemanticIDSpace(codes, codebook_size=2, num_user_tokens=0)
    valid = sp.item_tokens[0].tolist()
    malformed = [0, valid[0], 0, valid[1], valid[2], 1]  # start PAD, then generated PAD
    _, code, item = _parse_generated_sequence(malformed, sp, sp.code_len)
    assert item is None and code not in sp.valid_codes, (code, item)
    print("OK  generated PAD inside a Semantic ID is counted as invalid")


def test_nar_unconstrained_counts_off_catalogue_codes():
    """NAR headline decoding must expose invalid tuples; constrained is ablation."""
    codes = np.array([[0, 0], [1, 1]])
    sp = SemanticIDSpace(codes, codebook_size=2, num_user_tokens=0)

    class FixedNAR:
        code_len = sp.code_len
        MASK = sp.vocab_size
        pos_lo = sp.pos_offset
        pos_hi = [sp.pos_offset[p] + sp.pos_sizes[p] for p in range(sp.code_len)]

        def eval(self):
            return self

        def __call__(self, enc_ids, enc_mask, corrupted):
            logits = torch.full((enc_ids.size(0), sp.code_len, sp.vocab_size), -100.0)
            # Highest unconstrained tuple is (0, 1, 0), which is off-catalogue.
            prefs = [0, 1, 0]
            alts = [1, 0, 0]
            for p, (pref, alt) in enumerate(zip(prefs, alts)):
                logits[:, p, sp.pos_offset[p] + pref] = 10.0
                logits[:, p, sp.pos_offset[p] + alt] = 9.0
            return logits

    enc = torch.tensor([[sp.item_tokens[0, 0]]], dtype=torch.long)
    mask = torch.ones_like(enc)
    _, raw_free = nar_generate_candidates(
        FixedNAR(), enc, mask, sp, num_beams=2, constrained=False)
    _, raw_cons = nar_generate_candidates(
        FixedNAR(), enc, mask, sp, num_beams=2, constrained=True)
    assert invalid_code_rate(raw_free, sp.valid_codes) > 0.0
    assert invalid_code_rate(raw_cons, sp.valid_codes) == 0.0
    print("OK  NAR unconstrained decoding exposes off-catalogue codes")


def test_target_coverage_diagnostic():
    train = [(1, [1, 2], 3), (2, [2], 4)]
    seen = items_in_training_examples(train)
    cov = target_coverage({1: ([1], 3), 2: ([1], 5)}, seen)
    assert cov["num_targets"] == 2
    assert abs(cov["seen_target_rate"] - 0.5) < 1e-9
    assert abs(cov["cold_target_rate"] - 0.5) < 1e-9
    print("OK  target coverage diagnostics flag cold eval targets")


def test_temporal_training_window_keeps_singletons():
    """Seen-item diagnostics / fit-split must use the train window, not only
    next-item examples. A single pre-cutoff event is observed but cannot form a
    prefix->target training pair."""
    seq = {
        "1": [10, 20, 30],
        "2": [40, 50, 60],
        "3": [70, 80, 90],
        "4": [100, 110, 120],
    }
    ts = {
        "1": [1, 100, 200],
        "2": [2, 101, 201],
        "3": [3, 102, 202],
        "4": [4, 180, 220],
    }
    seen = items_in_training_window(seq, ts, split="temporal",
                                    val_frac=0.25, test_frac=0.25)
    assert seen == {10, 20, 40, 50, 70, 100}
    print("OK  temporal training-window items include singleton pre-cutoff events")


def test_train_temporal_kcore_uses_only_train_window():
    """Temporal preprocessing must not let future-only activity decide k-core."""
    interactions = [
        ("u1", "a", 1),
        ("u1", "b", 2),
        ("u2", "a", 3),
        ("u2", "b", 4),
        ("future_only", "c", 5),
        ("future_only", "d", 6),
        ("future_only", "c", 100),
        ("future_only", "d", 101),
        ("u1", "cold_future", 102),
        ("u4", "future_item", 103),
    ]
    kept, meta = apply_train_temporal_kcore(interactions, k=2, val_frac=0.3, test_frac=0.2)
    kept_users = {u for u, _, _ in kept}
    kept_rows = set(kept)
    assert meta["temporal_val_cut"] == 5 and meta["temporal_test_cut"] == 102
    assert meta["train_kcore_num_users"] == 2
    assert meta["train_kcore_num_items"] == 2
    assert "future_only" not in kept_users and "u4" not in kept_users
    assert ("u1", "cold_future", 102) in kept_rows
    print("OK  train_temporal k-core is fitted only on the train window")


def test_temporal_split_uses_fixed_meta_cutoffs():
    seq = {
        "1": [1, 2, 3, 4],
        "2": [5, 6, 7, 8],
    }
    ts = {
        "1": [1, 2, 100, 200],
        "2": [1, 50, 60, 300],
    }
    kwargs = temporal_kwargs_from_meta({"temporal_val_cut": 100, "temporal_test_cut": 200})
    train, val, test = split_sequences_temporal(seq, ts, **kwargs)
    assert val == {1: ([1, 2], [1, 2], 3)}
    assert test[1] == ([1, 2, 3], [1, 2, 100], 4)
    assert test[2] == ([5, 6, 7], [1, 50, 60], 8)
    assert len(train) == 3
    print("OK  temporal split uses fixed cutoffs stored in metadata")


def test_hkmeans_fit_subset_codes_all_items():
    """Temporal mode fits the tree on train-window items but still codes all items."""
    emb = np.array([[0.0], [0.1], [10.0], [10.1]], dtype=np.float32)
    codes = hierarchical_kmeans(emb, num_levels=2, branching=2,
                                fit_indices=np.array([0, 2]))
    assert codes.shape == (4, 2)
    assert codes.min() >= 0 and codes.max() < 2
    print("OK  hierarchical k-means can fit on a subset and code all items")


def test_codes_validation_rejects_shape_mismatch_and_temporal_leakage():
    with tempfile.TemporaryDirectory() as tmp:
        codes_path = os.path.join(tmp, "codes.npy")
        np.save(codes_path, np.zeros((3, 2), dtype=np.int64))
        data_meta = {"num_items": 4}
        try:
            load_codes_validated(codes_path, data_meta, split="leave_one_out",
                                 expected_codebook_size=8)
        except ValueError as e:
            assert "num_items=4" in str(e)
        else:
            raise AssertionError("shape mismatch should fail")

        data_meta = {"num_items": 3, "temporal_val_cut": 10, "temporal_test_cut": 20}
        try:
            load_codes_validated(codes_path, data_meta, split="temporal",
                                 expected_codebook_size=8)
        except ValueError as e:
            assert "requires Semantic-ID metadata" in str(e)
        else:
            raise AssertionError("temporal split without sidecar should fail")

        save_json(sidecar_path(codes_path), {
            "artifact": "semantic_id_codes",
            "fit_split": "all",
            "num_items": 3,
            "codebook_size": 8,
        })
        try:
            load_codes_validated(codes_path, data_meta, split="temporal",
                                 expected_codebook_size=8)
        except ValueError as e:
            assert "fit_split='all'" in str(e)
        else:
            raise AssertionError("temporal split with all-fit codes should fail")

        save_json(sidecar_path(codes_path), {
            "artifact": "semantic_id_codes",
            "fit_split": "temporal",
            "num_items": 3,
            "codebook_size": 8,
            "temporal_val_cut": 9,
            "temporal_test_cut": 20,
        })
        try:
            load_codes_validated(codes_path, data_meta, split="temporal",
                                 expected_codebook_size=8)
        except ValueError as e:
            assert "temporal_val_cut" in str(e)
        else:
            raise AssertionError("temporal split with stale cutoff should fail")

        save_json(sidecar_path(codes_path), {
            "artifact": "semantic_id_codes",
            "fit_split": "temporal",
            "num_items": 3,
            "codebook_size": 8,
            "temporal_val_cut": 10,
            "temporal_test_cut": 20,
            "fit_item_ids_sha256": integer_set_fingerprint([1, 3]),
        })
        try:
            load_codes_validated(
                codes_path, data_meta, split="temporal",
                expected_codebook_size=8,
                expected_fit_item_ids_sha256=integer_set_fingerprint([1, 2]))
        except ValueError as e:
            assert "fit_item_ids_sha256" in str(e)
        else:
            raise AssertionError("temporal split with stale train-item set should fail")

        save_json(sidecar_path(codes_path), {
            "artifact": "semantic_id_codes",
            "fit_split": "temporal",
            "num_items": 3,
            "codebook_size": 8,
            "temporal_val_cut": 10,
            "temporal_test_cut": 20,
            "fit_item_ids_sha256": integer_set_fingerprint([1, 2]),
        })
        codes, meta = load_codes_validated(codes_path, data_meta, split="temporal",
                                           expected_codebook_size=8,
                                           expected_fit_item_ids_sha256=
                                               integer_set_fingerprint([1, 2]))
        assert codes.shape == (3, 2) and meta["fit_split"] == "temporal"
    print("OK  codes validation rejects mismatches and temporal leakage")


def test_codes_validation_rejects_stale_item_text_fingerprint():
    with tempfile.TemporaryDirectory() as tmp:
        codes_path = os.path.join(tmp, "codes.npy")
        np.save(codes_path, np.zeros((2, 2), dtype=np.int64))
        data_meta = {"num_items": 2}
        try:
            load_codes_validated(codes_path, data_meta, split="leave_one_out",
                                 expected_codebook_size=4,
                                 expected_item_text_sha256="new")
        except ValueError as e:
            assert "missing Semantic-ID metadata sidecar" in str(e)
        else:
            raise AssertionError("missing sidecar should fail when fingerprint validation is requested")

        save_json(sidecar_path(codes_path), {
            "artifact": "semantic_id_codes",
            "fit_split": "all",
            "num_items": 2,
            "codebook_size": 4,
            "item_text_sha256": "old",
        })
        try:
            load_codes_validated(codes_path, data_meta, split="leave_one_out",
                                 expected_codebook_size=4,
                                 expected_item_text_sha256="new")
        except ValueError as e:
            assert "item_text_sha256 does not match" in str(e)
        else:
            raise AssertionError("stale item_text fingerprint should fail")

        save_json(sidecar_path(codes_path), {
            "artifact": "semantic_id_codes",
            "fit_split": "all",
            "num_items": 2,
            "codebook_size": 4,
            "item_text_sha256": "new",
        })
        load_codes_validated(codes_path, data_meta, split="leave_one_out",
                             expected_codebook_size=4,
                             expected_item_text_sha256="new")
    print("OK  codes validation rejects stale item_text fingerprints")


def test_item_text_fingerprint_is_stable_and_content_sensitive():
    a = {"2": "beta", "1": "alpha"}
    b = {"1": "alpha", "2": "beta"}
    c = {"1": "alpha", "2": "gamma"}
    assert item_text_fingerprint(a) == item_text_fingerprint(b)
    assert item_text_fingerprint(a) != item_text_fingerprint(c)
    print("OK  item_text fingerprint is stable and content-sensitive")


def test_build_item_text_matches_paper_fields():
    meta = {
        "title": "Mascara",
        "brand": "BrandX",
        "categories": [["Beauty", "Eyes"], ["Makeup"]],
        "description": ["Waterproof", ["Black"]],
    }
    text = build_item_text(meta)
    assert "categories: Beauty Eyes Makeup" in text
    assert "description:" not in text
    assert "Waterproof" not in text
    assert "[" not in text and "]" not in text and "'" not in text
    print("OK  build_item_text matches paper fields and flattens nested metadata")


def test_generator_defaults_are_unconstrained():
    assert inspect.signature(generate_candidates).parameters["constrained"].default is False
    assert inspect.signature(generate_candidates_timeaware).parameters["constrained"].default is False
    assert inspect.signature(nar_generate).parameters["constrained"].default is False
    print("OK  low-level generator defaults are paper-style unconstrained")


if __name__ == "__main__":
    test_tokens_to_item_roundtrip()
    test_invalid_code_rate_counts_none()
    test_raw_codes_are_validated()
    test_pre_disambiguation_collision_metric()
    test_disambiguation_position_uses_full_codebook_slice()
    test_temporal_disambiguation_priority_prevents_future_code_shift()
    test_rank_from_scores_breaks_ties_pessimistically()
    test_empty_rank_metrics_raise()
    test_generated_pad_inside_code_is_invalid()
    test_nar_unconstrained_counts_off_catalogue_codes()
    test_target_coverage_diagnostic()
    test_temporal_training_window_keeps_singletons()
    test_train_temporal_kcore_uses_only_train_window()
    test_temporal_split_uses_fixed_meta_cutoffs()
    test_hkmeans_fit_subset_codes_all_items()
    test_codes_validation_rejects_shape_mismatch_and_temporal_leakage()
    test_codes_validation_rejects_stale_item_text_fingerprint()
    test_item_text_fingerprint_is_stable_and_content_sensitive()
    test_build_item_text_matches_paper_fields()
    test_generator_defaults_are_unconstrained()
    print("\nALL CORRECTNESS TESTS PASSED")
