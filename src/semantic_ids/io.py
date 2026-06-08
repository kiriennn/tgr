"""I/O helpers for Semantic-ID artifacts.

The raw ``.npy`` codes are just an integer matrix, so by themselves they do not
say which dataset/split they were fitted on.  The small JSON sidecars written by
``train_semantic_ids.py`` make temporal experiments fail fast instead of silently
using all-catalogue codes and leaking future item content into the codebook.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sidecar_path(path: str | os.PathLike, suffix: str = ".meta.json") -> str:
    """Return ``foo.meta.json`` for ``foo.npy``."""
    p = Path(path)
    return str(p.with_suffix(suffix))


def item_text_fingerprint(item_text: dict) -> str:
    """Stable fingerprint of item text content keyed by 1-based item id."""
    h = hashlib.sha256()
    for key in sorted(item_text, key=lambda x: int(x)):
        h.update(str(int(key)).encode("utf-8"))
        h.update(b"\0")
        h.update(str(item_text[key]).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def integer_set_fingerprint(values) -> str:
    """Stable fingerprint for a set/list of integer ids."""
    h = hashlib.sha256()
    for value in sorted({int(v) for v in values}):
        h.update(str(value).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def load_json(path: str | os.PathLike):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | os.PathLike, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def maybe_load_sidecar(path: str | os.PathLike) -> dict | None:
    meta_path = sidecar_path(path)
    if not os.path.exists(meta_path):
        return None
    return load_json(meta_path)


def load_codes_validated(
    codes_path: str,
    data_meta: dict,
    split: str,
    expected_codebook_size: int | None = None,
    expected_item_text_sha256: str | None = None,
    expected_fit_item_ids_sha256: str | None = None,
) -> tuple[np.ndarray, dict | None]:
    """Load Semantic-ID codes and validate them against the dataset/split.

    For temporal evaluation we require a metadata sidecar and ``fit_split`` must
    be ``temporal``.  Without this check, a run can accidentally use all-catalogue
    Semantic IDs and leak future item content through the quantizer/tree.
    """
    codes = np.load(codes_path)
    if codes.ndim != 2:
        raise ValueError(f"{codes_path} must be a 2D code matrix, got shape={codes.shape}")

    num_items = int(data_meta["num_items"])
    if codes.shape[0] != num_items:
        raise ValueError(
            f"{codes_path} has {codes.shape[0]} rows, but dataset meta says "
            f"num_items={num_items}"
        )

    meta = maybe_load_sidecar(codes_path)
    if meta is None and expected_item_text_sha256 is not None:
        raise ValueError(
            f"{codes_path} is missing Semantic-ID metadata sidecar "
            f"{sidecar_path(codes_path)}, so it cannot be validated against "
            f"the current dataset item_text.json. Rebuild codes with "
            f"`train_semantic_ids.py --data ...`."
        )
    if meta is not None:
        meta_items = int(meta.get("num_items", -1))
        if meta_items != num_items:
            raise ValueError(
                f"{sidecar_path(codes_path)} says num_items={meta_items}, "
                f"but dataset meta says num_items={num_items}"
            )
        meta_k = meta.get("codebook_size")
        if expected_codebook_size is not None and meta_k is not None:
            if int(meta_k) != int(expected_codebook_size):
                raise ValueError(
                    f"{sidecar_path(codes_path)} says codebook_size={meta_k}, "
                    f"but CLI --codebook-size={expected_codebook_size}"
                )
        if expected_item_text_sha256 is not None:
            meta_sha = meta.get("item_text_sha256")
            if meta_sha != expected_item_text_sha256:
                raise ValueError(
                    f"{sidecar_path(codes_path)} item_text_sha256 does not match "
                    f"the current dataset item_text.json. Rebuild Semantic IDs with "
                    f"`train_semantic_ids.py --data ...`; expected "
                    f"{expected_item_text_sha256}, got {meta_sha!r}."
                )
        if expected_fit_item_ids_sha256 is not None:
            meta_fit_sha = meta.get("fit_item_ids_sha256")
            if meta_fit_sha != expected_fit_item_ids_sha256:
                raise ValueError(
                    f"{sidecar_path(codes_path)} fit_item_ids_sha256 does not match "
                    f"the current training-window item set. Rebuild Semantic IDs "
                    f"with the same processed dataset/split; expected "
                    f"{expected_fit_item_ids_sha256}, got {meta_fit_sha!r}."
                )

    if split == "temporal":
        if meta is None:
            raise ValueError(
                f"Temporal split requires Semantic-ID metadata sidecar "
                f"{sidecar_path(codes_path)}. Rebuild codes with "
                f"`train_semantic_ids.py --fit-split temporal`."
            )
        if meta.get("fit_split") != "temporal":
            raise ValueError(
                f"Temporal split requires codes fitted on the temporal training "
                f"window, but {sidecar_path(codes_path)} has "
                f"fit_split={meta.get('fit_split')!r}."
            )
        for key in ("temporal_val_cut", "temporal_test_cut"):
            expected = data_meta.get(key)
            if expected is None:
                raise ValueError(
                    f"Temporal split requires {key} in dataset meta.json. "
                    f"Re-run preprocess.py with --kcore-scope train_temporal."
                )
            actual = meta.get(key)
            if actual is None or int(actual) != int(expected):
                raise ValueError(
                    f"{sidecar_path(codes_path)} has {key}={actual!r}, but "
                    f"dataset meta.json has {key}={expected!r}. Rebuild Semantic "
                    f"IDs for this exact temporal split."
                )

    return codes, meta
