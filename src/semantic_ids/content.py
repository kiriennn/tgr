"""Content embeddings for items.

TIGER derives Semantic IDs from a frozen Sentence-T5 embedding of each item's
textual metadata (title, brand, categories, price, ...). We use
``sentence-transformers/sentence-t5-base`` (768-dim) to match the paper.
For the paper-aligned setting we compose text from title, brand, categories and
price. Descriptions are intentionally excluded: adding them changes the content
distribution and makes Semantic IDs less comparable to the reported TIGER setup.

If sentence-transformers / the model download is unavailable (e.g. the smoke
test runs with no internet), set ``backend="random"`` to get deterministic
pseudo-embeddings so the rest of the pipeline can still be exercised. Random
embeddings obviously give meaningless Semantic IDs -- they are for plumbing
tests only, never for reported results.
"""
from __future__ import annotations

import numpy as np


def _flatten_metadata_value(value) -> list[str]:
    """Flatten Amazon metadata fields without leaking Python list syntax."""
    if value is None or value == "":
        return []
    if isinstance(value, dict):
        out = []
        for key in sorted(value):
            out.extend(_flatten_metadata_value(value[key]))
        return out
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            out.extend(_flatten_metadata_value(item))
        return out
    return [str(value)]


def build_item_text(meta: dict) -> str:
    """Compose the text TIGER feeds to Sentence-T5 from an item's metadata."""
    parts = []
    for key in ("title", "brand", "categories", "price"):
        v = meta.get(key)
        flat = " ".join(_flatten_metadata_value(v)).strip()
        if flat:
            parts.append(f"{key}: {flat}")
    return ". ".join(parts)[:512]


def embed_texts(texts, backend: str = "sentence-t5", model_name="sentence-transformers/sentence-t5-base",
                batch_size: int = 256, device: str | None = None) -> np.ndarray:
    if backend == "random":
        import hashlib
        out = np.zeros((len(texts), 768), dtype=np.float32)
        for i, t in enumerate(texts):
            # Python's built-in hash() is salted per process (PYTHONHASHSEED), so
            # it is NOT reproducible across runs. Use a stable content hash so the
            # "random" plumbing backend is deterministic, as documented.
            h = hashlib.blake2b(str(t).encode("utf-8"), digest_size=8).digest()
            seed = int.from_bytes(h, "little") % (2**32)
            rng = np.random.default_rng(seed)
            out[i] = rng.standard_normal(768).astype(np.float32)
        return out

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(list(texts), batch_size=batch_size, show_progress_bar=True,
                       convert_to_numpy=True, normalize_embeddings=False)
    return emb.astype(np.float32)
