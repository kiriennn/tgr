"""Build Semantic ID codes for a processed dataset.

Computes Sentence-T5 content embeddings, then quantises them into per-item codes
with either RQ-VAE (default) or hierarchical k-means (RQ1 alternative). Saves the
raw codes so the downstream TIGER runs are fast and reproducible.

Usage:
  python3 src/train_semantic_ids.py --data data/processed/Beauty --id-method rqvae
  python3 src/train_semantic_ids.py --data data/processed/Beauty --id-method hkmeans
  python3 src/train_semantic_ids.py --data data/processed/Beauty_temporal --id-method rqvae --fit-split temporal
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

from data.dataset import items_in_training_window, temporal_kwargs_from_meta
from semantic_ids.content import embed_texts
from semantic_ids.io import (integer_set_fingerprint, item_text_fingerprint,
                              save_json, sidecar_path)
from semantic_ids.rqvae import RQVAE
from semantic_ids.hkmeans import hierarchical_kmeans
from runtime import select_device


def training_fit_indices(data_dir: str, fit_split: str, num_items: int) -> np.ndarray | None:
    """Return 0-based item indices used to fit Semantic-ID codebooks.

    ``fit_split='all'`` reproduces the transductive paper-style setting where
    the quantizer sees every catalogue item's content embedding. For temporal
    quality experiments, ``fit_split='temporal'`` fits the quantizer only on
    items observed in the temporal training window, then assigns codes to all
    items with the fitted quantizer/tree.
    """
    if fit_split == "all":
        return None

    sequences = json.load(open(f"{data_dir}/sequences.json"))
    meta = json.load(open(f"{data_dir}/meta.json"))
    timestamps = None
    if fit_split == "temporal":
        timestamps = json.load(open(f"{data_dir}/timestamps.json"))
    elif fit_split != "leave_one_out":
        raise ValueError(f"unknown fit_split={fit_split}")

    temporal_kwargs = temporal_kwargs_from_meta(meta) if fit_split == "temporal" else {}
    seen = items_in_training_window(sequences, timestamps, split=fit_split,
                                    **temporal_kwargs)
    idx = np.array(sorted(i - 1 for i in seen if 1 <= int(i) <= num_items), dtype=np.int64)
    if len(idx) == 0:
        raise ValueError(f"fit_split={fit_split} produced no training items")
    return idx


def train_rqvae(emb, num_levels, codebook_size, latent_dim, epochs, device, seed=42,
                fit_indices=None):
    torch.manual_seed(seed)
    x_all = torch.tensor(emb, dtype=torch.float32, device=device)
    if fit_indices is None:
        fit_indices = np.arange(len(emb), dtype=np.int64)
    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    x = x_all[torch.tensor(fit_indices, dtype=torch.long, device=device)]
    model = RQVAE(input_dim=emb.shape[1], latent_dim=latent_dim,
                  num_levels=num_levels, codebook_size=codebook_size).to(device)
    with torch.no_grad():
        model.init_codebooks(x)            # k-means init for EVERY level's codebook
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = x.shape[0]
    bs = min(1024, n)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, bs):
            xb = x[perm[i:i + bs]]
            _, _, loss, logs = model(xb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        # revive dead codes periodically to keep codebook utilisation high
        if (ep + 1) % 20 == 0:
            with torch.no_grad():
                z = model.encoder(x)
                res = z
                for vq in model.quantizers:
                    vq.revive_dead_codes(res)
                    q, _, _ = vq(res); res = res - q
        if (ep + 1) % 20 == 0:
            print(f"  epoch {ep+1}/{epochs} loss={tot/((n+bs-1)//bs):.4f} "
                  f"recon={logs['recon']:.4f} vq={logs['vq']:.4f}")
    return model.encode_codes(x_all).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="processed data dir")
    ap.add_argument("--id-method", choices=["rqvae", "hkmeans"], default="rqvae")
    ap.add_argument("--num-levels", type=int, default=3)
    ap.add_argument("--codebook-size", type=int, default=256)
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--rqvae-epochs", type=int, default=500)
    ap.add_argument("--embed-backend", default="sentence-t5",
                    help="sentence-t5 | random (random = offline plumbing only)")
    ap.add_argument("--force-recompute-embeddings", action="store_true",
                    help="ignore cached content embeddings and rebuild them")
    ap.add_argument("--fit-split", choices=["all", "leave_one_out", "temporal"],
                    default="all",
                    help="which items are used to FIT the Semantic-ID quantizer/tree. "
                         "Use temporal for leakage-free temporal experiments; all "
                         "matches the transductive paper-style reproduction setting.")
    ap.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    device = select_device(args.device)
    print(f"device={device}")

    item_text = json.load(open(f"{args.data}/item_text.json"))
    data_meta = json.load(open(f"{args.data}/meta.json"))
    num_items = max(int(k) for k in item_text)
    texts = [item_text[str(i)] for i in range(1, num_items + 1)]
    text_sha = item_text_fingerprint(item_text)

    cache = f"{args.data}/content_emb_{args.embed_backend}.npy"
    cache_meta_path = sidecar_path(cache)
    cache_ok = False
    if os.path.exists(cache) and os.path.exists(cache_meta_path) and not args.force_recompute_embeddings:
        cache_meta = json.load(open(cache_meta_path))
        cache_ok = (
            cache_meta.get("backend") == args.embed_backend
            and cache_meta.get("num_items") == num_items
            and cache_meta.get("item_text_sha256") == text_sha
        )
    if os.path.exists(cache) and cache_ok:
        emb = np.load(cache)
        if emb.ndim != 2 or emb.shape[0] != num_items:
            raise ValueError(
                f"cached embeddings {cache} have shape={emb.shape}, "
                f"expected first dimension num_items={num_items}; "
                f"rerun with --force-recompute-embeddings"
            )
        print(f"loaded cached embeddings {emb.shape}")
    else:
        if os.path.exists(cache) and not args.force_recompute_embeddings:
            print("cached embeddings are missing/stale metadata; recomputing...")
        print(f"embedding {len(texts)} items with backend={args.embed_backend}...")
        emb = embed_texts(texts, backend=args.embed_backend, device=device)
        np.save(cache, emb)
        save_json(cache_meta_path, {
            "artifact": "content_embeddings",
            "backend": args.embed_backend,
            "num_items": num_items,
            "embedding_dim": int(emb.shape[1]),
            "item_text_sha256": text_sha,
        })

    fit_indices = training_fit_indices(args.data, args.fit_split, num_items)
    fit_count = len(emb) if fit_indices is None else len(fit_indices)
    fit_item_ids = (
        list(range(1, num_items + 1))
        if fit_indices is None
        else [int(i) + 1 for i in fit_indices]
    )
    print(f"fitting Semantic-ID codebooks on {fit_count}/{len(emb)} items "
          f"(fit_split={args.fit_split})")

    if args.id_method == "rqvae":
        print("training RQ-VAE...")
        codes = train_rqvae(emb, args.num_levels, args.codebook_size,
                            args.latent_dim, args.rqvae_epochs, device, args.seed,
                            fit_indices=fit_indices)
    else:
        print("hierarchical k-means...")
        codes = hierarchical_kmeans(emb, num_levels=args.num_levels,
                                    branching=args.codebook_size, seed=args.seed,
                                    fit_indices=fit_indices)

    tag = f"{args.id_method}_L{args.num_levels}_K{args.codebook_size}"
    if args.fit_split != "all":
        tag += f"_fit-{args.fit_split}"
    out = f"{args.data}/codes_{tag}.npy"
    np.save(out, codes)
    sidecar = {
        "artifact": "semantic_id_codes",
        "version": 1,
        "id_method": args.id_method,
        "fit_split": args.fit_split,
        "num_items": num_items,
        "num_levels": args.num_levels,
        "codebook_size": args.codebook_size,
        "latent_dim": args.latent_dim if args.id_method == "rqvae" else None,
        "rqvae_epochs": args.rqvae_epochs if args.id_method == "rqvae" else None,
        "embed_backend": args.embed_backend,
        "item_text_sha256": text_sha,
        "fit_num_items": int(fit_count),
        "fit_item_ids_sha256": integer_set_fingerprint(fit_item_ids),
        "seed": args.seed,
        "codes_shape": list(map(int, codes.shape)),
    }
    if args.fit_split == "temporal":
        for key in ("temporal_val_cut", "temporal_test_cut"):
            if key not in data_meta:
                raise ValueError(
                    f"fit_split=temporal requires {key} in {args.data}/meta.json; "
                    "re-run preprocess.py with --kcore-scope train_temporal"
                )
            sidecar[key] = int(data_meta[key])
    save_json(sidecar_path(out), sidecar)
    # quick collision diagnostic before disambiguation
    from collections import Counter
    c = Counter(map(tuple, codes))
    coll = sum(v for v in c.values() if v > 1) / len(codes)
    print(f"saved {out}  shape={codes.shape}  pre-disambiguation collision rate={coll:.4f}")


if __name__ == "__main__":
    main()
