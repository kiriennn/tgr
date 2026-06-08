"""Preprocess raw Amazon files into the artifacts the models consume.

Steps (standard sequential-recommendation protocol, matching TIGER / SASRec):
1. Parse reviews -> (user, item, timestamp) triples.
2. Iterative k-core filtering. For paper-style leave-one-out we keep the
   standard full-data 5-core protocol. For temporal experiments, use
   ``--kcore-scope train_temporal``: the core is fitted only on interactions
   before the validation cutoff, avoiding future interactions in the user/item
   filter.
3. Re-index users (1..U) and items (1..I); 0 is reserved for padding.
4. Sort each user's interactions by timestamp -> a sequence of item ids.
5. Leave-one-out split: last item = test target, 2nd-last = val target, the
   rest = training history.
6. Build per-item text from metadata for content embeddings.

Outputs (saved with numpy/json) under ``--out``:
  sequences.json   {user_id: [item ids in time order]}
  item_text.json   {item_id: text string}
  meta.json        dataset sizes/statistics
  user_map.json    {raw_user_id: 1-based user_id}
  item_map.json    {raw_asin: 1-based item_id}
"""
from __future__ import annotations

import argparse
import ast
import gzip
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def parse_gz(path):
    open_fn = gzip.open if path.endswith(".gz") else open
    with open_fn(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                try:
                    yield ast.literal_eval(line)  # metadata uses python dict repr
                except Exception:
                    continue


def iterative_kcore(interactions, k=5):
    """interactions: list of (user, item, ts). Returns filtered list."""
    if k <= 0:
        return list(interactions)
    while True:
        uc, ic = defaultdict(int), defaultdict(int)
        for u, i, _ in interactions:
            uc[u] += 1
            ic[i] += 1
        keep = [(u, i, t) for (u, i, t) in interactions if uc[u] >= k and ic[i] >= k]
        if len(keep) == len(interactions):
            return keep
        interactions = keep


def temporal_cutoffs_from_interactions(interactions, val_frac=0.1, test_frac=0.1):
    """Global temporal cutoffs from raw (pre-core) interactions."""
    all_ts = sorted(t for _, _, t in interactions)
    if not all_ts:
        raise ValueError("temporal preprocessing requires at least one interaction")
    val_idx = int(len(all_ts) * (1 - val_frac - test_frac))
    test_idx = int(len(all_ts) * (1 - test_frac))
    val_idx = min(max(val_idx, 0), len(all_ts) - 1)
    test_idx = min(max(test_idx, 0), len(all_ts) - 1)
    return int(all_ts[val_idx]), int(all_ts[test_idx])


def apply_train_temporal_kcore(interactions, k, val_frac=0.1, test_frac=0.1):
    """Fit k-core only on the train window, then keep future rows for eval.

    Users/items are selected using only interactions with ``t < val_cut``. Future
    interactions for train-core users are kept so validation/test targets can be
    evaluated chronologically without using their counts to decide the core.
    Future items may therefore be cold relative to training, which downstream
    target-coverage diagnostics report explicitly.
    """
    val_cut, test_cut = temporal_cutoffs_from_interactions(interactions, val_frac, test_frac)
    train_raw = [(u, i, t) for (u, i, t) in interactions if t < val_cut]
    train_core = iterative_kcore(train_raw, k)
    if not train_core:
        raise ValueError(
            "train_temporal k-core produced no interactions; lower --kcore or "
            "check the input reviews file"
        )
    core_users = {u for u, _, _ in train_core}
    train_items = {i for _, i, _ in train_core}
    kept = [
        (u, i, t)
        for (u, i, t) in interactions
        if u in core_users and (t >= val_cut or i in train_items)
    ]
    return kept, {
        "temporal_val_cut": val_cut,
        "temporal_test_cut": test_cut,
        "train_kcore_num_users": len(core_users),
        "train_kcore_num_items": len(train_items),
        "train_kcore_num_interactions": len(train_core),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="Beauty")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed/Beauty")
    ap.add_argument("--kcore", type=int, default=5)
    ap.add_argument("--kcore-scope", choices=["full", "train_temporal"], default="full",
                    help="full = standard paper-style k-core over all rows; "
                         "train_temporal = fit k-core only before the temporal "
                         "validation cutoff")
    ap.add_argument("--review-set", choices=["auto", "5core", "all"], default="auto",
                    help="which default review file to read. auto uses 5core for "
                         "full k-core and all reviews for train_temporal.")
    ap.add_argument("--reviews-file", default=None,
                    help="override reviews path, e.g. data/raw/reviews_Beauty.json.gz")
    ap.add_argument("--meta-file", default=None,
                    help="override metadata path")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    review_set = args.review_set
    if review_set == "auto":
        review_set = "all" if args.kcore_scope == "train_temporal" else "5core"
    suffix = "_5" if review_set == "5core" else ""
    reviews_path = args.reviews_file or f"{args.raw}/reviews_{args.category}{suffix}.json.gz"
    meta_path = args.meta_file or f"{args.raw}/meta_{args.category}.json.gz"
    if args.kcore_scope == "train_temporal" and "_5" in os.path.basename(reviews_path):
        print("WARNING: train_temporal k-core is most meaningful with the full "
              f"reviews file, but got {reviews_path}")

    print("parsing reviews...")
    inter = []
    for r in parse_gz(reviews_path):
        u = r.get("reviewerID")
        i = r.get("asin")
        t = r.get("unixReviewTime")
        if u and i and t is not None:
            inter.append((u, i, int(t)))

    print(f"  raw interactions: {len(inter)}")
    meta_extra = {}
    if args.kcore_scope == "full":
        inter = iterative_kcore(inter, args.kcore)
        print(f"  after full-data {args.kcore}-core: {len(inter)}")
    else:
        inter, meta_extra = apply_train_temporal_kcore(
            inter, args.kcore, val_frac=args.val_frac, test_frac=args.test_frac)
        print(f"  after train-window {args.kcore}-core + future eval rows: {len(inter)}")
        print(f"  temporal cutoffs: val={meta_extra['temporal_val_cut']} "
              f"test={meta_extra['temporal_test_cut']}")

    # re-index (1-based; 0 = pad)
    users, items = {}, {}
    def uid(u): return users.setdefault(u, len(users) + 1)
    def iid(i): return items.setdefault(i, len(items) + 1)

    by_user = defaultdict(list)
    for u, i, t in inter:
        by_user[uid(u)].append((t, iid(i)))

    sequences = {}
    timestamps = {}
    for u, lst in by_user.items():
        lst.sort()  # by timestamp
        sequences[u] = [i for _, i in lst]
        timestamps[u] = [t for t, _ in lst]

    # item text from metadata
    asin_to_id = items
    item_text = {}
    print("parsing metadata...")
    for m in parse_gz(meta_path):
        asin = m.get("asin")
        if asin in asin_to_id:
            from semantic_ids.content import build_item_text  # lazy import
            item_text[asin_to_id[asin]] = build_item_text(m)
    # any item without metadata gets a placeholder
    for iid_ in range(1, len(items) + 1):
        item_text.setdefault(iid_, f"item {iid_}")

    meta = {
        "category": args.category,
        "review_set": review_set,
        "kcore": args.kcore,
        "kcore_scope": args.kcore_scope,
        "num_users": len(users),
        "num_items": len(items),
        "num_interactions": len(inter),
        "avg_seq_len": sum(len(s) for s in sequences.values()) / len(sequences),
    }
    meta.update(meta_extra)
    json.dump({str(u): s for u, s in sequences.items()}, open(f"{args.out}/sequences.json", "w"))
    json.dump({str(u): t for u, t in timestamps.items()}, open(f"{args.out}/timestamps.json", "w"))
    json.dump({str(k): v for k, v in item_text.items()}, open(f"{args.out}/item_text.json", "w"))
    json.dump(meta, open(f"{args.out}/meta.json", "w"), indent=2)
    json.dump(users, open(f"{args.out}/user_map.json", "w"), indent=2, sort_keys=True)
    json.dump(items, open(f"{args.out}/item_map.json", "w"), indent=2, sort_keys=True)
    print("meta:", json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
