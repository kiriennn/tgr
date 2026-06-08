"""Train + evaluate the SASRec baseline with full-ranking metrics.

Usage:
  python3 src/train_sasrec.py --data data/processed/Beauty --epochs 200 \
      --out runs/beauty_sasrec.json
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import (split_sequences, split_sequences_temporal,
                          items_in_training_window, target_coverage,
                          temporal_kwargs_from_meta, SasrecDataset)
from models.sasrec import SASRec
from metrics import metrics_from_ranks, rank_from_scores
from runtime import select_device


@torch.no_grad()
def evaluate(model, eval_dict, max_len, device, exclude_seen=False):
    model.eval(); ranks = []
    for u, val in eval_dict.items():
        hist, tgt = (val[0], val[-1])
        h = hist[-max_len:]; h = [0] * (max_len - len(h)) + h
        scores = model.score_last(torch.tensor([h], device=device))[0].cpu().numpy()
        exclude = set(hist) if exclude_seen else None
        ranks.append(rank_from_scores(scores, tgt, exclude=exclude))
    return metrics_from_ranks(ranks, ks=(5, 10, 20))


def train_sequences_from_examples(train_examples):
    """Recover one chronological training sequence per user from prefix examples."""
    seqs = {}
    for ex in train_examples:
        u = int(ex[0])
        hist, tgt = ex[1], ex[-1]
        seq = list(hist) + [tgt]
        if len(seq) > len(seqs.get(u, [])):
            seqs[u] = seq
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--max-len", type=int, default=50)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--num-heads", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--split", choices=["leave_one_out", "temporal"],
                    default="leave_one_out")
    ap.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    ap.add_argument("--out", default="runs/sasrec.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = select_device(args.device)
    print(f"device={device}")

    sequences = json.load(open(f"{args.data}/sequences.json"))
    data_meta = json.load(open(f"{args.data}/meta.json"))
    num_items = data_meta["num_items"]
    temporal_kwargs = {}
    if args.split == "temporal":
        timestamps = json.load(open(f"{args.data}/timestamps.json"))
        temporal_kwargs = temporal_kwargs_from_meta(data_meta)
        train_ex, val, test = split_sequences_temporal(sequences, timestamps,
                                                       **temporal_kwargs)
        train_seqs = train_sequences_from_examples(train_ex)
    else:
        train_ex, val, test = split_sequences(sequences)
        train_seqs = train_sequences_from_examples(train_ex)
    print(f"split={args.split} train users={len(train_seqs)} val={len(val)} test={len(test)}")

    ds = SasrecDataset(train_seqs, num_items, max_len=args.max_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    model = SASRec(num_items, max_len=args.max_len, d_model=args.d_model,
                   num_layers=args.num_layers, num_heads=args.num_heads,
                   dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    import copy
    best, best_state = -1, None
    for ep in range(args.epochs):
        model.train(); tot = 0
        for seq, tgt in dl:
            loss = model.loss(seq.to(device), tgt.to(device))
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        if (ep + 1) % max(1, args.epochs // 10) == 0:
            vm = evaluate(model, val, args.max_len, device)
            print(f"epoch {ep+1} loss={tot/len(dl):.4f} val R@10={vm['recall@10']:.4f}")
            if vm["recall@10"] > best:
                best = vm["recall@10"]
                best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    tm = evaluate(model, test, args.max_len, device)
    seen_items = items_in_training_window(
        sequences, timestamps if args.split == "temporal" else None,
        split=args.split, **temporal_kwargs)
    result = {"config": vars(args), "test": tm, "best_val_recall@10": best,
              "split_diagnostics": {
                  "val_target_coverage": target_coverage(val, seen_items),
                  "test_target_coverage": target_coverage(test, seen_items),
              }}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print(json.dumps(tm, indent=2)); print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
