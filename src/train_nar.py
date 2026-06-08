"""Train + evaluate the non-autoregressive (masked-denoising) retrieval model.

This is the RQ4 alternative to the autoregressive TIGER decoder. The model is a
BERT/MaskGIT-style denoiser: during training, random positions of the target
Semantic ID are replaced by [MASK] and reconstructed jointly; at inference the
code is read out in one shot from an all-masked input. Headline metrics use the
same paper-style unconstrained decoding policy as TIGER: invalid generated code
tuples are dropped from the candidate list and counted as a diagnostic.

It shares the dataset, Semantic ID space, metrics and full-ranking evaluation
protocol with ``train_tiger.py`` so the two are directly comparable.

Example:
  python3 src/train_nar.py --data data/processed/Beauty \
      --codes data/processed/Beauty/codes_rqvae_L3_K256.npy \
      --epochs 100 --out runs/beauty_nar.json
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from torch.utils.data import DataLoader

from semantic_ids.build import SemanticIDSpace
from semantic_ids.io import (integer_set_fingerprint, item_text_fingerprint,
                              load_codes_validated)
from data.dataset import (split_sequences, split_sequences_temporal,
                          items_in_training_window, target_coverage,
                          temporal_kwargs_from_meta, TigerDataset, tiger_collate)
from models.nar import (NARDenoiser, nar_loss, nar_generate_candidates,
                        ParallelHeadNAR, parallel_loss)
from metrics import (metrics_from_ranks, rank_from_candidates,
                     invalid_code_rate, collision_rate)
from runtime import select_device


def evaluate(model, eval_dict, sp, device, num_beams, max_items,
             batch_size=64, constrained=False, measure_invalid=True,
             max_users=None):
    items = list(eval_dict.items())
    if max_users:
        items = items[:max_users]
    ranks, raw_all, valid_counts = [], [], []
    t0 = time.time(); n_dec = 0
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        examples = []
        for u, val in chunk:
            hist, tgt = val[0], val[-1]
            examples.append((u, hist, tgt))
        eds = TigerDataset(examples, sp, max_items=max_items, add_user_token=True)
        enc, mask, _ = tiger_collate([eds[j] for j in range(len(eds))])
        cand, raw = nar_generate_candidates(
            model, enc, mask, sp, num_beams=num_beams,
            constrained=constrained, device=device)
        for ex, c in zip(examples, cand):
            ranks.append(rank_from_candidates(c, ex[-1]))
            valid_counts.append(len(c))
        raw_all.extend(raw)
        n_dec += len(examples)
    latency = (time.time() - t0) / max(n_dec, 1)
    m = metrics_from_ranks(ranks, ks=(5, 10, 20))
    vc = np.asarray(valid_counts, dtype=np.int64)
    m["mean_valid_candidates"] = float(vc.mean()) if len(vc) else 0.0
    m["min_valid_candidates"] = int(vc.min()) if len(vc) else 0
    for k in (5, 10, 20):
        m[f"valid_candidates_at_least_{k}_rate"] = float((vc >= k).mean()) if len(vc) else 0.0
    if measure_invalid:
        m["invalid_code_rate"] = invalid_code_rate(raw_all, sp.valid_codes)
        for k in (5, 10, 20):
            m[f"invalid_code_rate@{k}"] = invalid_code_rate(raw_all, sp.valid_codes, limit=k)
    m["decode_latency_s"] = latency
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--codes", required=True, help=".npy of raw per-item codes")
    ap.add_argument("--codebook-size", type=int, default=256)
    ap.add_argument("--num-user-tokens", type=int, default=2000)
    ap.add_argument("--model", choices=["denoiser", "parallel"], default="denoiser",
                    help="denoiser = masked-denoising (MaskGIT-style); "
                         "parallel = simple parallel linear heads (RQ4 baseline)")
    ap.add_argument("--split", choices=["leave_one_out", "temporal"],
                    default="leave_one_out")
    ap.add_argument("--max-items", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--mask-prob", type=float, default=0.6)
    ap.add_argument("--num-beams", type=int, default=30)
    ap.add_argument("--constrained-eval", action="store_true",
                    help="also report trie-constrained decoding as an ablation "
                         "(headline metrics use paper-style unconstrained)")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--eval-users", type=int, default=None)
    ap.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    ap.add_argument("--out", default="runs/nar.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = select_device(args.device)
    print(f"device={device}")

    sequences = json.load(open(f"{args.data}/sequences.json"))
    data_meta = json.load(open(f"{args.data}/meta.json"))
    item_text = json.load(open(f"{args.data}/item_text.json"))
    text_sha = item_text_fingerprint(item_text)
    timestamps = None
    if args.split == "temporal":
        timestamps = json.load(open(f"{args.data}/timestamps.json"))
    temporal_kwargs = temporal_kwargs_from_meta(data_meta) if args.split == "temporal" else {}
    disambiguation_priority = None
    fit_item_ids_sha = None
    if args.split == "temporal":
        disambiguation_priority = sorted(
            items_in_training_window(sequences, timestamps, split=args.split,
                                     **temporal_kwargs)
        )
        fit_item_ids_sha = integer_set_fingerprint(disambiguation_priority)
    raw_codes, codes_meta = load_codes_validated(
        args.codes, data_meta, split=args.split,
        expected_codebook_size=args.codebook_size,
        expected_item_text_sha256=text_sha,
        expected_fit_item_ids_sha256=fit_item_ids_sha)

    sp = SemanticIDSpace(
        raw_codes, codebook_size=args.codebook_size,
        num_user_tokens=args.num_user_tokens,
        disambiguation_priority=disambiguation_priority,
        seed=args.seed)
    print(f"vocab={sp.vocab_size} code_len={sp.code_len} "
          f"max_collisions={sp.max_collisions} collision_rate={sp.collision_rate:.4f}")

    if args.split == "temporal":
        train_ex, val, test = split_sequences_temporal(sequences, timestamps,
                                                       **temporal_kwargs)
    else:
        train_ex, val, test = split_sequences(sequences)
    print(f"split={args.split} train examples={len(train_ex)} val={len(val)} test={len(test)}")

    if args.model == "denoiser":
        model = NARDenoiser(sp, d_model=args.d_model, num_layers=args.num_layers,
                            max_items=args.max_items).to(device)
    else:
        model = ParallelHeadNAR(sp, d_model=args.d_model, num_layers=args.num_layers,
                                max_items=args.max_items).to(device)
    print(f"NAR variant: {args.model}")
    ds = TigerDataset(train_ex, sp, max_items=args.max_items, add_user_token=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=tiger_collate)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    import copy
    best, best_state = None, None
    for ep in range(args.epochs):
        model.train(); tot = 0.0
        for enc, mask, dec in dl:
            if args.model == "denoiser":
                loss = nar_loss(model, enc.to(device), mask.to(device), dec.to(device),
                                mask_prob=args.mask_prob)
            else:
                loss = parallel_loss(model, enc.to(device), mask.to(device), dec.to(device))
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        if (ep + 1) % max(1, args.epochs // 10) == 0:
            vm = evaluate(model, val, sp, device, args.num_beams, args.max_items,
                          constrained=False, measure_invalid=False,
                          max_users=args.eval_users)
            print(f"epoch {ep+1} loss={tot/len(dl):.4f} "
                  f"val R@10={vm['recall@10']:.4f} N@10={vm['ndcg@10']:.4f}")
            if best is None or vm["recall@10"] > best:
                best = vm["recall@10"]
                best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"final test evaluation (best val R@10={best})...")
    tm = evaluate(model, test, sp, device, args.num_beams, args.max_items,
                  constrained=False, measure_invalid=True, max_users=args.eval_users)
    tm["collision_rate_full"] = collision_rate(sp.full_codes)
    tm["collision_rate_pre_disambiguation"] = collision_rate(sp.content_codes)
    seen_items = (
        set(disambiguation_priority)
        if disambiguation_priority is not None
        else items_in_training_window(sequences, timestamps, split=args.split,
                                      **temporal_kwargs)
    )
    split_diagnostics = {
        "val_target_coverage": target_coverage(val, seen_items),
        "test_target_coverage": target_coverage(test, seen_items),
    }
    result = {"config": vars(args), "test": tm, "best_val_recall@10": best,
              "id_space": {"vocab_size": sp.vocab_size, "code_len": sp.code_len,
                           "max_collisions": sp.max_collisions,
                           "disambiguation_size": sp.disambiguation_size,
                           "disambiguation_priority_size":
                               sp.disambiguation_priority_size},
              "codes_metadata": codes_meta,
              "split_diagnostics": split_diagnostics}
    if args.constrained_eval:
        result["test_constrained"] = evaluate(
            model, test, sp, device, args.num_beams, args.max_items,
            constrained=True, measure_invalid=True, max_users=args.eval_users)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print(json.dumps(tm, indent=2))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
