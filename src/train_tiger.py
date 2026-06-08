"""Train + evaluate a TIGER-style generative retrieval model.

This single script runs the autoregressive TIGER baseline and all of the
controlled variants used in the report:

  RQ1  --codes codes_hkmeans_*.npy        (hierarchical k-means IDs)
  RQ2  --token-order {original,reversed,permuted}
       --num-levels / --codebook-size      (code-length sweep; choose codes file)
       --collision-first                    (move disambiguation token to front)
  RQ3  --time-gaps                          (interleave gap tokens; needs timestamps)

Outputs a JSON with Recall@K, NDCG@K (full ranking via paper-style
unconstrained beam search) plus diagnostics: collision rate, invalid-code rate,
and mean decode latency. Trie-constrained decoding is available as an optional
ablation via ``--constrained-eval``.

Example:
  python3 src/train_tiger.py --data data/processed/Beauty \
      --codes data/processed/Beauty/codes_rqvae_L3_K256.npy \
      --codebook-size 256 --max-items 20 --split leave_one_out \
      --epochs 100 --lr 3e-4 --num-beams 100 \
      --out runs/beauty_tiger_lr3e4_e100.json
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
from models.tiger import (build_tiger_model, tiger_loss, generate_candidates,
                          TimeAwareTiger, time_aware_loss,
                          generate_candidates_timeaware)
from metrics import (metrics_from_ranks, rank_from_candidates,
                     invalid_code_rate, collision_rate)
from runtime import select_device


def _flatten_metrics(prefix, metrics):
    return {f"{prefix}/{k}": float(v) for k, v in metrics.items()
            if isinstance(v, (int, float, np.floating))}


def _global_grad_norm(parameters):
    norms = []
    for p in parameters:
        if p.grad is not None:
            norms.append(torch.linalg.vector_norm(p.grad.detach(), ord=2))
    if not norms:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(norms), ord=2).detach().cpu())


def evaluate(model, eval_dict, sp, device, num_beams, max_items, use_time_gaps,
             batch_size=64, constrained=False, measure_invalid=True, max_users=None,
             time_embed=False):
    """Full-ranking evaluation.

    ``constrained=False`` (default) matches the paper: the model generates
    Semantic IDs freely and invalid generations are simply dropped from the
    candidate list (and counted in ``invalid_code_rate``). ``constrained=True``
    forces every beam onto a valid item via the decoding trie -- this is an
    engineering variant we report as an ablation, not the headline number,
    because it cannot produce invalid codes by construction. ``time_embed=True``
    feeds per-token gap-bucket ids so the time-aware embedding is used.
    """
    items = list(eval_dict.items())
    if max_users:
        items = items[:max_users]
    ranks, raw_all, valid_counts = [], [], []
    t0 = time.time(); n_dec = 0
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        examples = []
        for u, val in chunk:
            if len(val) == 3:
                hist, times, tgt = val
                examples.append((u, hist, times, tgt))
            else:
                hist, tgt = val
                examples.append((u, hist, tgt))
        eds = TigerDataset(examples, sp, max_items=max_items, add_user_token=True,
                           use_time_gaps=use_time_gaps, use_time_embed=time_embed)
        batch = tiger_collate([eds[j] for j in range(len(eds))])
        if time_embed:
            enc, mask, _, gap = batch
            cand, raw = generate_candidates_timeaware(
                model, enc, mask, gap, sp, num_beams=num_beams,
                constrained=constrained, device=device)
        else:
            enc, mask, _ = batch
            cand, raw = generate_candidates(model, enc, mask, sp, num_beams=num_beams,
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
        # pass raw beams UNFILTERED: None/short generations count as invalid.
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
    ap.add_argument("--token-order", default="original",
                    choices=["original", "reversed", "permuted"])
    ap.add_argument("--collision-first", action="store_true")
    time_group = ap.add_mutually_exclusive_group()
    time_group.add_argument("--time-gaps", action="store_true")
    time_group.add_argument("--time-embed", action="store_true",
                            help="add a learned gap-bucket embedding to each event's "
                                 "token embeddings (instructor rec #1; RQ3 variant)")
    ap.add_argument("--split", choices=["leave_one_out", "temporal"],
                    default="leave_one_out",
                    help="temporal = global time-based holdout (recommended for "
                         "quality eval / time-aware experiments)")
    ap.add_argument("--num-user-tokens", type=int, default=2000)
    ap.add_argument("--max-items", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader worker processes")
    ap.add_argument("--pin-memory", action="store_true",
                    help="pin DataLoader memory for faster CUDA transfers")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num-beams", type=int, default=100)
    ap.add_argument("--constrained-eval", action="store_true",
                    help="also report trie-constrained decoding as an ablation "
                         "(headline metrics always use paper-style unconstrained)")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--eval-users", type=int, default=None,
                    help="cap eval users for speed during development")
    ap.add_argument("--eval-every", type=int, default=None,
                    help="run validation every N epochs; default = epochs//10")
    ap.add_argument("--metrics-log", default=None,
                    help="optional JSONL file with per-epoch train/val/test metrics")
    ap.add_argument("--log-grad-norm", action="store_true",
                    help="log global L2 gradient norm during training")
    ap.add_argument("--grad-norm-every", type=int, default=1,
                    help="compute grad norm every N train batches when --log-grad-norm is set")
    ap.add_argument("--amp", action="store_true",
                    help="use CUDA automatic mixed precision during training")
    ap.add_argument("--amp-dtype", choices=["bfloat16", "float16"], default="bfloat16",
                    help="AMP dtype on CUDA; bfloat16 is preferred on A100")
    ap.add_argument("--wandb-project", default=None,
                    help="optional Weights & Biases project for online charts")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-run-name", default=None)
    ap.add_argument("--wandb-tags", default=None,
                    help="comma-separated W&B tags")
    ap.add_argument("--wandb-mode", choices=["online", "offline", "disabled"],
                    default="online")
    ap.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    ap.add_argument("--out", default="runs/tiger.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = select_device(args.device)
    print(f"device={device}", flush=True)
    eval_every = args.eval_every if args.eval_every is not None else max(1, args.epochs // 10)
    if eval_every <= 0:
        raise ValueError("--eval-every must be positive")
    if args.grad_norm_every <= 0:
        raise ValueError("--grad-norm-every must be positive")
    amp_enabled = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    scaler_enabled = amp_enabled and args.amp_dtype == "float16"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    if args.amp and not amp_enabled:
        print(f"AMP requested but disabled on device={device}", flush=True)
    elif amp_enabled:
        print(f"amp=True dtype={args.amp_dtype}", flush=True)

    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
        except ImportError as e:
            raise RuntimeError(
                "W&B logging requested but wandb is not installed; run `pip install wandb`"
            ) from e
        if args.wandb_mode == "online":
            has_env_key = bool(os.environ.get("WANDB_API_KEY"))
            has_netrc_key = os.path.exists(os.path.expanduser("~/.netrc"))
            if not has_env_key and not has_netrc_key:
                raise RuntimeError(
                    "W&B online logging requested but no login was found. "
                    "In Colab, run the W&B login cell first, or set USE_WANDB=False."
                )
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()] if args.wandb_tags else None
        run_name = args.wandb_run_name or os.path.splitext(os.path.basename(args.out))[0]
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=run_name,
            tags=tags,
            config=vars(args),
            mode=args.wandb_mode,
        )

    def write_metrics(record):
        if not args.metrics_log:
            return
        os.makedirs(os.path.dirname(args.metrics_log) or ".", exist_ok=True)
        with open(args.metrics_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    if args.metrics_log:
        os.makedirs(os.path.dirname(args.metrics_log) or ".", exist_ok=True)
        with open(args.metrics_log, "w", encoding="utf-8") as f:
            f.write(json.dumps({"phase": "config", "config": vars(args)}, sort_keys=True) + "\n")

    sequences = json.load(open(f"{args.data}/sequences.json"))
    data_meta = json.load(open(f"{args.data}/meta.json"))
    item_text = json.load(open(f"{args.data}/item_text.json"))
    text_sha = item_text_fingerprint(item_text)
    timestamps = None
    if args.time_gaps or args.time_embed or args.split == "temporal":
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
        raw_codes, codebook_size=args.codebook_size, token_order=args.token_order,
        num_user_tokens=args.num_user_tokens,
        num_gap_tokens=5 if args.time_gaps else 0,
        collision_first=args.collision_first,
        disambiguation_priority=disambiguation_priority,
        seed=args.seed)
    print(f"vocab={sp.vocab_size} code_len={sp.code_len} "
          f"max_collisions={sp.max_collisions} collision_rate={sp.collision_rate:.4f}",
          flush=True)

    if args.split == "temporal":
        train_ex, val, test = split_sequences_temporal(sequences, timestamps,
                                                       **temporal_kwargs)
    else:
        train_ex, val, test = split_sequences(sequences, timestamps)
    print(f"split={args.split} train examples={len(train_ex)} val={len(val)} test={len(test)}",
          flush=True)

    if args.time_embed:
        model = TimeAwareTiger(sp.vocab_size, num_gap_buckets=5,
                               d_model=args.d_model, num_layers=args.num_layers).to(device)
    else:
        model = build_tiger_model(sp.vocab_size, d_model=args.d_model,
                                  num_layers=args.num_layers).to(device)
    ds = TigerDataset(train_ex, sp, max_items=args.max_items, add_user_token=True,
                      use_time_gaps=args.time_gaps, use_time_embed=args.time_embed)
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, collate_fn=tiger_collate,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory and device.type == "cuda"))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    import copy
    best, best_state = None, None
    for ep in range(args.epochs):
        model.train(); tot = 0
        grad_norm_sum = grad_norm_max = 0.0
        grad_norm_count = 0
        for batch_idx, batch in enumerate(dl):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type if amp_enabled else "cpu",
                                dtype=amp_dtype,
                                enabled=amp_enabled):
                if args.time_embed:
                    enc, mask, dec, gap = batch
                    loss = time_aware_loss(model, enc.to(device), mask.to(device),
                                           gap.to(device), dec.to(device))
                else:
                    enc, mask, dec = batch
                    loss = tiger_loss(model, enc.to(device), mask.to(device), dec.to(device))
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if args.log_grad_norm and (batch_idx + 1) % args.grad_norm_every == 0:
                    scaler.unscale_(opt)
                    gn = _global_grad_norm(model.parameters())
                    grad_norm_sum += gn
                    grad_norm_max = max(grad_norm_max, gn)
                    grad_norm_count += 1
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if args.log_grad_norm and (batch_idx + 1) % args.grad_norm_every == 0:
                    gn = _global_grad_norm(model.parameters())
                    grad_norm_sum += gn
                    grad_norm_max = max(grad_norm_max, gn)
                    grad_norm_count += 1
                opt.step()
            tot += float(loss.detach().cpu())
        avg_loss = tot / len(dl)
        record = {"phase": "epoch", "epoch": ep + 1, "loss": float(avg_loss)}
        wandb_payload = {"epoch": ep + 1, "train/loss": float(avg_loss)}
        grad_txt = ""
        if grad_norm_count:
            grad_mean = grad_norm_sum / grad_norm_count
            record["grad_norm_mean"] = float(grad_mean)
            record["grad_norm_max"] = float(grad_norm_max)
            wandb_payload["train/grad_norm_mean"] = float(grad_mean)
            wandb_payload["train/grad_norm_max"] = float(grad_norm_max)
            grad_txt = f" grad={grad_mean:.3f}"
        if (ep + 1) % eval_every == 0:
            vm = evaluate(model, val, sp, device, args.num_beams, args.max_items,
                          args.time_gaps, constrained=False, measure_invalid=False,
                          time_embed=args.time_embed,
                          max_users=args.eval_users)
            record["val"] = vm
            wandb_payload.update(_flatten_metrics("val", vm))
            print(f"epoch {ep+1} loss={avg_loss:.4f} "
                  f"val R@10={vm['recall@10']:.4f} N@10={vm['ndcg@10']:.4f}{grad_txt}",
                  flush=True)
            if best is None or vm["recall@10"] > best:
                best = vm["recall@10"]
                best_state = copy.deepcopy(model.state_dict())  # checkpoint best
                record["best_val_recall@10"] = best
                wandb_payload["val/best_recall@10"] = float(best)
        else:
            print(f"epoch {ep+1} loss={avg_loss:.4f}{grad_txt}", flush=True)
        write_metrics(record)
        if wandb_run is not None:
            wandb_run.log(wandb_payload, step=ep + 1)

    if best_state is not None:           # evaluate test on the best-val checkpoint
        model.load_state_dict(best_state)
    print(f"final test evaluation (best val R@10={best})...", flush=True)
    # headline metrics: paper-style UNCONSTRAINED decoding
    tm = evaluate(model, test, sp, device, args.num_beams, args.max_items,
                  args.time_gaps, constrained=False, measure_invalid=True,
                  time_embed=args.time_embed, max_users=args.eval_users)
    tm["collision_rate_full"] = collision_rate(sp.full_codes)
    # pre-disambiguation collisions: measured on the CONTENT codes directly so the
    # number is correct regardless of where the disambiguation token sits.
    tm["collision_rate_pre_disambiguation"] = collision_rate(sp.content_codes)
    write_metrics({"phase": "test", "best_val_recall@10": best, "test": tm})
    if wandb_run is not None:
        payload = {"test/best_val_recall@10": float(best) if best is not None else -1.0}
        payload.update(_flatten_metrics("test", tm))
        wandb_run.log(payload, step=args.epochs + 1)
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
    if args.constrained_eval:            # engineering ablation, not the headline
        result["test_constrained"] = evaluate(
            model, test, sp, device, args.num_beams, args.max_items,
            args.time_gaps, constrained=True, measure_invalid=True,
            time_embed=args.time_embed, max_users=args.eval_users)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print(json.dumps(tm, indent=2), flush=True)
    print(f"saved -> {args.out}", flush=True)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
