# TIGER-GenRec: a controlled study of generative retrieval

Reproduction and ablation study of **TIGER** — *Recommender Systems with
Generative Retrieval* (Rajput et al., NeurIPS 2023, [arXiv:2305.05065](https://arxiv.org/abs/2305.05065))
— built for the HSE *DeepRecSys* final project.

TIGER replaces nearest-neighbour retrieval with **autoregressive generation of
Semantic ID tokens**: each item is encoded into a short tuple of discrete
codewords (RQ-VAE over Sentence-T5 content embeddings), and a seq2seq Transformer
is trained from scratch to generate the next item's code from the interaction
history. This repo reimplements that pipeline and adds the controlled experiments
described in our report.

## Research questions

| RQ  | Question | How it is run |
|-----|----------|---------------|
| RQ1 | Is residual quantization necessary, or can a simpler prefix-conditional content tree work? | `train_semantic_ids.py --id-method {rqvae,hkmeans}` |
| RQ2 | How does Semantic ID *structure* affect quality and failure modes (code length, codebook size, token order, collision-token placement)? | `train_tiger.py --token-order {original,reversed,permuted} --collision-first`, plus code-length sweep via the codes file |
| RQ3 | Do explicit time signals help generative retrieval? | `train_tiger.py --time-gaps` (separate gap token) or `--time-embed` (gap embedding added to each event, instructor's suggestion) |
| RQ4 | Is the seq2seq **autoregressive** pipeline necessary, or can a non-autoregressive model match it? | `train_nar.py --model {denoiser,parallel}` (masked-denoising, or the simpler parallel-head baseline) |

All models are evaluated with the **same full-ranking protocol** (no sampled
negatives) so the numbers are directly comparable. Two splits are supported:
leave-one-out (`--split leave_one_out`, for reproducing the paper) and a global
time-based holdout (`--split temporal`, recommended for the quality/time-aware
experiments — see *Evaluation notes*). For temporal experiments, preprocess from
the full review file with train-window k-core, then build Semantic IDs with
`train_semantic_ids.py --fit-split temporal` so the quantizer/tree is fit only
on the training window before assigning codes to all items. TIGER and NAR
headline metrics use paper-style **unconstrained** decoding; trie-constrained
decoding is available as an ablation via `--constrained-eval`.

## What is implemented

```
src/
  metrics.py                  Recall@K / NDCG@K (full ranking) + diagnostics
  semantic_ids/
    content.py                item text -> Sentence-T5 embeddings (or random fallback)
    rqvae.py                  RQ-VAE (residual VQ, k-means init, dead-code revival)
    hkmeans.py                hierarchical k-means codes (RQ1 alternative)
    build.py                  Semantic ID space: offsets, collision token,
                              token-order variants, gap/user tokens, decoding trie
  data/
    download.py               Amazon 2014 reviews (5-core or full) + metadata
    preprocess.py             full-data or train-window k-core, reindex,
                              leave-one-out / temporal-ready item text
    dataset.py                splits, time-gap bucketing, datasets + collators
  models/
    tiger.py                  T5 seq2seq + unconstrained / trie-constrained
                              beam search (TIGER)
    sasrec.py                 SASRec baseline (full-vocab softmax)
    nar.py                    non-autoregressive denoiser + parallel-head
                              baseline (RQ4)
  train_semantic_ids.py       build + cache Semantic IDs
  train_tiger.py              train/eval TIGER and all RQ1-RQ3 variants
  train_sasrec.py             train/eval the SASRec baseline
  train_nar.py                train/eval the RQ4 model
tests/
  smoke_test.py               end-to-end pipeline test on synthetic data (CPU, ~1 min)
  nar_test.py                 RQ4 pipeline test on synthetic data
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+. A **GPU is strongly recommended** for the real experiments
(training the T5 model from scratch on Beauty takes ~hours on CPU but is fast on
a single GPU; Google Colab works). Building the Semantic IDs downloads the
Sentence-T5 model from HuggingFace, so the first run needs internet access.

Verify the install without any downloads:

```bash
python3 tests/smoke_test.py      # full pipeline on synthetic data
python3 tests/nar_test.py        # RQ4 pipeline on synthetic data
python3 tests/test_correctness.py # regression tests for audit findings
```

## Reproducing the paper (Amazon Beauty)

### 1. Download + preprocess

```bash
python3 src/data/download.py   --category Beauty --out data/raw --review-set 5core
python3 src/data/preprocess.py --category Beauty --raw data/raw \
                              --out data/processed/Beauty --kcore 5
```

This produces `data/processed/Beauty/{sequences,timestamps,item_text,meta}.json`
using iterative 5-core filtering and the standard leave-one-out split (last item
= test, second-to-last = validation). If the SNAP mirror is unavailable, pass
`--reviews-url` / `--meta-url` (alternative mirrors are listed in
`src/data/download.py`).

For leakage-clean temporal experiments, build a separate processed dataset from
the full (non-5-core) review file. Here k-core is fitted only on the train
window (`t < val_cut`), and the fixed temporal cutoffs are saved in `meta.json`
so all downstream scripts use the same split:

```bash
python3 src/data/download.py   --category Beauty --out data/raw --review-set all
python3 src/data/preprocess.py --category Beauty --raw data/raw \
       --out data/processed/Beauty_temporal --kcore 5 \
       --kcore-scope train_temporal --review-set all
```

### 2. Build Semantic IDs

```bash
# RQ-VAE codes (paper setting: 3 levels x codebook 256)
python3 src/train_semantic_ids.py --data data/processed/Beauty \
       --id-method rqvae --num-levels 3 --codebook-size 256 \
       --rqvae-epochs 500

# RQ1 alternative: hierarchical k-means with the same code shape
python3 src/train_semantic_ids.py --data data/processed/Beauty \
       --id-method hkmeans --num-levels 3 --codebook-size 256
```

Outputs `codes_rqvae_L3_K256.npy` / `codes_hkmeans_L3_K256.npy` plus a
matching `*.meta.json` sidecar with the dataset size, codebook size, fit split
and item-text fingerprint.
(For an offline plumbing check you can add `--embed-backend random`, but those
codes are meaningless and are only for testing that the wiring runs.)

For temporal experiments, fit the Semantic-ID codebook/tree on the temporal
training window and then code the whole catalogue:

```bash
python3 src/train_semantic_ids.py --data data/processed/Beauty_temporal \
       --id-method rqvae --num-levels 3 --codebook-size 256 \
       --rqvae-epochs 500 --fit-split temporal
```

This writes `codes_rqvae_L3_K256_fit-temporal.npy` under
`data/processed/Beauty_temporal/`.
When `--split temporal` is used downstream, train-window items are also assigned
collision/disambiguation tokens before future/cold catalogue items, so a future
item with the same content code cannot change a train item target ID.
The trainers require the temporal sidecar metadata and will fail fast if
temporal evaluation is pointed at codes fitted on the full catalogue.

### 3. Train + evaluate

```bash
# TIGER (main result)
python3 src/train_tiger.py --data data/processed/Beauty \
       --codes data/processed/Beauty/codes_rqvae_L3_K256.npy \
       --codebook-size 256 --max-items 20 --split leave_one_out \
       --epochs 100 --lr 3e-4 --num-beams 100 \
       --out runs/beauty_tiger_lr3e4_e100.json

# SASRec baseline
python3 src/train_sasrec.py --data data/processed/Beauty \
       --epochs 100 --out runs/beauty_sasrec.json
```

Each run writes a JSON with Recall@{5,10,20}, NDCG@{5,10,20} and the
generative-retrieval diagnostics (collision rate, invalid-code rate, invalid
code rate @K, valid-candidate coverage after filtering, decode latency).

### Target numbers (paper, Amazon Beauty)

| Model  | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|--------|---------:|-------:|----------:|--------:|
| SASRec | 0.0387   | 0.0249 | 0.0605    | 0.0318  |
| TIGER  | 0.0454   | 0.0321 | 0.0648    | 0.0384  |

(Values from Rajput et al. 2023, Table 1; Beauty has 22,363 users / 12,101 items
after 5-core.) Training from scratch on a single category is high-variance, so
exact reproduction is not expected; we report our own runs and the gap in the
report.

## Running the experiments

```bash
# RQ1 -- residual quantization vs hierarchical k-means
python3 src/train_tiger.py --data data/processed/Beauty \
       --codes data/processed/Beauty/codes_hkmeans_L3_K256.npy \
       --codebook-size 256 --max-items 20 --split leave_one_out \
       --epochs 100 --lr 3e-4 --num-beams 100 --out runs/beauty_hkmeans.json

# RQ2 -- token order
python3 src/train_tiger.py --data data/processed/Beauty \
       --codes data/processed/Beauty/codes_rqvae_L3_K256.npy --codebook-size 256 \
       --max-items 20 --token-order reversed --epochs 100 --lr 3e-4 \
       --num-beams 100 --out runs/beauty_reversed.json
python3 src/train_tiger.py ... --max-items 20 --token-order permuted \
       --epochs 100 --lr 3e-4 --num-beams 100 --out runs/beauty_permuted.json

# RQ2 -- collision token first instead of last
python3 src/train_tiger.py ... --max-items 20 --collision-first \
       --epochs 100 --lr 3e-4 --num-beams 100 --out runs/beauty_collfirst.json

# RQ2 -- code-length / codebook sweep: build other codes then point --codes at them
python3 src/train_semantic_ids.py --data data/processed/Beauty \
       --id-method rqvae --num-levels 4 --codebook-size 256 \
       --rqvae-epochs 500
python3 src/train_tiger.py ... --codes .../codes_rqvae_L4_K256.npy \
       --max-items 20 --epochs 100 --lr 3e-4 --num-beams 100

# RQ3 -- time signal as a separate gap token ...
# first run the same plain TIGER model on the same temporal split
python3 src/train_tiger.py --data data/processed/Beauty_temporal \
       --codes data/processed/Beauty_temporal/codes_rqvae_L3_K256_fit-temporal.npy --codebook-size 256 \
       --max-items 20 --split temporal --epochs 100 --lr 3e-4 --num-beams 100 \
       --out runs/beauty_tiger_temporal_lr3e4_e100.json
python3 src/train_tiger.py --data data/processed/Beauty_temporal \
       --codes data/processed/Beauty_temporal/codes_rqvae_L3_K256_fit-temporal.npy --codebook-size 256 \
       --max-items 20 --time-gaps --split temporal --epochs 100 --lr 3e-4 \
       --num-beams 100 --out runs/beauty_timegaps_temporal_lr3e4_e100.json
# RQ3 -- ... vs a gap embedding ADDED to each event (instructor's suggestion)
python3 src/train_tiger.py ... --max-items 20 --time-embed --split temporal \
       --epochs 100 --lr 3e-4 --num-beams 100 \
       --out runs/beauty_timeembed_temporal_lr3e4_e100.json

# RQ4 -- non-autoregressive: masked-denoising model ...
python3 src/train_nar.py --data data/processed/Beauty \
       --codes data/processed/Beauty/codes_rqvae_L3_K256.npy \
       --codebook-size 256 --model denoiser --epochs 100 --out runs/beauty_nar.json
# RQ4 -- ... vs the simpler parallel-linear-head baseline
python3 src/train_nar.py ... --model parallel --out runs/beauty_nar_parallel.json
```

The same commands work on `Sports_and_Outdoors` and `Toys_and_Games` by changing
`--category` in steps 1-2 and the `--data` / `--codes` paths.

A convenience script that runs the full Beauty pipeline end to end is provided in
`scripts/run_beauty.sh`.
For only Research Question 3, run `bash scripts/run_rq3.sh`; it keeps the
leave-one-out reproduction as a reference run and uses the temporal split as the
main comparison for plain TIGER, gap-token TIGER, and gap-embedding TIGER.

## Evaluation notes

* **Full ranking, not sampled negatives.** Every catalogue item is a candidate.
  For generative models the candidate list is what beam search decodes; the
  ground-truth rank is read from that ordered list and items never generated
  count as misses (the correct full-ranking treatment).
* **Semantic-ID vocabulary.** The collision/disambiguation position gets the
  same codebook-sized token slice as the content positions (`K` tokens, or more
  only if a pathological collision group needs it), matching the paper's
  `4 * K` Semantic-ID token layout.
* **Decoding mode.** TIGER and NAR headline metrics use **unconstrained**
  decoding, as in the paper: invalid generations are dropped from the candidate
  list and counted in `invalid_code_rate` (a real failure mode).
  Trie-`--constrained-eval` is reported separately as an engineering ablation —
  it cannot emit invalid codes, so it is more optimistic and not directly
  comparable to the paper-style headline.
* **Splits.** `--split leave_one_out` (last item test, second-to-last val)
  reproduces SASRec/TIGER. `--split temporal` is a global time-based holdout
  (two timestamp cutoffs; history always precedes the target in wall-clock time)
  and is the recommended setting for the quality and time-aware experiments, per
  the course feedback that leave-one-out should not be the main quality protocol.
  For this split, preprocess from the full review file with
  `preprocess.py --kcore-scope train_temporal --review-set all`, then build codes
  on that processed dataset with `train_semantic_ids.py --fit-split temporal`;
  otherwise the user/item core or the RQ-VAE/k-means codebook can use future
  information. TIGER, SASRec, and NAR trainers all expose `--split`; compare
  methods only within the same split.
* **Model selection.** All trainers keep the best checkpoint by validation
  Recall@10 on the full validation split and report test on it. `--eval-users`
  is only a development-time speed cap when passed explicitly.
* **Diagnostics.** `invalid_code_rate` uses unconstrained decoding and counts
  malformed/short generations as invalid. Runs also log
  `invalid_code_rate@{5,10,20}`, matching the top-K invalid-ID plots in the
  paper. `collision_rate` (pre-disambiguation) is computed on the raw content
  codes, so it is correct regardless of where the collision token sits. Each run
  also records `valid_candidates_at_least_{5,10,20}_rate` so it is visible when
  beam search did not leave enough valid, de-duplicated items after filtering
  invalid codes. Each run also records target coverage diagnostics for validation/test
  (`seen_target_rate` and `cold_target_rate`) so temporal-split results are not
  accidentally interpreted as pure warm-item recommendation.
* **Artifact validation.** Semantic-ID codes are validated against
  `meta.json["num_items"]`, `--codebook-size`, and the current
  `item_text.json` fingerprint before training, so `codes.npy` must have the
  `*.meta.json` sidecar written by `train_semantic_ids.py`. Temporal runs also
  require `fit_split: temporal`, preventing accidental leakage from
  all-catalogue Semantic-ID fitting.

## Differences from the paper (state these in the report)

* **RQ-VAE training budget.** The paper trains the RQ-VAE far longer (≈20k steps,
  Adagrad, lr 0.4, target codebook usage ≥80%); our default is 500 epochs with
  Adam. This is a scaled-down student reimplementation, not a bit-exact
  reproduction — expect ballpark numbers and the same ordering, not identical
  digits.
* **Time-awareness.** We implement both the minimal gap-token variant (`--time-gaps`)
  and, following the course feedback, a gap **embedding** added to each event
  (`--time-embed`), analogous to SASRec's positional embedding.
* **Temporal split.** Added per the feedback that leave-one-out should not be the
  sole quality protocol. The clean temporal dataset is built from full reviews
  with `preprocess.py --kcore-scope train_temporal --review-set all`, so user/item
  filtering is based only on the train window. Use `--fit-split temporal` when
  building Semantic IDs for this setting; this fits the codebook/tree only on
  train-window items. The trainers also prioritize train-window items when
  resolving Semantic-ID collisions, preventing future/cold items from shifting
  train-item disambiguation tokens.
* **Simpler NAR baseline.** In addition to the masked-denoising decoder we include
  the parallel-linear-head model (`--model parallel`) suggested as a baseline.
* **Hierarchical k-means baseline.** This is a prefix-conditional content tree,
  not a perfect drop-in replacement for residual quantization. Interpret RQ1 as
  a comparison against a simpler tree-structured content discretizer.

## License / attribution

Research/educational reimplementation for the HSE DeepRecSys course. The TIGER
method is due to Rajput et al. (2023); SASRec to Kang & McAuley (2018). No code
from the original (unreleased) TIGER implementation is used.
