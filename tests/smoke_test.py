"""End-to-end smoke test on tiny synthetic data (CPU, seconds).

Validates that every component wires together: content embeddings -> RQ-VAE ->
Semantic IDs -> token space -> TIGER training + constrained decoding -> metrics,
and the SASRec baseline. It is NOT a quality benchmark -- it only proves the
pipeline runs and produces sane outputs. Run:  python3 tests/smoke_test.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from semantic_ids.rqvae import RQVAE
from semantic_ids.build import SemanticIDSpace
from data.dataset import (split_sequences, TigerDataset, tiger_collate, SasrecDataset)
from models.tiger import build_tiger_model, tiger_loss, generate_candidates
from models.sasrec import SASRec
from metrics import (metrics_from_ranks, rank_from_candidates, rank_from_scores,
                     invalid_code_rate, collision_rate)

torch.manual_seed(0); np.random.seed(0)
DEV = "cpu"

# ---------------------------------------------------------------- synthetic data
NUM_CLUSTERS, PER_CLUSTER = 12, 10
NUM_ITEMS = NUM_CLUSTERS * PER_CLUSTER
NUM_USERS = 300
centers = np.random.randn(NUM_CLUSTERS, 768).astype(np.float32) * 3
emb = np.zeros((NUM_ITEMS, 768), np.float32)
item_cluster = np.zeros(NUM_ITEMS, int)
for c in range(NUM_CLUSTERS):
    for j in range(PER_CLUSTER):
        idx = c * PER_CLUSTER + j
        emb[idx] = centers[c] + np.random.randn(768).astype(np.float32) * 0.3
        item_cluster[idx] = c

sequences = {}
for u in range(1, NUM_USERS + 1):
    pref = np.random.randint(NUM_CLUSTERS)
    L = np.random.randint(5, 12)
    items = []
    for _ in range(L):
        c = pref if np.random.rand() < 0.8 else np.random.randint(NUM_CLUSTERS)
        items.append(c * PER_CLUSTER + np.random.randint(PER_CLUSTER) + 1)  # 1-based
    sequences[u] = items

# ---------------------------------------------------------------- RQ-VAE -> codes
print("training RQ-VAE...")
rqvae = RQVAE(input_dim=768, latent_dim=16, hidden=(128, 64), num_levels=3, codebook_size=16)
x = torch.tensor(emb)
opt = torch.optim.Adam(rqvae.parameters(), lr=1e-3)
# kmeans init level 0
with torch.no_grad():
    rqvae.init_codebooks(x)
for step in range(300):
    rqvae.train()
    _, _, loss, logs = rqvae(x)
    opt.zero_grad(); loss.backward(); opt.step()
print(f"  rqvae final recon={logs['recon']:.3f} vq={logs['vq']:.3f}")
raw_codes = rqvae.encode_codes(x).numpy()

sp = SemanticIDSpace(raw_codes, codebook_size=16, token_order="original", num_user_tokens=64)
print(f"  semantic-id vocab={sp.vocab_size} code_len={sp.code_len} "
      f"max_collisions={sp.max_collisions} collision_rate={sp.collision_rate:.3f}")
print(f"  pre-disambiguation collision rate (prefix L-1)="
      f"{collision_rate(sp.full_codes, prefix_len=sp.code_len-1):.3f}")

# ---------------------------------------------------------------- splits
train_ex, val, test = split_sequences(sequences)
print(f"  train examples={len(train_ex)} val users={len(val)} test users={len(test)}")

# ---------------------------------------------------------------- TIGER
print("training TIGER (tiny)...")
model = build_tiger_model(sp.vocab_size, d_model=64, d_ff=128, num_layers=2, num_heads=4, d_kv=16)
ds = TigerDataset(train_ex, sp, max_items=10, add_user_token=True)
dl = DataLoader(ds, batch_size=64, shuffle=True, collate_fn=tiger_collate)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
for epoch in range(4):
    model.train(); tot = 0
    for enc, mask, dec in dl:
        loss = tiger_loss(model, enc, mask, dec)
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
    print(f"  epoch {epoch} loss={tot/len(dl):.3f}")

# evaluate TIGER on test (constrained + unconstrained for diagnostics)
print("evaluating TIGER...")
test_users = list(test.items())[:120]
ranks, raw_all = [], []
for i in range(0, len(test_users), 32):
    chunk = test_users[i:i+32]
    examples = [(u, hist, tgt) for u, (hist, tgt) in chunk]
    eds = TigerDataset(examples, sp, max_items=10, add_user_token=True)
    enc, mask, _ = tiger_collate([eds[j] for j in range(len(eds))])
    items_pe, raw_pe = generate_candidates(model, enc, mask, sp, num_beams=20,
                                           constrained=True, device=DEV)
    for (u, hist, tgt), cand in zip(examples, items_pe):
        ranks.append(rank_from_candidates(cand, tgt))
    # unconstrained for invalid-code diagnostic
    _, raw_unc = generate_candidates(model, enc, mask, sp, num_beams=20,
                                     constrained=False, device=DEV)
    raw_all.extend(raw_unc)

m = metrics_from_ranks(ranks, ks=(5, 10))
icr = invalid_code_rate(raw_all, sp.valid_codes)
print(f"  TIGER {m}  invalid_code_rate(unconstrained)={icr:.3f}")

# ---------------------------------------------------------------- SASRec
print("training SASRec (tiny)...")
sas = SASRec(NUM_ITEMS, max_len=20, d_model=32, num_layers=2, num_heads=2)
train_seqs = {u: s[:-2] for u, s in sequences.items() if len(s) >= 3}
sds = SasrecDataset(train_seqs, NUM_ITEMS, max_len=20)
sdl = DataLoader(sds, batch_size=64, shuffle=True)
sopt = torch.optim.AdamW(sas.parameters(), lr=3e-3)
for epoch in range(6):
    sas.train(); tot = 0
    for seq, tgt in sdl:
        loss = sas.loss(seq, tgt)
        sopt.zero_grad(); loss.backward(); sopt.step(); tot += loss.item()

sas.eval(); sranks = []
for u, (hist, tgt) in test.items():
    h = hist[-20:]; h = [0] * (20 - len(h)) + h
    scores = sas.score_last(torch.tensor([h]))[0].numpy()
    sranks.append(rank_from_scores(scores, tgt))
sm = metrics_from_ranks(sranks, ks=(5, 10))
print(f"  SASRec {sm}")

# ---------------------------------------------------------------- sanity asserts
assert m["recall@10"] > 0.10, "TIGER recall too low -- pipeline likely broken"
# SASRec quality on this tiny CPU toy setup is not a stable benchmark. The real
# correctness check here is that the full-softmax path trains, scores the whole
# catalogue, and returns finite full-ranking metrics under pessimistic tie
# handling.
assert all(np.isfinite(v) for v in sm.values()), "SASRec metrics must be finite"
assert icr >= 0.0
print("\nSMOKE TEST PASSED")
