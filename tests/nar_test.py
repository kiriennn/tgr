"""Quick synthetic test for the non-autoregressive (RQ4) denoiser.

Exercises only the NAR path: content embeddings -> RQ-VAE -> Semantic IDs ->
NARDenoiser train (masked denoising) -> single-shot trie-guided generation ->
metrics. CPU, seconds.  Run:  python3 tests/nar_test.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from semantic_ids.rqvae import RQVAE
from semantic_ids.build import SemanticIDSpace
from data.dataset import split_sequences, TigerDataset, tiger_collate
from models.nar import NARDenoiser, nar_loss, nar_generate
from metrics import metrics_from_ranks, rank_from_candidates

torch.manual_seed(0); np.random.seed(0)
DEV = "cpu"

# ---------------------------------------------------------------- synthetic data
NUM_CLUSTERS, PER_CLUSTER = 12, 10
NUM_ITEMS = NUM_CLUSTERS * PER_CLUSTER
NUM_USERS = 300
centers = np.random.randn(NUM_CLUSTERS, 768).astype(np.float32) * 3
emb = np.zeros((NUM_ITEMS, 768), np.float32)
for c in range(NUM_CLUSTERS):
    for j in range(PER_CLUSTER):
        idx = c * PER_CLUSTER + j
        emb[idx] = centers[c] + np.random.randn(768).astype(np.float32) * 0.3

sequences = {}
for u in range(1, NUM_USERS + 1):
    pref = np.random.randint(NUM_CLUSTERS)
    L = np.random.randint(5, 12)
    items = []
    for _ in range(L):
        c = pref if np.random.rand() < 0.8 else np.random.randint(NUM_CLUSTERS)
        items.append(c * PER_CLUSTER + np.random.randint(PER_CLUSTER) + 1)
    sequences[u] = items

# ---------------------------------------------------------------- RQ-VAE -> codes
print("training RQ-VAE...")
rqvae = RQVAE(input_dim=768, latent_dim=16, hidden=(128, 64), num_levels=3, codebook_size=16)
x = torch.tensor(emb)
opt = torch.optim.Adam(rqvae.parameters(), lr=1e-3)
with torch.no_grad():
    rqvae.init_codebooks(x)
for step in range(300):
    rqvae.train()
    _, _, loss, logs = rqvae(x)
    opt.zero_grad(); loss.backward(); opt.step()
codes = rqvae.encode_codes(x).numpy()
print("codes shape:", codes.shape)

# ---------------------------------------------------------------- semantic ID space
sp = SemanticIDSpace(codes, codebook_size=16, num_user_tokens=NUM_USERS + 2)
print(f"code_len={sp.code_len} vocab_size={sp.vocab_size} collisions={sp.collision_rate:.3f}")

# ---------------------------------------------------------------- split + loaders
train_ex, val, test = split_sequences(sequences)
train_ds = TigerDataset(train_ex, sp, max_items=20, add_user_token=True)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=tiger_collate)

# ---------------------------------------------------------------- NAR train
print("training NAR denoiser...")
model = NARDenoiser(sp, d_model=128, num_layers=3, num_heads=4).to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
for epoch in range(20):
    model.train()
    tot = 0.0
    for enc_ids, enc_mask, dec_ids in train_dl:
        loss = nar_loss(model, enc_ids.to(DEV), enc_mask.to(DEV), dec_ids.to(DEV), mask_prob=0.6)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += float(loss)
    if epoch % 4 == 0:
        print(f"  epoch {epoch:2d} loss {tot/len(train_dl):.4f}")

# ---------------------------------------------------------------- generate + eval
print("evaluating...")
items = list(test.items())
all_ranks = []
for i in range(0, len(items), 64):
    chunk = items[i:i + 64]
    examples = [(u, val[0], val[1]) if len(val) == 3 else (u, val[0], val[1]) for u, val in chunk]
    examples = [(u, val[0], val[-1]) for u, val in chunk]
    eds = TigerDataset(examples, sp, max_items=20, add_user_token=True)
    enc, mask, _ = tiger_collate([eds[j] for j in range(len(eds))])
    cand = nar_generate(model, enc, mask, sp, num_beams=20, device=DEV,
                        constrained=True)
    for ex, c in zip(examples, cand):
        all_ranks.append(rank_from_candidates(c, ex[-1]))

m = metrics_from_ranks(all_ranks, ks=(5, 10))
print("NAR metrics:", {k: round(v, 4) for k, v in m.items()})
assert m["recall@10"] > 0.10, "NAR recall too low -- pipeline likely broken"
print("\nNAR (RQ4) test PASSED")
