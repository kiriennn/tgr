"""Residual-Quantized VAE (RQ-VAE) for Semantic ID construction.

Maps a content embedding x (e.g. Sentence-T5 of the item text) to a tuple of
``num_levels`` discrete codes (c_1, ..., c_L), each in ``[0, codebook_size)``.
Level l quantizes the residual left over after levels 1..l-1, so early codes
capture coarse structure and later codes refine it -- the "semantic hierarchy"
claimed by TIGER (which RQ #1 and RQ #2 in the report probe directly).

Implementation notes
--------------------
* Straight-through estimator for gradients through the argmin quantizer.
* Codebook loss + commitment loss per level (van den Oord et al., 2017).
* Optional k-means initialisation of each codebook from the residuals seen on
  the first training batch (stabilises training, as in the TIGER paper).
* Dead-code revival: codes unused for a while are re-seeded from random encoder
  outputs, which keeps codebook utilisation (and thus collision rate) healthy.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(sizes, act=nn.ReLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, dim: int, commitment: float = 0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.commitment = commitment
        self.codebook = nn.Parameter(torch.randn(codebook_size, dim) * 0.1)
        self.register_buffer("_inited", torch.tensor(0))
        self.register_buffer("_usage", torch.zeros(codebook_size))

    @torch.no_grad()
    def kmeans_init(self, x: torch.Tensor, iters: int = 10):
        """Initialise the codebook with k-means over the given residuals."""
        n = x.shape[0]
        if n < self.codebook_size:
            idx = torch.randint(0, n, (self.codebook_size,), device=x.device)
            self.codebook.data.copy_(x[idx])
            self._inited.fill_(1)
            return
        idx = torch.randperm(n, device=x.device)[: self.codebook_size]
        centroids = x[idx].clone()
        for _ in range(iters):
            d = torch.cdist(x, centroids)
            assign = d.argmin(1)
            for k in range(self.codebook_size):
                sel = x[assign == k]
                if len(sel) > 0:
                    centroids[k] = sel.mean(0)
        self.codebook.data.copy_(centroids)
        self._inited.fill_(1)

    def forward(self, x: torch.Tensor):
        # x: (B, dim). Returns quantized (straight-through), codes, loss.
        d = torch.cdist(x, self.codebook)             # (B, K)
        codes = d.argmin(1)                           # (B,)
        q = self.codebook[codes]                      # (B, dim) raw codebook vector
        codebook_loss = F.mse_loss(q, x.detach())
        commit_loss = F.mse_loss(x, q.detach())
        loss = codebook_loss + self.commitment * commit_loss
        if self.training:
            with torch.no_grad():
                self._usage.scatter_add_(0, codes, torch.ones_like(codes, dtype=self._usage.dtype))
        # Return the RAW quantized vector (no straight-through here). The single
        # straight-through estimator is applied once on the accumulated code in
        # RQVAE.quantize, which keeps the residual chain differentiable w.r.t. the
        # encoder at every level (a per-level ST makes the residual cancel and
        # zeros the encoder gradient for levels > 0).
        return q, codes, loss

    @torch.no_grad()
    def revive_dead_codes(self, x: torch.Tensor, threshold: float = 1.0):
        """Re-seed codes used fewer than ``threshold`` times from random inputs."""
        dead = torch.where(self._usage < threshold)[0]
        if len(dead) and len(x):
            idx = torch.randint(0, len(x), (len(dead),), device=x.device)
            self.codebook.data[dead] = x[idx]
        self._usage.zero_()


class RQVAE(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        latent_dim: int = 32,
        hidden=(512, 256, 128),
        num_levels: int = 3,
        codebook_size: int = 256,
        commitment: float = 0.25,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.encoder = _mlp([input_dim, *hidden, latent_dim])
        self.decoder = _mlp([latent_dim, *reversed(hidden), input_dim])
        self.quantizers = nn.ModuleList(
            [VectorQuantizer(codebook_size, latent_dim, commitment) for _ in range(num_levels)]
        )

    def quantize(self, z: torch.Tensor):
        residual = z
        quantized = torch.zeros_like(z)
        codes, vq_loss = [], 0.0
        for vq in self.quantizers:
            q, c, loss = vq(residual)          # q is the RAW codebook vector
            quantized = quantized + q
            residual = residual - q.detach()   # stop-grad residual (standard RQ-VAE):
                                               # keeps d(residual)/d(z)=1 at every level
            codes.append(c)
            vq_loss = vq_loss + loss
        # one straight-through estimator on the accumulated quantized vector so the
        # decoder/reconstruction gradient reaches the encoder cleanly.
        quantized_st = z + (quantized - z).detach()
        return quantized_st, torch.stack(codes, 1), vq_loss  # codes: (B, L)

    @torch.no_grad()
    def init_codebooks(self, x: torch.Tensor, iters: int = 10):
        """k-means-initialise EVERY level's codebook on its own residuals.

        The paper initialises codebooks with k-means to avoid collapse; doing it
        only for level 0 leaves later codebooks poorly seeded. We encode the data
        once, then for each level run k-means on the running residual and subtract
        the resulting assignment before moving to the next level.
        """
        self.eval()
        z = self.encoder(x)
        residual = z
        for vq in self.quantizers:
            vq.kmeans_init(residual, iters=iters)
            d = torch.cdist(residual, vq.codebook)
            q = vq.codebook[d.argmin(1)]
            residual = residual - q

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        quantized, codes, vq_loss = self.quantize(z)
        x_hat = self.decoder(quantized)
        recon = F.mse_loss(x_hat, x)
        return x_hat, codes, recon + vq_loss, {"recon": recon.item(), "vq": float(vq_loss.detach())}

    @torch.no_grad()
    def encode_codes(self, x: torch.Tensor) -> torch.Tensor:
        """Return integer codes (B, num_levels) for a batch of embeddings."""
        self.eval()
        z = self.encoder(x)
        _, codes, _ = self.quantize(z)
        return codes
