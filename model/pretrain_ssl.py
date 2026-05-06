"""pretrain_ssl.py — SimCLR-style contrastive pretraining on cardiac signals.

Path B (Self-Supervised Pretraining):
  Encoder learns time-series features from a large UNLABELED corpus
  (CEBSDB b+m+p + PCG + ECG when available), then we fine-tune for the
  downstream SCG-3-class task.  Hypothesis: encoder sees more cardiac
  cycle variety → better generalization (less overfit to 19 subjects).

Augmentations form positive pairs (two views of the same window):
  * circular time shift ±20 samples
  * Gaussian noise (sigma=0.05)
  * random amplitude scale [0.8, 1.2]
  * random crop 224/256 with rezo-pad (keeps length 256)

Loss: NT-Xent, temperature 0.5.

Usage:
    python model/pretrain_ssl.py --data data_excl100/all.npz \
        --epochs 30 --bs 256 --tag ssl_cebs
"""
from __future__ import annotations
import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Encoder (matches SCGNetV2 conv backbone) + projection head
# ---------------------------------------------------------------------------
class ConvEncoder(nn.Module):
    """1D CNN backbone, returns 128-dim feature per sample (after GAP)."""

    def __init__(self, channels=(32, 64, 128), k_first: int = 5):
        super().__init__()
        c0, c1, c2 = channels
        self.conv0 = nn.Conv1d(1,  c0, k_first, padding=k_first // 2)
        self.bn0   = nn.BatchNorm1d(c0)
        self.conv1 = nn.Conv1d(c0, c1, 5, padding=2)
        self.bn1   = nn.BatchNorm1d(c1)
        self.conv2 = nn.Conv1d(c1, c2, 5, padding=2)
        self.bn2   = nn.BatchNorm1d(c2)
        self.feat_dim = c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool1d(F.relu(self.bn0(self.conv0(x))), 2)
        x = F.max_pool1d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool1d(F.relu(self.bn2(self.conv2(x))), 2)
        return x.mean(dim=2)              # GAP → (B, c2)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 128, hid_dim: int = 64, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, x): return self.net(x)


class SimCLRModel(nn.Module):
    def __init__(self, channels=(32, 64, 128)):
        super().__init__()
        self.encoder = ConvEncoder(channels)
        self.proj    = ProjectionHead(self.encoder.feat_dim, 64, 32)

    def forward(self, x):
        h = self.encoder(x)
        z = self.proj(h)
        return F.normalize(z, dim=1)      # unit-norm for cosine similarity


# ---------------------------------------------------------------------------
# NT-Xent loss
# ---------------------------------------------------------------------------
def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    """Symmetric NT-Xent over a batch of B positive pairs.

    Both z1/z2 are (B, D) unit-norm.  Returns scalar loss.
    """
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)                          # (2B, D)
    sim = z @ z.T / tau                                     # (2B, 2B)
    # Mask out self-similarity
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float("-inf"))
    # Positive pair indices: (i, i+B) and (i+B, i)
    targets = torch.arange(2 * B, device=z.device)
    targets = (targets + B) % (2 * B)
    return F.cross_entropy(sim, targets)


# ---------------------------------------------------------------------------
# Augmentations (per-sample, applied on float [-1,1] tensors)
# ---------------------------------------------------------------------------
def aug_view(x: torch.Tensor) -> torch.Tensor:
    """Fully-vectorized two-view augmentation. x: (B, 1, 256) float on any device.

    GPU-friendly: no Python for-loops over batch dim → one cuda dispatch per
    augmentation step.
    """
    B, _, L = x.shape
    dev = x.device

    # Random circular shift ±20 samples — vectorized via gather
    shifts = torch.randint(-20, 21, (B,), device=dev)        # (B,)
    base = torch.arange(L, device=dev).unsqueeze(0).expand(B, L)   # (B, L)
    idx_shift = (base - shifts.unsqueeze(1)) % L              # (B, L)
    out = torch.gather(x, 2, idx_shift.unsqueeze(1))          # (B, 1, L)

    # Gaussian noise
    out = out + torch.randn_like(out) * 0.05

    # Random amplitude scale
    scale = 0.8 + 0.4 * torch.rand(B, 1, 1, device=dev)
    out = out * scale

    # Random 32-sample temporal masking — vectorized via mask construction
    # For each sample i, drop window [s_i, s_i+32). Build mask of shape (B, L).
    starts = torch.randint(0, L - 32, (B,), device=dev)       # (B,)
    pos = torch.arange(L, device=dev).unsqueeze(0)            # (1, L)
    mask = (pos < starts.unsqueeze(1)) | (pos >= (starts + 32).unsqueeze(1))   # (B, L)
    out = out * mask.unsqueeze(1).to(out.dtype)

    return out.clamp(-1.5, 1.5)


# ---------------------------------------------------------------------------
# Dataset (loads X only; no labels needed for SSL)
# ---------------------------------------------------------------------------
class UnlabeledNPZ(Dataset):
    def __init__(self, path: Path):
        d = np.load(path, allow_pickle=True)
        self.X = torch.from_numpy(d["X"].astype(np.float32) / 127.0)
        # Optional sid for diagnostics
        self.sid = d["sid"] if "sid" in d.files else None

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data_excl100/all.npz"))
    p.add_argument("--out", type=Path, default=Path("model/ckpt"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temp", type=float, default=0.5)
    p.add_argument("--tag", type=str, default="ssl_cebs")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    ds = UnlabeledNPZ(args.data)
    print(f"loaded {args.data}: N={len(ds)}")
    dl = DataLoader(ds, args.bs, shuffle=True, drop_last=True, num_workers=0)

    device = torch.device(args.device)
    model = SimCLRModel(channels=(32, 64, 128)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SimCLR encoder params={n_params}  device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    t0 = time.time()
    best_loss = float("inf")
    for ep in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0; n_seen = 0
        for x in dl:
            x = x.to(device)
            v1 = aug_view(x)
            v2 = aug_view(x)
            z1 = model(v1)
            z2 = model(v2)
            loss = nt_xent_loss(z1, z2, tau=args.temp)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()
            loss_sum += loss.item() * x.size(0)
            n_seen += x.size(0)
        sch.step()
        avg_loss = loss_sum / max(n_seen, 1)
        elapsed = time.time() - t0
        print(f"ep {ep:02d}  ssl_loss={avg_loss:.4f}  ({elapsed:.0f}s)", flush=True)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"encoder_state": model.encoder.state_dict(),
                        "proj_state":    model.proj.state_dict(),
                        "channels": (32, 64, 128),
                        "feat_dim": model.encoder.feat_dim,
                        "epoch": ep, "ssl_loss": avg_loss,
                        "tag": args.tag},
                       args.out / f"ssl_{args.tag}.pt")
    print(f"\nbest ssl_loss = {best_loss:.4f}  -> {args.out}/ssl_{args.tag}.pt")


if __name__ == "__main__":
    main()
