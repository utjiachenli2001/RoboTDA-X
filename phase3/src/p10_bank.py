"""The FIXED paired (t, epsilon) noise bank for P10 diffusion attribution.

WHY A FIXED BANK (preregistered). A diffusion training loss is an EXPECTATION over a random
timestep t and a random noise draw epsilon. If the train-side and test-side gradients were each
computed with fresh random (t, eps), their inner product would be dominated by that sampling
noise -- TracIn/TRAK would be measuring the RNG, not the data. Fixing ONE bank of (t, eps) pairs
and using it for BOTH sides:

  * removes the sampling variance from the inner product (MOTIVE-style variance reduction), and
  * removes the timestep-induced bias -- every demo and every target is scored at exactly the
    same set of noise levels, so a demo cannot look influential merely because it drew easier t's.

The bank is drawn ONCE with default_rng(1031), STRATIFIED over t (evenly spread across the
diffusion horizon), frozen to disk with a SHA-256, and never redrawn.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3lib as L
from p3lib import P3_RESULTS

import diffusion_policy as DP  # noqa: E402

BANK_SEED = 1031          # preregistered
BANK_K = 32               # preregistered
BANK_PATH = os.path.join(P3_RESULTS, "p10_noise_bank.json")
_CACHE = {}


def build_bank(K=BANK_K, seed=BANK_SEED, T=DP.N_TRAIN_STEPS, H=DP.H_CHUNK, force=False):
    if os.path.exists(BANK_PATH) and not force:
        return json.load(open(BANK_PATH))
    rng = np.random.default_rng(seed)
    # STRATIFIED over t: one t per equal-width bin of [1, T], jittered inside its bin
    edges = np.linspace(0, T, K + 1)
    ts = []
    for k in range(K):
        lo, hi = edges[k], edges[k + 1]
        ts.append(int(np.clip(np.floor(rng.uniform(lo, hi)) + 1, 1, T)))
    eps = rng.standard_normal((K, H, DP.ACTION_DIM)).astype(np.float32)
    bank = {
        "seed": seed, "K": K, "T": T, "H": H, "action_dim": DP.ACTION_DIM,
        "stratification": "one t per equal-width bin of [1,T], jittered within the bin",
        "t": ts, "eps": eps.tolist(),
        "purpose": ("the SAME (t, eps) bank is used for BOTH the train-side and the test-side "
                    "gradients, so the inner product measures the data, not the RNG"),
    }
    L.atomic_write_json(BANK_PATH, bank)
    bank["sha256"] = L.sha256_file(BANK_PATH)
    L.atomic_write_json(BANK_PATH, bank)
    print(f"[bank] wrote {BANK_PATH}: K={K} t={ts[:6]}... sha={bank['sha256'][:16]}")
    return bank


def load_bank(device="cuda"):
    """-> (t_bank (K,) long, eps_bank (K,H,7) float) on `device`."""
    key = str(device)
    if key not in _CACHE:
        b = build_bank()
        t = torch.tensor(b["t"], dtype=torch.long, device=device)
        e = torch.tensor(np.array(b["eps"], dtype=np.float32), device=device)
        _CACHE[key] = (t, e)
    return _CACHE[key]


if __name__ == "__main__":
    b = build_bank(force="--force" in sys.argv)
    print(f"bank K={b['K']} T={b['T']} H={b['H']}")
    print(f"t values (stratified): {b['t']}")
    print(f"sha256: {b.get('sha256')}")
