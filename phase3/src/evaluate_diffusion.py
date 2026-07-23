"""Held-out functionals for the P10 diffusion policy.

TWO outcomes, both reported:

  PRIMARY (matched to the BC-Transformer, so the two policy classes are comparable at all):
      plain_loss       = mean L2 between the EXECUTED DDIM action and the ground-truth action,
                         over the target cluster's 10 held-out demos
      transport_loss / interaction_loss = the same L2, phase-masked (identical mask definitions)

  DIFFUSION-NATIVE SECONDARY:
      denoise_loss     = the DDPM epsilon-prediction MSE evaluated with the FIXED (t, eps) bank
                         (the same bank the attribution uses), over the same held-out frames

The key was choosing the PRIMARY to be the outcome on which the BC-Transformer null was
established. Using a diffusion-only outcome would have made the two policy classes incomparable
and the replication meaningless.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join("/mnt/sdb/ljc/RoboTDA-X", "src"))
import bootstrap  # noqa: F401,E402
import dataset  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diffusion_policy as DP  # noqa: E402
from diffusion_data import heldout_chunk_bank  # noqa: E402
from p10_bank import load_bank  # noqa: E402


def load_model(ckpt_path, device="cuda"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = DP.build(ck["state_dim"], ck["cfg"]).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m


@torch.no_grad()
def per_frame(model, bank, device="cuda", batch=512, n_ddim=None):
    """Per-frame (L2 on the executed DDIM action, denoising loss on the fixed bank)."""
    n_ddim = n_ddim or DP.N_DDIM_STEPS
    tb, eb = load_bank(device)
    l2 = np.empty(bank.n, dtype=np.float64)
    dn = np.empty(bank.n, dtype=np.float64)
    for i in range(0, bank.n, batch):
        s = torch.from_numpy(bank.S[i:i + batch]).to(device)
        l = torch.from_numpy(bank.L[i:i + batch]).to(device)
        a1 = torch.from_numpy(bank.A1[i:i + batch]).to(device)
        ac = torch.from_numpy(bank.AC[i:i + batch]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v2 = model.l2(s, l, a1, n_steps=n_ddim)
            vd = model.bank_loss(s, l, ac, tb, eb)
        l2[i:i + batch] = v2.float().cpu().numpy()
        dn[i:i + batch] = vd.float().cpu().numpy()
    return l2, dn


def _masked(v, m):
    return float((v * m).sum() / max(m.sum(), 1))


def heldout_losses(model, device="cuda", variant="base", n_ddim=None):
    bank = heldout_chunk_bank(variant)
    l2, dn = per_frame(model, bank, device, n_ddim=n_ddim)
    _, by_c = dataset.heldout_pool()
    id2k = {d: k for k, d in enumerate(bank.ids)}
    tr = bank.masks[variant]["transport"]
    it = bank.masks[variant]["interaction"]
    res = {}
    for c, dd in by_c.items():
        rows = np.concatenate([np.nonzero(bank.owner == id2k[d])[0] for d in dd])
        q, n, t, i = l2[rows], dn[rows], tr[rows], it[rows]
        res[c] = {
            "plain_loss": float(q.mean()),                  # L2 on the executed DDIM action
            "transport_loss": _masked(q, t),
            "interaction_loss": _masked(q, i),
            "denoise_loss": float(n.mean()),                # diffusion-native secondary
            "denoise_transport": _masked(n, t),
            "denoise_interaction": _masked(n, i),
            "n_frames": int(len(rows)),
        }
    return res
