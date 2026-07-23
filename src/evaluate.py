"""Held-out loss functionals (spec §3 ii/iii): plain, transport-masked, interaction-masked.

For a target cluster t, the functional over its 10 held-out demos is

    plain(t)        = mean_over_frames  NLL(a_f | window_f)
    transport(t)    = sum_f m^transport_f * NLL_f  /  sum_f m^transport_f
    interaction(t)  = sum_f m^interaction_f * NLL_f / sum_f m^interaction_f

(i.e. per-timestep loss x phase indicator, normalized by the mask sum -- exactly the spec).
Frames are pooled across the cluster's 10 held-out demos (a demo-length-weighted mean).
"""
import os
import sys
import functools
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bootstrap  # noqa: F401
import dataset
import policy as P


@functools.lru_cache(maxsize=8)
def heldout_bank(variant="base"):
    ids, _ = dataset.heldout_pool()
    return dataset.Bank(ids, variants=(variant,))


def load_model(ckpt_path, device="cuda"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = P.build(ck["state_dim"], ck["cfg"]).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m


@torch.no_grad()
def per_frame(model, bank, device="cuda", batch=1024):
    """Per-frame (l2, nll). L2 is the PRIMARY functional; NLL is kept for transparency."""
    l2 = np.empty(bank.n, dtype=np.float64)
    nll = np.empty(bank.n, dtype=np.float64)
    for i in range(0, bank.n, batch):
        s = torch.from_numpy(bank.S[i:i + batch]).to(device)
        l = torch.from_numpy(bank.L[i:i + batch]).to(device)
        a = torch.from_numpy(bank.A[i:i + batch]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v2 = model.l2(s, l, a)
            vn = model.nll(s, l, a)
        l2[i:i + batch] = v2.float().cpu().numpy()
        nll[i:i + batch] = vn.float().cpu().numpy()
    return l2, nll


def per_frame_nll(model, bank, device="cuda", batch=1024):
    return per_frame(model, bank, device, batch)[1]


def _masked(v, m):
    return float((v * m).sum() / max(m.sum(), 1))


def heldout_losses(model, device="cuda", variant="base"):
    """-> {cluster: {plain/transport/interaction loss (L2, PRIMARY) + the NLL versions}}.

    plain_loss/transport_loss/interaction_loss are the L2 functional (see policy.l2 for why).
    *_nll keys retain the GMM NLL numbers so the instability is auditable, but they are NOT
    used as study outcomes.
    """
    bank = heldout_bank(variant)
    l2, nll = per_frame(model, bank, device)
    _, by_c = dataset.heldout_pool()
    id2k = {d: k for k, d in enumerate(bank.ids)}
    tr = bank.masks[variant]["transport"]
    it = bank.masks[variant]["interaction"]
    res = {}
    for c, dd in by_c.items():
        rows = np.concatenate([np.nonzero(bank.owner == id2k[d])[0] for d in dd])
        q, n, t, i = l2[rows], nll[rows], tr[rows], it[rows]
        res[c] = {
            "plain_loss": float(q.mean()),                 # L2 (primary)
            "transport_loss": _masked(q, t),
            "interaction_loss": _masked(q, i),
            "plain_loss_nll": float(n.mean()),             # GMM NLL (audit only; unstable)
            "transport_loss_nll": _masked(n, t),
            "interaction_loss_nll": _masked(n, i),
            "median_nll": float(np.median(n)),
            "n_frames": int(len(rows)),
            "transport_frac": float(t.mean()),
        }
    return res


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", default="base")
    a = ap.parse_args()
    m = load_model(a.ckpt)
    print(json.dumps(heldout_losses(m, variant=a.variant), indent=1))
