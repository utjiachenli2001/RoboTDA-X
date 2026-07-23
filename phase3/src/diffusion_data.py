"""Action-CHUNK data bank for the P10 diffusion policy.

Identical observations to the BC-Transformer (dataset.demo_windows -> the SAME CTX=10 windows,
z-normalized with the SAME frozen Phase-1 norm_stats, never refit). The only addition is the
action CHUNK target: for each frame t, the next H actions

    Achunk[t] = [a_t, a_{t+1}, ..., a_{t+H-1}]

right-padded at the demo's end by repeating the final action (the standard convention -- the arm
is at rest there, so the repeat is behaviourally correct rather than a fabrication).

A1[t] = a_t is kept separately: it is the ground truth for the EXECUTED-action L2 functional,
which is the outcome shared with the BC-Transformer arm.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join("/mnt/sdb/ljc/RoboTDA-X", "src"))
import bootstrap  # noqa: F401,E402
import dataset  # noqa: E402
import phases  # noqa: E402
import taskemb  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diffusion_policy import H_CHUNK  # noqa: E402


def demo_chunks(demo_id, H=H_CHUNK):
    """-> S (T,CTX,128) normalized, Achunk (T,H,7), A1 (T,7), task."""
    S, A, task = dataset.demo_windows(demo_id)          # the SAME windows the BC policy sees
    T = A.shape[0]
    idx = np.minimum(np.arange(T)[:, None] + np.arange(H)[None, :], T - 1)   # right-pad by repeat
    return S, A[idx].astype(np.float32), A.astype(np.float32), task


class ChunkBank:
    def __init__(self, demo_ids, H=H_CHUNK, variants=("base",)):
        self.ids = list(demo_ids)
        self.H = H
        emb = taskemb.load()
        S, AC, A1, L, owner = [], [], [], [], []
        self.masks = {v: {"transport": [], "interaction": []} for v in variants}
        for k, d in enumerate(self.ids):
            s, ac, a1, task = demo_chunks(d, H)
            S.append(s)
            AC.append(ac)
            A1.append(a1)
            L.append(np.repeat(emb[task][None, :], a1.shape[0], 0))
            owner.append(np.full(a1.shape[0], k, dtype=np.int64))
            P = dataset.load_raw(d)["proprio"]
            for v in variants:
                inter, trans = phases.phase_masks(P, **phases.VARIANTS[v])
                self.masks[v]["transport"].append(trans)
                self.masks[v]["interaction"].append(inter)
        self.S = np.concatenate(S, 0)                   # (N,CTX,128)
        self.AC = np.concatenate(AC, 0)                 # (N,H,7)
        self.A1 = np.concatenate(A1, 0)                 # (N,7)
        self.L = np.concatenate(L, 0)                   # (N,384)
        self.owner = np.concatenate(owner, 0)
        for v in variants:
            for k in ("transport", "interaction"):
                self.masks[v][k] = np.concatenate(self.masks[v][k], 0)
        self.n = self.S.shape[0]

    def __len__(self):
        return self.n

    def demo_slices(self):
        return {d: np.nonzero(self.owner == k)[0] for k, d in enumerate(self.ids)}


_HB = {}


def heldout_chunk_bank(variant="base"):
    if variant not in _HB:
        ids, _ = dataset.heldout_pool()
        _HB[variant] = ChunkBank(ids, variants=(variant,))
    return _HB[variant]
