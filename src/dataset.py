"""Corpus/dataset layer: demo IDs, state featurization, normalization, windows, phase masks.

Canonical demo id:  "<suite>/<task>/<demo_k>"   (globally unique)

State (frozen, see ENV.md):
    s_t = [ proprio(16) , object-state zero-padded to obj_pad_dim(112) ]   -> 128-D
    z-normalized with mean/std computed ONCE over the 135-demo training pool
    (results/norm_stats.npz). The SAME frozen stats are used by every run in the study,
    including Stage C runs that see up to 500 Goal demos, so that masks/quantities never
    change the input scaling (which would confound the comparison).

Policy conditioning window: L=10 frames ending at t (left-padded at the demo start by
repeating frame 0), predicting a_t. This matches rollout-time conditioning exactly.
"""
import os
import sys
import json
import functools
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bootstrap  # noqa: F401
from bootstrap import PROC, RESULTS
import phases
import taskemb

CTX = 10                 # context window length (frames)
NORM_STATS = os.path.join(RESULTS, "norm_stats.npz")
MANIFEST = os.path.join(RESULTS, "corpus_manifest.json")


# ---------------------------------------------------------------- manifest helpers
@functools.lru_cache(maxsize=1)
def manifest():
    return json.load(open(MANIFEST))


@functools.lru_cache(maxsize=1)
def obj_pad_dim():
    return int(manifest()["obj_pad_dim"])


@functools.lru_cache(maxsize=1)
def state_dim():
    return int(manifest()["state_dim"])


def did(suite, task, demo):
    return f"{suite}/{task}/{demo}"


def parse_did(d):
    return d.split("/")


@functools.lru_cache(maxsize=1)
def cluster_of_task():
    """{task -> cluster}"""
    out = {}
    for c in manifest()["clusters"]:
        for t in c["tasks"]:
            out[t] = c["cluster"]
    return out


@functools.lru_cache(maxsize=1)
def train_pool():
    """Ordered list of the 135 training demo ids, and {cluster -> [demo ids]}."""
    ids, by_c = [], {}
    for c in manifest()["clusters"]:
        cl = []
        for d in c["train_demos"]:
            i = did(c["suite"], d["task"], d["demo"])
            ids.append(i)
            cl.append(i)
        by_c[c["cluster"]] = cl
    return ids, by_c


@functools.lru_cache(maxsize=1)
def heldout_pool():
    ids, by_c = [], {}
    for c in manifest()["clusters"]:
        cl = []
        for d in c["heldout_demos"]:
            i = did(c["suite"], d["task"], d["demo"])
            ids.append(i)
            cl.append(i)
        by_c[c["cluster"]] = cl
    return ids, by_c


@functools.lru_cache(maxsize=1)
def probe_tasks():
    """{cluster -> [3 task names]}"""
    return {c["cluster"]: list(c["probe_tasks"]) for c in manifest()["clusters"]}


@functools.lru_cache(maxsize=1)
def clusters():
    return [c["cluster"] for c in manifest()["clusters"]]


@functools.lru_cache(maxsize=1)
def suite_of_cluster():
    return {c["cluster"]: c["suite"] for c in manifest()["clusters"]}


# ---------------------------------------------------------------- raw demo loading
@functools.lru_cache(maxsize=4096)
def load_raw(demo_id):
    """-> dict(state(T,128) UNNORMALIZED, actions(T,7), proprio(T,16))"""
    suite, task, demo = parse_did(demo_id)
    z = np.load(os.path.join(PROC, suite, task, f"{demo}.npz"))
    P, O, A = z["proprio"], z["object"], z["actions"]
    D = obj_pad_dim()
    if O.shape[1] < D:
        O = np.concatenate([O, np.zeros((O.shape[0], D - O.shape[1]), np.float32)], 1)
    elif O.shape[1] > D:
        O = O[:, :D]
    return {"state": np.concatenate([P, O], 1).astype(np.float32),
            "actions": A.astype(np.float32), "proprio": P.astype(np.float32)}


def phase_mask(demo_id, variant="base", kind="transport"):
    """Boolean (T,) mask for the given phase + threshold variant."""
    P = load_raw(demo_id)["proprio"]
    inter, trans = phases.phase_masks(P, **phases.VARIANTS[variant])
    return trans if kind == "transport" else inter


# ---------------------------------------------------------------- normalization
def build_norm_stats(force=False):
    if os.path.exists(NORM_STATS) and not force:
        return
    ids, _ = train_pool()
    S = np.concatenate([load_raw(i)["state"] for i in ids], 0)   # (sum T, 128)
    mean = S.mean(0)
    std = S.std(0)
    std = np.maximum(std, 1e-3)          # floor: constant dims -> unit scale
    np.savez(NORM_STATS, mean=mean.astype(np.float32), std=std.astype(np.float32),
             n_demos=len(ids), n_frames=S.shape[0])
    print(f"[norm] from {len(ids)} train-pool demos, {S.shape[0]} frames -> {NORM_STATS}")


@functools.lru_cache(maxsize=1)
def norm_stats():
    if not os.path.exists(NORM_STATS):
        build_norm_stats()
    z = np.load(NORM_STATS)
    return z["mean"], z["std"]


def normalize(state):
    m, s = norm_stats()
    return (state - m) / s


# ---------------------------------------------------------------- window tensors
def demo_windows(demo_id):
    """All CTX-length windows of a demo, ending at each t.

    Returns states (T, CTX, 128) normalized, actions (T, 7), task (str).
    Window for t = frames [t-CTX+1 .. t], left-padded by repeating frame 0.
    """
    suite, task, _ = parse_did(demo_id)
    r = load_raw(demo_id)
    S = normalize(r["state"])                        # (T,128)
    A = r["actions"]                                 # (T,7)
    T = S.shape[0]
    pad = np.repeat(S[:1], CTX - 1, axis=0)          # (CTX-1,128)
    SP = np.concatenate([pad, S], 0)                 # (T+CTX-1,128)
    idx = np.arange(T)[:, None] + np.arange(CTX)[None, :]   # (T,CTX)
    return SP[idx].astype(np.float32), A, task


class Bank:
    """In-memory bank of windowed demos + their task embeddings + phase masks."""

    def __init__(self, demo_ids, variants=("base",)):
        self.ids = list(demo_ids)
        emb = taskemb.load()
        self.S, self.A, self.L, self.owner = [], [], [], []
        self.masks = {v: {"transport": [], "interaction": []} for v in variants}
        for k, d in enumerate(self.ids):
            s, a, task = demo_windows(d)
            self.S.append(s)
            self.A.append(a)
            self.L.append(np.repeat(emb[task][None, :], a.shape[0], 0))
            self.owner.append(np.full(a.shape[0], k, dtype=np.int64))
            P = load_raw(d)["proprio"]
            for v in variants:
                inter, trans = phases.phase_masks(P, **phases.VARIANTS[v])
                self.masks[v]["transport"].append(trans)
                self.masks[v]["interaction"].append(inter)
        self.S = np.concatenate(self.S, 0)           # (N,CTX,128)
        self.A = np.concatenate(self.A, 0)           # (N,7)
        self.L = np.concatenate(self.L, 0)           # (N,384)
        self.owner = np.concatenate(self.owner, 0)   # (N,) demo index
        for v in variants:
            for k in ("transport", "interaction"):
                self.masks[v][k] = np.concatenate(self.masks[v][k], 0)   # (N,) bool
        self.n = self.S.shape[0]

    def __len__(self):
        return self.n

    def demo_slices(self):
        """{demo_id -> np.ndarray of row indices}"""
        return {d: np.nonzero(self.owner == k)[0] for k, d in enumerate(self.ids)}


if __name__ == "__main__":
    build_norm_stats()
    ids, by_c = train_pool()
    ho, ho_c = heldout_pool()
    print(f"train pool {len(ids)}  heldout {len(ho)}  state_dim {state_dim()}")
    m, s = norm_stats()
    print("norm mean[:4]", np.round(m[:4], 3), "std[:4]", np.round(s[:4], 3))
    # transport fraction distribution per cluster (spec §2 redundancy check)
    print("\ntransport fraction per cluster (base thresholds):")
    rows = []
    for c, dd in by_c.items():
        fr = np.array([phase_mask(d, "base", "transport").mean() for d in dd])
        rows.append((c, fr.mean(), fr.std()))
        print(f"  {c}: mean={fr.mean():.3f} std={fr.std():.3f} min={fr.min():.3f} max={fr.max():.3f}")
    allfr = np.array([phase_mask(d, "base", "transport").mean() for d in ids])
    print(f"  ALL: mean={allfr.mean():.3f} std={allfr.std():.3f}")
    if allfr.std() < 0.05:
        print("  !! FLAG: phase contrast under-identified (std < 0.05)")
