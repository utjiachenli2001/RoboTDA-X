"""PASS 18 -- gradient attribution features for campaign U (the H1 and H1f arms).

WHAT THIS COMPUTES. For each pool and each ensemble member, a per-demonstration GradDot score

    score_i = < grad_theta NLL(demo_i) , grad_theta L2(eval bank) >

using the repo's own primitives -- `attribution.demo_gradient` for the training-side gradient and
`attribution.target_gradient` for the held-out L2 functional. No estimator is introduced here; the
prereg's contribution is evaluation, not method. The train/test pairing is deliberately asymmetric
(GMM NLL on the training side, L2 on the test side) exactly as `attribution.target_gradient`
documents: the mean GMM NLL is unusable as a test functional.

WHY THE OLD GRAM CACHES CANNOT BE USED. `if_repair/data.py`'s p6/p11 caches are keyed to the old
135-demo corpus's `train_ids`; campaign U's demos are `libero_goal` pool members that appear in
none of them, and its target is the fresh 100-demo eval bank rather than the 9-cluster one. The
features have to be computed for this corpus.

WHY NO CHECKPOINTS. `retrain.py` deliberately keeps no weights. Rather than reintroduce them, each
scoring model is trained IN PROCESS and its gradients taken immediately, so nothing touches disk
but the scores.

THE TWO ARMS.
  H1  (full-pool surrogate)  scores at pool P come from a model trained on all of pool P. This is
      the conventional arm, and its surrogate-to-retrain extrapolation distance GROWS with P --
      each retrain trains on 25 demos while the scorer saw P of them.
  H1f (fixed surrogate)      scores at every pool come from a POOL-50 model, holding that distance
      fixed. H1f is DESCRIPTIVE and carries no alpha: a fixed 50-demo surrogate sits inside the
      ladder, so the fraction of scored demos it has seen falls 100% -> 13.5% along the pools, and
      seen/unseen demos get systematically different gradient scores (self- vs cross-influence,
      which passes 4-7 of this project are the cautionary tale about). H1 - H1f therefore bounds
      the extrapolation effect rather than isolating it, and the prereg says so.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402
from if_repair import p18_corpus as C  # noqa: E402
from if_repair import p18_eval as EVT  # noqa: E402
from if_repair import retrain as R  # noqa: E402

D.add_repo_paths()
import dataset  # noqa: E402
import train as TR  # noqa: E402
import attribution as AT  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# Ensemble depth for the SCORING model. Averaging over members is what campaigns A-S did (the
# published GradDot figure is an E=20 mean); 5 is the affordable analogue here and is fixed before
# any score is computed.
MEMBERS = (7001, 7002, 7003, 7004, 7005)
SURROGATE_POOL = 50            # the H1f fixed surrogate


def _scores_from_model(model, demo_ids, tbank, slices):
    """-> np.ndarray of GradDot scores, one per demo in `demo_ids`."""
    hb = EVT.bank()
    tg = AT.target_gradient(model, hb, np.arange(hb.n), None)     # plain mean L2, whole bank
    out = np.empty(len(demo_ids), dtype=np.float64)
    for i, d in enumerate(demo_ids):
        g = AT.demo_gradient(model, tbank, slices[d])
        out[i] = float(torch.dot(g, tg).item())
    del tg
    torch.cuda.empty_cache()
    return out


def compute(pool, member, score_pool=None, cfg=None):
    """Train one scoring model on `score_pool` (default: `pool`) and score `pool`'s demos."""
    cfg = cfg or TR.load_cfg()
    score_pool = score_pool or pool
    train_ids = list(C.rung_demos(score_pool))
    demo_ids = list(C.rung_demos(pool))
    t0 = time.time()
    model, meta = R.train_one(train_ids, member, member, cfg)
    model.eval()
    tbank = dataset.Bank(demo_ids)
    slices = tbank.demo_slices()
    s = _scores_from_model(model, demo_ids, tbank, slices)
    del model, tbank
    torch.cuda.empty_cache()
    return s, {"pool": pool, "score_pool": score_pool, "member": member,
               "n_scored": len(demo_ids), "train_loss": meta["final_loss"],
               "wall_s": round(time.time() - t0, 1)}


def path_for(pool, arm):
    return os.path.join(RESULTS, f"p18_graddot_{arm}_pool{pool}.npz")


def run(pool, arm="H1", members=MEMBERS):
    """arm 'H1' -> surrogate is the pool itself; 'H1f' -> surrogate is always pool 50."""
    out = path_for(pool, arm)
    if os.path.exists(out):
        print(f"[p18/gram] skip {out}", flush=True)
        return
    sp = pool if arm == "H1" else SURROGATE_POOL
    cfg = TR.load_cfg()
    S, metas = [], []
    for m in members:
        s, meta = compute(pool, m, score_pool=sp, cfg=cfg)
        S.append(s)
        metas.append(meta)
        print(f"[p18/gram] {arm} pool={pool} surrogate={sp} member={m} "
              f"train_loss={meta['train_loss']:.3f} ({meta['wall_s']:.0f}s)", flush=True)
    S = np.stack(S)
    os.makedirs(RESULTS, exist_ok=True)
    np.savez_compressed(out, scores=S, mean=S.mean(0), demo_ids=np.array(C.rung_demos(pool)),
                        meta=json.dumps({"arm": arm, "pool": pool, "surrogate_pool": sp,
                                         "members": list(members), "per_member": metas}))
    print(f"[p18/gram] wrote {out}  shape={S.shape}", flush=True)


def load(pool, arm="H1"):
    z = np.load(path_for(pool, arm), allow_pickle=True)
    return {d: float(v) for d, v in zip(z["demo_ids"], z["mean"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, required=True)
    ap.add_argument("--arm", default="H1", choices=["H1", "H1f"])
    a = ap.parse_args()
    run(a.pool, a.arm)


if __name__ == "__main__":
    main()
