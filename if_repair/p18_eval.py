"""PASS 18 -- the campaign-T outcome: held-out loss on the ladder's own 100-demo bank.

The existing outcome path (`retrain.heldout_frame_losses` + `retrain.aggregate_outcomes`) scores
against the old corpus's fixed 90-demo, 9-cluster bank and aggregates per CLUSTER. Campaign T
trains only on `libero_goal` and holds out 100 `libero_goal` demos of its own, so it needs its own
bank and its own aggregation. Nothing here changes the old path; the two coexist.

WHAT IS STORED, AND WHY IT IS PER-FRAME. Campaign A's lesson (see retrain.py's docstring): storing
only pre-aggregated functionals means any RE-WEIGHTED target invented later has no matching outcome
and cannot be honestly scored. So every campaign-T retrain writes PER-FRAME l2 and nll over the
fixed 100-demo bank, and every functional below is derived from that on demand, forever, at zero
further GPU cost. At 0.1s per retrain (results/p18_costmodel.json) the eval is free anyway.

THE PRIMARY FUNCTIONAL is `plain_loss`: the unweighted mean per-frame l2 over the whole bank. The
per-task breakdown and the transport/interaction splits are carried because they cost nothing, but
they are DESCRIPTIVE -- the prereg names `plain_loss` and only `plain_loss`.
"""
from __future__ import annotations

import functools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402
from if_repair import p18_corpus as C  # noqa: E402

D.add_repo_paths()
import dataset  # noqa: E402
import evaluate as EV  # noqa: E402


@functools.lru_cache(maxsize=1)
def bank(variant="base"):
    """The ladder's held-out bank: 100 libero_goal demos, 10 of every task.

    Disjoint from every rung by construction (p18_corpus asserts it, and so do the tests).
    """
    return dataset.Bank(list(C.eval_bank()), variants=(variant,))


@functools.lru_cache(maxsize=1)
def frame_index():
    """Row -> (task, phase-mask) map for the bank. Identical for every campaign-T run."""
    b = bank()
    task_of_row = np.empty(b.n, dtype=object)
    for k, demo_id in enumerate(b.ids):
        task_of_row[b.owner == k] = demo_id.split("/")[1]
    return {"task_of_row": task_of_row,
            "transport": b.masks["base"]["transport"],
            "interaction": b.masks["base"]["interaction"],
            "owner": b.owner, "ids": list(b.ids), "n": int(b.n)}


def heldout_frame_losses(model, device="cuda"):
    """Per-frame (l2, nll) over the ladder's bank."""
    return EV.per_frame(model, bank(), device=device)


def aggregate(l2, nll, fidx=None):
    """Per-frame losses -> the campaign-T outcome record.

    `plain_loss` is the primary. Everything else is descriptive and carried because it is free.
    """
    fidx = fidx or frame_index()
    t, i = fidx["transport"], fidx["interaction"]
    out = {
        "plain_loss": float(l2.mean()),
        "plain_loss_nll": float(nll.mean()),
        "transport_loss": float((l2 * t).sum() / max(t.sum(), 1)),
        "interaction_loss": float((l2 * i).sum() / max(i.sum(), 1)),
        "n_frames": int(l2.size),
        "by_task": {},
    }
    for task in sorted(set(fidx["task_of_row"])):
        rows = fidx["task_of_row"] == task
        out["by_task"][task] = {"plain_loss": float(l2[rows].mean()),
                                "n_frames": int(rows.sum())}
    return out


def per_demo_losses(l2, fidx=None):
    """Mean l2 per held-out DEMO. Not an outcome -- a diagnostic for the variance pilot."""
    fidx = fidx or frame_index()
    return {d: float(l2[fidx["owner"] == k].mean()) for k, d in enumerate(fidx["ids"])}
