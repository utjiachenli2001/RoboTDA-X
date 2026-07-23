"""End-to-end confirmation of the PHASE2_DEFECT.md init-state duplication claim.

Claim: with a deterministic policy (policy.act = argmax-mode mean) and a deterministic env,
episode `ep` reuses init state `ep % 50`, so episodes 50..54 must be BIT-IDENTICAL replays of
episodes 0..4.

Also checks the pipeline really is deterministic by running episodes 0..4 twice.

Writes phase2/results/p4_determinism_check.json. Prints the vectors that go into the report.
"""
import json
import os
import sys

sys.path.insert(0, "/mnt/sdb/ljc/RoboTDA-X/src")
import bootstrap  # noqa: F401
from bootstrap import ROOT, RUNS  # noqa: E402
import rollout as R  # noqa: E402
import dataset  # noqa: E402

# A DISCRIMINATING model is required: a near-floor checkpoint fails every episode at the full
# 600-step horizon, so its success/step vectors would match trivially and prove nothing. The
# Stage-C Q=50 co-train model sits near 50% success, so its per-episode vector is MIXED and an
# exact match is real evidence of replay.
CKPT = os.path.join(RUNS, "stage_C/Q50_cotrain_s101/final.pt")
SUITE, TASK = "libero_goal", "open_the_middle_drawer_of_the_cabinet"


def run(eps):
    """Run exactly the Phase-1 rollout worker on a given list of episode indices."""
    mean, std = dataset.norm_stats()
    R._init_worker(CKPT, dataset.obj_pad_dim(), mean, std)
    r = R._rollout_task((SUITE, TASK, list(eps), R.HORIZON))
    assert not r["err"], r["err"]
    return r["success"], r["steps"]


if __name__ == "__main__":
    N = 10
    s_a, t_a = run(range(0, N))            # episodes 0..9
    s_b, t_b = run(range(50, 50 + N))      # episodes 50..59 -> init rows 0..9
    s_a2, t_a2 = run(range(0, N))          # episodes 0..9 again -> determinism control

    dup = (s_a == s_b) and (t_a == t_b)
    det = (s_a == s_a2) and (t_a == t_a2)
    # the comparison is only meaningful if the vectors are NOT degenerate (all-fail/all-timeout)
    discriminating = (len(set(s_a)) > 1) or (len(set(t_a)) > 1)

    out = {
        "vectors_are_discriminating": bool(discriminating),
        "note": ("a near-floor model fails every episode at the full horizon and would match "
                 "trivially; this checkpoint's per-episode outcomes vary, so the match is real"),
        "ckpt": CKPT, "suite": SUITE, "task": TASK,
        "episodes_0_9": {"success": s_a, "steps": t_a},
        "episodes_50_59": {"success": s_b, "steps": t_b},
        "episodes_0_9_repeat": {"success": s_a2, "steps": t_a2},
        "pipeline_is_deterministic": bool(det),
        "eps_50_54_duplicate_eps_0_4": bool(dup),
        "DEFECT_CONFIRMED": bool(det and dup and discriminating),
    }
    json.dump(out, open(os.path.join(ROOT, "phase2/results/p4_determinism_check.json"), "w"),
              indent=1)
    print(f"  ep 0-9    success={s_a}\n            steps={t_a}")
    print(f"  ep 50-59  success={s_b}\n            steps={t_b}")
    print(f"  ep 0-9 (repeat) success={s_a2}")
    print(f"  vectors discriminating (not all-fail): {discriminating}")
    print(f"  pipeline deterministic: {det}")
    print(f"  episodes 50-59 duplicate 0-9: {dup}")
    print(f"  DEFECT_CONFIRMED: {det and dup and discriminating}")
