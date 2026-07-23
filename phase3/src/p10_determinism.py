"""P10 DETERMINISM GATE -- must pass before ANY P10 experiment runs. Preregistered.

Phase 1's entire instrument rests on rollouts being deterministic: every model is evaluated from
IDENTICAL initial conditions, so a difference in success between two models is caused by the
models, not by rollout sampling. The BC-Transformer gets this from argmax. A diffusion policy
only gets it if BOTH of these hold:

    (a) the sampler is DDIM with eta = 0   -> no noise injected between denoising steps
    (b) the initial latent x_T is FIXED    -> not freshly sampled per call

We do both (diffusion_policy.ddim_chunk). This script PROVES it end-to-end, the same way Phase 2
proved the 50-init-state defect: replay the same episodes twice and require BIT-IDENTICAL results
-- not just the same success flags, but the same STEP COUNTS, which is a far stricter test (a
near-floor model would match trivially on flags alone by failing everything at the full horizon).

A failure here is an INSTRUMENT DEFECT -> stop and write PHASE3_DEFECT.md.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3lib as L
from p3lib import P3_RESULTS

sys.path.insert(0, os.path.join(L.ROOT, "src"))
import dataset  # noqa: E402


def main():
    ck = sys.argv[1] if len(sys.argv) > 1 else None
    if ck is None:
        raise SystemExit("usage: p10_determinism.py <path/to/final.pt>")

    import rollout_diffusion as RD
    suite = dataset.suite_of_cluster()["C1"]
    task = dataset.probe_tasks()["C1"][0]

    r1, i1 = RD.run_rollouts(ck, [(suite, task)], 6, workers=1)
    r2, i2 = RD.run_rollouts(ck, [(suite, task)], 6, workers=1)

    # step counts are returned per task by the worker; re-run to capture them explicitly
    RD._init_worker(ck, dataset.obj_pad_dim(), *dataset.norm_stats(), 10)
    a = RD._rollout_task((suite, task, list(range(6)), 600))
    b = RD._rollout_task((suite, task, list(range(6)), 600))

    same_flags = a["success"] == b["success"]
    same_steps = a["steps"] == b["steps"]

    # the 50-init-state wrap check (Phase-2 defect): episodes 0..5 vs 50..55 must be identical
    c = RD._rollout_task((suite, task, list(range(50, 56)), 600))
    wrap_identical = (c["success"] == a["success"]) and (c["steps"] == a["steps"])

    out = {
        "stage": "P10 determinism gate",
        "checkpoint": ck, "task": task, "n_episodes": 6,
        "run1_success": a["success"], "run1_steps": a["steps"],
        "run2_success": b["success"], "run2_steps": b["steps"],
        "BIT_IDENTICAL_success_flags": bool(same_flags),
        "BIT_IDENTICAL_step_counts": bool(same_steps),
        "GATE": bool(same_flags and same_steps),
        "why_step_counts": ("Step counts are the strict test: a near-floor model matches on "
                            "success flags trivially (it fails every episode at the full horizon), "
                            "so flags alone would not prove determinism."),
        "episode_50_wraps_to_0": {
            "episodes_50_55_success": c["success"], "episodes_50_55_steps": c["steps"],
            "identical_to_episodes_0_5": bool(wrap_identical),
            "meaning": ("LIBERO supplies 50 init states per task, and rollout selects them with "
                        "ep % 50 -- so episode 50 IS episode 0. This re-confirms the Phase-2 "
                        "instrument defect for the diffusion policy: there are at most 50 "
                        "distinct episodes per task, and P10 never exceeds that."),
        },
        "sampler": "DDIM, eta = 0, fixed initial latent (seed 12345)",
    }
    L.atomic_write_json(os.path.join(P3_RESULTS, "p10_determinism.json"), out)

    print("=" * 80)
    print("P10 DETERMINISM GATE")
    print("=" * 80)
    print(f"  run 1 success: {a['success']}")
    print(f"  run 2 success: {b['success']}")
    print(f"  run 1 steps  : {a['steps']}")
    print(f"  run 2 steps  : {b['steps']}")
    print(f"  bit-identical flags: {same_flags}   steps: {same_steps}")
    print(f"  episodes 50-55 replay 0-5 exactly: {wrap_identical} (the 50-init-state wrap)")
    print(f"\n  GATE: {out['GATE']}")
    print("=" * 80)
    return 0 if out["GATE"] else 1


if __name__ == "__main__":
    sys.exit(main())
