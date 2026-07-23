"""STAGE P0 -- intake audit.

Recomputes TWO Phase-1 headline numbers from RAW per-run artifacts (runs/*/outcomes.json,
runs/*/cluster_eval.json), NOT from the summary JSONs, and compares against REPORT.md.

  (a) Gate-1 best rho (report: +0.252)   <- runs/stage_D/*/outcomes.json
                                            + results/stage_D_influence_C1.parquet
                                            + results/stage_D_mask_manifest.json
  (b) Stage-C margins (+2.3 / +0.8 / -3.2) <- runs/stage_C/*/cluster_eval.json

Mismatch (beyond print-rounding tolerance) => write phase2/INTAKE_MISMATCH.md and STOP.
"""
import json
import os
import sys
from fractions import Fraction

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/sdb/ljc/RoboTDA-X/src")
ROOT = "/mnt/sdb/ljc/RoboTDA-X"
P2 = os.path.join(ROOT, "phase2")

from lds import spearman, spearman_p_onesided, mask_pred_score  # noqa: E402

rep = {}

# ----------------------------------------------------------------------------- (a) Gate 1
mans = json.load(open(f"{ROOT}/results/stage_D_mask_manifest.json"))
masks = mans["masks"]

# --- raw outcomes: mean over the 2 seeds of the L2 plain loss, negated (higher = better)
raw = {}
for m in masks:
    mid = m["mask_id"]
    per_seed = []
    for s in (101, 102):
        f = f"{ROOT}/runs/stage_D/{mid}_s{s}/outcomes.json"
        o = json.load(open(f))["outcomes"]["C1"]
        per_seed.append(o["plain_loss"])
    raw[mid] = {"neg_plain_loss": -float(np.mean(per_seed)), "n_seeds": len(per_seed)}

# cross-check the raw recomputation against the archived mask_outcomes block
arch = json.load(open(f"{ROOT}/results/stage_D_gate1.json"))
mo_delta = max(abs(raw[k]["neg_plain_loss"] - v["neg_plain_loss"])
               for k, v in arch["mask_outcomes"].items())

# --- predicted mask scores from the archived per-demo influence
inf = pd.read_parquet(f"{ROOT}/results/stage_D_influence_C1.parquet")
inf = inf[inf.functional == "plain"]

gate1 = {}
for attr in sorted(inf.attributor.unique()):
    sc = dict(zip(inf[inf.attributor == attr].demo_id, inf[inf.attributor == attr].score))
    pred = [mask_pred_score(sc, m["demos"]) for m in masks]
    out = [raw[m["mask_id"]]["neg_plain_loss"] for m in masks]
    r = spearman(pred, out)
    gate1[attr] = {"rho": r, "n": len(masks), "p_onesided": spearman_p_onesided(r, len(masks))}

best_attr = max(gate1, key=lambda a: gate1[a]["rho"])
best_rho = gate1[best_attr]["rho"]

rep["gate1"] = {
    "recomputed_from": "runs/stage_D/*/outcomes.json + stage_D_influence_C1.parquet",
    "per_attributor": gate1,
    "best_attributor": best_attr,
    "best_rho_recomputed": best_rho,
    "report_value": 0.252,
    "archived_value": arch["attributors"][best_attr]["neg_plain_loss"]["rho"],
    "max_abs_delta_vs_archived_mask_outcomes": mo_delta,
    "delta_vs_report": abs(best_rho - 0.252),
    "MATCH": abs(best_rho - 0.252) < 5e-4 and mo_delta < 1e-9,
}

# ----------------------------------------------------------------------------- (b) Stage C
# exact rational arithmetic, as in Phase 1
margins = {}
for Q in (15, 50, 490):
    per_cond = {}
    for cond in ("target", "cotrain"):
        vals = []
        for s in (101, 102, 103):
            f = f"{ROOT}/runs/stage_C/Q{Q}_{cond}_s{s}/cluster_eval.json"
            ce = json.load(open(f))
            # recompute cluster_success from per_task_success rather than trusting the field
            pts = ce["per_task_success"]
            recomputed = sum(Fraction(v).limit_denominator(10**6) for v in pts.values()) / len(pts)
            assert abs(float(recomputed) - ce["cluster_success"]) < 1e-9, (f, recomputed, ce["cluster_success"])
            vals.append(Fraction(ce["cluster_success"]).limit_denominator(10**6))
        per_cond[cond] = vals
    seed_margins = [(c - t) * 100 for c, t in zip(per_cond["cotrain"], per_cond["target"])]
    mm = sum(seed_margins) / len(seed_margins)
    margins[Q] = {
        "target_only_pct_mean": float(sum(per_cond["target"]) / 3 * 100),
        "cotrain_pct_mean": float(sum(per_cond["cotrain"]) / 3 * 100),
        "margin_pts_mean_exact": str(mm),
        "margin_pts_mean": float(mm),
        "per_seed_margins": [float(x) for x in seed_margins],
    }

expect = {15: 2.3, 50: 0.8, 490: -3.2}
stage_c_ok = all(abs(margins[Q]["margin_pts_mean"] - expect[Q]) < 0.05 for Q in expect)
rep["stage_C"] = {
    "recomputed_from": "runs/stage_C/*/cluster_eval.json (exact rational arithmetic)",
    "by_Q": margins,
    "report_values": expect,
    "MATCH": stage_c_ok,
}

# Q=490 headline success (report: 93.8%)
q490 = margins[490]["target_only_pct_mean"]
rep["stage_C"]["Q490_target_only_pct"] = q490
rep["stage_C"]["Q490_matches_report_93.8"] = abs(q490 - 93.8) < 0.06

# ----------------------------------------------------------------------------- checkpoints
def count(d, pat=None):
    p = f"{ROOT}/runs/{d}"
    return len([x for x in os.listdir(p) if pat is None or pat in x]) if os.path.isdir(p) else 0

ck = {
    "stage_B": count("stage_B"), "stage_E": count("stage_E"),
    "stage_F": count("stage_F"), "stage_G": count("stage_G"),
    "stage_D": count("stage_D"), "stage_C": count("stage_C"),
}
expect_ck = {"stage_B": 30, "stage_E": 10, "stage_F": 168, "stage_G": 48}
ck["MATCH"] = all(ck[k] == v for k, v in expect_ck.items())
# every reused run must carry its train.marker + a final.pt
missing = []
for st in ("stage_B", "stage_E", "stage_F", "stage_G"):
    for r in sorted(os.listdir(f"{ROOT}/runs/{st}")):
        d = f"{ROOT}/runs/{st}/{r}"
        if not (os.path.exists(f"{d}/train.marker") and os.path.exists(f"{d}/final.pt")):
            missing.append(f"{st}/{r}")
ck["runs_missing_train_marker_or_final_pt"] = missing
ck["all_reusable_ckpts_intact"] = not missing
rep["checkpoints"] = ck

# ------------------------------------------------------------------- stage-G mask manifest
gm = json.load(open(f"{ROOT}/results/demo_mask_manifest.json"))
rep["stage_G_masks"] = {
    "K": gm["K"], "demos_per_mask": gm["demos_per_mask"], "seed": gm["seed"],
    "per_demo_inclusion_ok": gm["per_demo_inclusion_ok"],
    "n_masks_listed": len(gm["masks"]),
    "MATCH": gm["K"] == 24 and gm["demos_per_mask"] == 68 and len(gm["masks"]) == 24,
}

# stage_B per-task success present? (decides whether P2 needs re-evaluation)
b_pt = all(os.path.exists(f"{ROOT}/runs/stage_B/{r}/cluster_eval.json")
           for r in os.listdir(f"{ROOT}/runs/stage_B"))
rep["stage_B_per_task_success_available"] = b_pt

rep["ALL_PASS"] = bool(rep["gate1"]["MATCH"] and rep["stage_C"]["MATCH"]
                       and ck["MATCH"] and ck["all_reusable_ckpts_intact"]
                       and rep["stage_G_masks"]["MATCH"])

os.makedirs(f"{P2}/results", exist_ok=True)
json.dump(rep, open(f"{P2}/results/p0_intake_audit.json", "w"), indent=1)

print("=" * 78)
print("(a) GATE-1  recomputed from raw runs/stage_D/*/outcomes.json")
for a, v in gate1.items():
    print(f"      {a:7s} rho={v['rho']:+.6f}  p1={v['p_onesided']:.4f}")
print(f"    best = {best_attr} {best_rho:+.6f}   report=+0.252   "
      f"delta={abs(best_rho-0.252):.2e}  -> {'MATCH' if rep['gate1']['MATCH'] else 'MISMATCH'}")
print(f"    raw-vs-archived mask outcome max|delta| = {mo_delta:.3e}")
print()
print("(b) STAGE-C recomputed from raw runs/stage_C/*/cluster_eval.json (exact rationals)")
for Q in (15, 50, 490):
    print(f"      Q={Q:3d}  target={margins[Q]['target_only_pct_mean']:6.2f}%  "
          f"cotrain={margins[Q]['cotrain_pct_mean']:6.2f}%  "
          f"margin={margins[Q]['margin_pts_mean']:+.4f} (exact {margins[Q]['margin_pts_mean_exact']})"
          f"   report={expect[Q]:+.1f}")
print(f"    Q=490 target-only = {q490:.2f}%  (report 93.8%)")
print(f"    -> {'MATCH' if stage_c_ok else 'MISMATCH'}")
print()
print("checkpoints:", {k: v for k, v in ck.items() if k.startswith('stage')},
      "intact:", ck["all_reusable_ckpts_intact"])
print("stage-G masks:", rep["stage_G_masks"])
print("stage_B per-task success saved:", b_pt, "(P2 needs no re-evaluation)" if b_pt else "")
print("=" * 78)
print("ALL_PASS:", rep["ALL_PASS"])
sys.exit(0 if rep["ALL_PASS"] else 2)
