"""Phase-4 budget + status. Reads phase4/logs/*_summary.json (written by orch4) and the
attribution logs, and emits phase4/results/budget_actual.json + phase4/STATUS.md.

Every number here is read from a run artifact. The per-stage budget guard (>1.5x a ledger line
=> BUDGET_ALERT.md, pause, decide, document) is evaluated here.
"""
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p4lib as L
from p4lib import P4_RESULTS, P4_LOGS, P4

# PREREGISTERED ledger (preregistration_phase4.json: BUDGET_LEDGER)
LEDGER = {
    "P12": {"retrains": 96, "gpu_h": 13, "episodes": 25920, "what": "BC ground truth S=10"},
    "P11": {"retrains": 0, "gpu_h": 1, "episodes": 0, "what": "champion test (attribution only)"},
    "P13": {"retrains": 48, "gpu_h": 17, "episodes": 12960, "what": "diffusion S=8 + verdict"},
    "P14": {"retrains": 64, "gpu_h": 9, "episodes": 3840, "what": "fresh heterogeneity corpus"},
}
ALERT_GPU_H = 60          # PREREGISTERED global alert threshold
EP_PER_RUN = {"P12": 270, "P13": 270, "P14": 60}     # 27 tasks x10 ; 27 x10 ; 6 x10


def main():
    rows, alerts = [], []
    tot_r = tot_g = tot_e = tot_ok = tot_fail = 0
    for stage in ("P12", "P13", "P14"):
        p = os.path.join(P4_LOGS, f"{stage}_summary.json")
        if not os.path.exists(p):
            rows.append({"stage": stage, **LEDGER[stage], "status": "did not run",
                         "retrains_actual": 0, "gpu_h_actual": 0.0, "n_ok": 0, "n_fail": 0,
                         "episodes_actual": 0})
            continue
        s = json.load(open(p))
        n_run = s["n_run"]
        n_done = s["n_ok"] + s["n_skipped"]
        ep = n_done * EP_PER_RUN[stage]
        over = s["gpu_h"] > 1.5 * LEDGER[stage]["gpu_h"]
        if over:
            alerts.append(f"{stage}: {s['gpu_h']:.2f} GPU-h vs ledger {LEDGER[stage]['gpu_h']} "
                          f"(>1.5x)")
        rows.append({"stage": stage, **LEDGER[stage], "status": "complete" if s["n_fail"] == 0
                     and n_done == s["n_jobs"] else "incomplete/failed",
                     "retrains_actual": n_done, "gpu_h_actual": s["gpu_h"],
                     "n_ok": s["n_ok"], "n_fail": s["n_fail"], "n_skipped": s["n_skipped"],
                     "episodes_actual": ep, "wall_h": s["wall_s"] / 3600.0,
                     "over_1.5x_ledger": over})
        tot_r += n_done
        tot_g += s["gpu_h"]
        tot_e += ep
        tot_ok += s["n_ok"]
        tot_fail += s["n_fail"]

    # attribution GPU-h (P11 Gram pass) -- recorded separately, 0 retrains
    attr_h = 0.0
    ap = os.path.join(P4_LOGS, "p11_gram.time")
    if os.path.exists(ap):
        attr_h = float(open(ap).read().strip()) / 3600.0
    tot_g += attr_h

    out = {
        "phase": 4, "preregistration_sha256": L.PREREG_SHA,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stages": rows,
        "P11_attribution_gpu_h": attr_h,
        "TOTAL_retrains": tot_r, "TOTAL_gpu_h": tot_g, "TOTAL_episodes": tot_e,
        "TOTAL_ok": tot_ok, "TOTAL_failed": tot_fail,
        "LEDGER_total_retrains": sum(v["retrains"] for v in LEDGER.values()),
        "LEDGER_total_gpu_h": sum(v["gpu_h"] for v in LEDGER.values()),
        "LEDGER_total_episodes": sum(v["episodes"] for v in LEDGER.values()),
        "ALERT_THRESHOLD_gpu_h": ALERT_GPU_H,
        "ALERT_TRIPPED": bool(tot_g > ALERT_GPU_H or alerts),
        "per_stage_alerts": alerts,
        "cuts": "NONE" if all(r["status"] != "did not run" for r in rows) else
                [r["stage"] for r in rows if r["status"] == "did not run"],
    }
    L.atomic_write_json(os.path.join(P4_RESULTS, "budget_actual.json"), out)

    lines = ["# RoboTDA-X — PHASE 4 STATUS", "",
             f"Preregistration `phase4/preregistration_phase4.json`, "
             f"SHA-256 `{L.PREREG_SHA}` — locked before any Phase-4 training, attribution, or "
             f"verdict.", "",
             "| stage | what | retrains (ledger) | retrains (actual) | ok/fail | GPU-h (ledger) "
             "| GPU-h (actual) | episodes | status |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['stage']} | {r['what']} | {r['retrains']} | {r['retrains_actual']} | "
            f"{r['n_ok']}/{r['n_fail']} | {r['gpu_h']} | {r['gpu_h_actual']:.2f} | "
            f"{r['episodes_actual']} | {r['status']} |")
    lines += [
        f"| P11 | attribution (Gram pass, 0 retrains) | 0 | 0 | — | 1 | {attr_h:.2f} | 0 | "
        f"{'complete' if attr_h else 'did not run'} |",
        f"| **TOTAL** | | **{out['LEDGER_total_retrains']}** | **{tot_r}** | "
        f"**{tot_ok}/{tot_fail}** | **{out['LEDGER_total_gpu_h']}** | **{tot_g:.2f}** | "
        f"**{tot_e}** | |", "",
        f"**GPU-h alert threshold: {ALERT_GPU_H}. "
        f"{'TRIPPED — see BUDGET_ALERT.md' if out['ALERT_TRIPPED'] else 'Not tripped.'}**", "",
        f"Cuts: {out['cuts']}", ""]
    open(os.path.join(P4, "STATUS.md"), "w").write("\n".join(lines) + "\n")

    print("\n".join(lines))
    if out["ALERT_TRIPPED"]:
        print("\n*** BUDGET ALERT TRIPPED -- write phase4/BUDGET_ALERT.md, PAUSE, decide ***")


if __name__ == "__main__":
    main()
