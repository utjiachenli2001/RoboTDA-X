"""Append a stage entry to STATUS.md (spec §0: after EVERY stage).

Reads only artifact files: logs/<stage>_summary.json (written by the orchestrator) plus any
gate verdict json. Never invents numbers; a stage that did not run says so.
"""
import os
import sys
import json
import argparse
import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bootstrap  # noqa: F401
from bootstrap import ROOT, LOGS, RESULTS

STATUS = os.path.join(ROOT, "STATUS.md")


def stage_summary(stage):
    p = os.path.join(LOGS, f"{stage}_summary.json")
    return json.load(open(p)) if os.path.exists(p) else None


def append(stage, verdict=None, notes=None, episodes=None):
    s = stage_summary(stage)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"\n## {stage}  ({ts})\n"]
    if s is None:
        lines.append("- **did not run** (no orchestrator summary found)\n")
    else:
        gpu_h = s["gpu_h_est"]
        lines.append(
            f"- runs: {s['n_jobs']} total, {s['n_run']} executed this pass, "
            f"{s['n_skipped']} already complete (resumed), "
            f"**{s['n_ok']} ok / {s['n_fail']} failed**\n"
            f"- wall-clock: {s['wall_s']/60:.1f} min; GPU-h (sum of job wall on 1 GPU each): "
            f"{gpu_h:.2f}\n")
        if s["n_fail"]:
            bad = [os.path.basename(r["run_dir"]) for r in s["results"] if not r["ok"]]
            lines.append(f"- FAILURES: {bad}\n")
    if episodes is not None:
        lines.append(f"- rollout episodes: {episodes:,}\n")
    if verdict:
        lines.append(f"- **GATE VERDICT: {verdict}**\n")
    if notes:
        for n in (notes if isinstance(notes, list) else [notes]):
            lines.append(f"- {n}\n")
    with open(STATUS, "a") as f:
        f.writelines(lines)
    print(f"[status] appended {stage} to {STATUS}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--verdict", default=None)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--note", action="append", default=None)
    a = ap.parse_args()
    append(a.stage, a.verdict, a.note, a.episodes)


if __name__ == "__main__":
    main()
