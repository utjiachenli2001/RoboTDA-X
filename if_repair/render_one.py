"""Render ONE demo and merge it into the gallery manifest. Designed to be killed from outside.

WHY THIS EXISTS -- a correction to `p17_render.py`'s timeout, which did not work.

That module guarded each demo with `signal.alarm()`. Python signal handlers only run between
bytecode instructions, so an alarm cannot interrupt a single long-running C call. MuJoCo's
`env.step()` on a pathological contact state is exactly that: one C call that never returns. The
guard sat pending for **14.5 hours at 100% CPU** on
`KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet/demo_1` before the process was killed by hand.

The only reliable timeout for a C-bound stall is an OS-level one. So each demo now renders in its
own short-lived process, and the caller wraps it in `timeout(1)`, which sends SIGTERM/SIGKILL and
does not care what the process is doing. A stall costs the timeout and nothing more.

The manifest is read-modify-written here, so a killed process simply leaves the manifest as it was.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair.p17_render import (WEB, RESULTS, render_demo, already_rendered,  # noqa: E402
                                  keyframe_names, slug_of, n_steps_of, write_manifest)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402

D.add_repo_paths()
import dataset  # noqa: E402


def load_manifest(path, agreement):
    if os.path.exists(path):
        try:
            m = json.load(open(path))
            return m.get("demos", []), m.get("failures", [])
        except Exception:
            pass
    return [], []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", required=True)
    ap.add_argument("--band", required=True, choices=["high", "low"])
    ap.add_argument("--out", default=WEB)
    a = ap.parse_args()

    scores = json.load(open(os.path.join(RESULTS, "p17_demo_scores.json")))
    row = next(r for r in scores["demos"] if r["demo_id"] == a.demo)
    meta = {k: row[k] for k in ("cluster", "score_O", "score_R", "z_O", "z_R", "z_mean",
                                "rank_O", "rank_R", "rank_gap")}
    man_path = os.path.join(a.out, "gallery.json")
    demos, failures = load_manifest(man_path, scores["agreement"])
    demos = [d for d in demos if d["demo_id"] != a.demo]
    failures = [f for f in failures if f["demo_id"] != a.demo]

    if already_rendered(a.demo, a.out):
        s, t, d = dataset.parse_did(a.demo)
        vid = f"{slug_of(a.demo)}.mp4"
        demos.append({**meta, "demo_id": a.demo, "suite": s, "task": t, "demo": d,
                      "n_steps": n_steps_of(a.demo), "n_frames": None,
                      "used_recorded_init_state": True, "band": a.band,
                      "keyframes": keyframe_names(a.demo),
                      "video": vid if os.path.exists(os.path.join(a.out, vid)) else None})
        write_manifest(man_path, scores["agreement"], demos, failures)
        print(f"SKIP {a.demo}")
        return 0

    got = render_demo(a.demo, a.out, timeout=10 ** 6)   # no in-process timeout; the caller owns it
    if "error" in got:
        failures.append({"demo_id": a.demo, "band": a.band, "reason": got["error"]})
        write_manifest(man_path, scores["agreement"], demos, failures)
        print(f"FAILED {a.demo}: {got['error']}")
        return 1
    got.update(meta)
    got["band"] = a.band
    demos.append(got)
    write_manifest(man_path, scores["agreement"], demos, failures)
    print(f"OK {a.demo} — {got['n_frames']} frames in {got['render_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
