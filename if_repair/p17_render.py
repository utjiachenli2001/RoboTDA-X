"""PASS 17 -- render LIBERO keyframes and clips for the demo gallery. CPU (OSMesa), no GPU.

WHY RENDERING IS NEEDED AT ALL. The processed corpus is state-only -- `data/proc/**/demo_N.npz`
holds `proprio (T,16)`, `object (T,28)` and `actions (T,7)` and no pixels. So frames have to be
re-created in simulation: reset the task, set the demo's recorded initial state, replay its recorded
actions, and capture the camera.

FIDELITY. The initial state comes from LIBERO's per-task `.pruned_init` at the demo's own index and
the actions are the demo's own, so this reproduces the demonstration rather than illustrating the
task. What it is not is the ORIGINAL rendering: these frames are made now, at this resolution and
camera.

WHY OSMesa AND NOT THE GPU. `libEGL_nvidia` is absent on this box and the EGL driver lacks the
PLATFORM_DEVICE extension, so hardware headless rendering cannot initialise; installing the NVIDIA
EGL ICD needs root. OSMesa software rendering runs ~0.45 s/frame, irrelevant for a fixed gallery.

THREE ROBUSTNESS PROPERTIES, each added after the failure it prevents.

1. **Per-demo wall-clock timeout.** Replaying recorded actions can drive MuJoCo into a bad contact
   state where the solver crawls. Observed once: `push_the_plate_to_the_front_of_the_stove/demo_0`,
   155 steps -- FEWER than a neighbour that finished in two minutes -- ran 39 minutes at ~15 s/step
   before being killed. A timeout turns that from a stalled pipeline into one recorded failure.
2. **The manifest is written after EVERY demo.** The first version wrote it once at the end, so
   killing the stalled run lost the bookkeeping for ten finished demos even though their frames were
   safely on disk.
3. **Skipping is decided by FILES ON DISK, not by the manifest.** That makes the two independent:
   work already done is recognised even if the manifest was lost, which is exactly the case (2)
   created.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time

import numpy as np

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402

D.add_repo_paths()
import dataset  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
WEB = os.path.join(os.path.dirname(HERE), "docs", "demo_gallery")
N_KEYFRAMES = 6
VIDEO_STRIDE = 3
FPS = 20
CAM_H = CAM_W = 256
TIMEOUT_S = 420


class Timeout(Exception):
    pass


def _alarm(_s, _f):
    raise Timeout()


def _libero_paths():
    import libero
    root = os.path.join(os.path.dirname(libero.__file__), "libero")
    return os.path.join(root, "bddl_files"), os.path.join(root, "init_files")


def slug_of(demo_id):
    return demo_id.replace("/", "__")


def keyframe_names(demo_id, n=N_KEYFRAMES):
    s = slug_of(demo_id)
    return [f"{s}__kf{j}.jpg" for j in range(n)]


def already_rendered(demo_id, out_dir):
    return all(os.path.exists(os.path.join(out_dir, f)) for f in keyframe_names(demo_id))


def n_steps_of(demo_id):
    s, t, d = dataset.parse_did(demo_id)
    return int(np.load(os.path.join(dataset.PROC, s, t, f"{d}.npz"))["actions"].shape[0])


def render_demo(demo_id, out_dir, stride=VIDEO_STRIDE, timeout=TIMEOUT_S):
    import imageio.v2 as iio
    from libero.libero.envs import OffScreenRenderEnv

    suite, task, demo = dataset.parse_did(demo_id)
    bddl_dir, init_dir = _libero_paths()
    bddl = os.path.join(bddl_dir, suite, f"{task}.bddl")
    init = os.path.join(init_dir, suite, f"{task}.pruned_init")
    if not os.path.exists(bddl):
        return {"error": f"no bddl for {suite}/{task}"}

    actions = np.load(os.path.join(dataset.PROC, suite, task, f"{demo}.npz"))["actions"]
    idx = int(demo.split("_")[-1])
    t0 = time.time()
    env = frames = None
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout)
    try:
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=CAM_H, camera_widths=CAM_W)
        env.reset()
        used_init = False
        if os.path.exists(init):
            try:
                import torch
                st = torch.load(init, weights_only=False)
                if idx < len(st):
                    env.set_init_state(st[idx])
                    used_init = True
            except Exception:
                pass
        frames = []
        for t, a in enumerate(actions):
            obs, _, done, _ = env.step(a)
            if t % stride == 0:
                frames.append(obs["agentview_image"][::-1].copy())
            if done:
                break
    except Timeout:
        return {"error": f"TIMEOUT after {timeout}s at ~{len(frames or [])} frames "
                         f"(MuJoCo solver stall on replayed actions)"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        signal.alarm(0)
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    if not frames:
        return {"error": "no frames captured"}

    os.makedirs(out_dir, exist_ok=True)
    picks = np.linspace(0, len(frames) - 1, N_KEYFRAMES).astype(int)
    kf = keyframe_names(demo_id)
    for fn, p in zip(kf, picks):
        iio.imwrite(os.path.join(out_dir, fn), frames[p], quality=88)
    vid = f"{slug_of(demo_id)}.mp4"
    try:
        iio.mimwrite(os.path.join(out_dir, vid), frames, fps=FPS, quality=7)
    except Exception:
        vid = None
    return {"demo_id": demo_id, "suite": suite, "task": task, "demo": demo,
            "n_steps": int(len(actions)), "n_frames": len(frames),
            "used_recorded_init_state": used_init, "render_s": round(time.time() - t0, 1),
            "keyframes": kf, "video": vid}


def write_manifest(path, agreement, demos, failures):
    json.dump({"agreement": agreement,
               "render": {"renderer": "OSMesa software (libEGL_nvidia absent; EGL lacks "
                                      "PLATFORM_DEVICE)",
                          "camera": "agentview", "resolution": f"{CAM_W}x{CAM_H}", "fps": FPS,
                          "video_stride": VIDEO_STRIDE, "timeout_s": TIMEOUT_S},
               "failures": failures, "demos": demos},
              open(path, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--bottom", type=int, default=12)
    ap.add_argument("--out", default=WEB)
    ap.add_argument("--timeout", type=int, default=TIMEOUT_S)
    a = ap.parse_args()

    scores = json.load(open(os.path.join(RESULTS, "p17_demo_scores.json")))
    rows = sorted(scores["demos"], key=lambda r: -r["z_mean"])
    sel = rows[:a.top] + rows[-a.bottom:]
    tag = {r["demo_id"]: ("high" if i < a.top else "low") for i, r in enumerate(sel)}
    man_path = os.path.join(a.out, "gallery.json")
    os.makedirs(a.out, exist_ok=True)

    demos, failures = [], []
    for i, r in enumerate(sel, 1):
        did = r["demo_id"]
        meta = {k: r[k] for k in ("cluster", "score_O", "score_R", "z_O", "z_R", "z_mean",
                                  "rank_O", "rank_R", "rank_gap")}
        if already_rendered(did, a.out):
            s, t, d = dataset.parse_did(did)
            vid = f"{slug_of(did)}.mp4"
            demos.append({**meta, "demo_id": did, "suite": s, "task": t, "demo": d,
                          "n_steps": n_steps_of(did), "n_frames": None,
                          "used_recorded_init_state": True, "band": tag[did],
                          "keyframes": keyframe_names(did),
                          "video": vid if os.path.exists(os.path.join(a.out, vid)) else None})
            print(f"[{i}/{len(sel)}] SKIP (on disk)  {did}", flush=True)
            write_manifest(man_path, scores["agreement"], demos, failures)
            continue

        print(f"[{i}/{len(sel)}] {tag[did]:4s} z={r['z_mean']:+.2f}  {did}", flush=True)
        got = render_demo(did, a.out, timeout=a.timeout)
        if "error" in got:
            print(f"    FAILED: {got['error']}", flush=True)
            failures.append({"demo_id": did, "band": tag[did], "reason": got["error"]})
        else:
            got.update(meta)
            got["band"] = tag[did]
            demos.append(got)
            print(f"    {got['n_frames']} frames in {got['render_s']}s, "
                  f"init={got['used_recorded_init_state']}", flush=True)
        write_manifest(man_path, scores["agreement"], demos, failures)

    print(f"\n[p17/render] {len(demos)} rendered, {len(failures)} failed -> {a.out}")
    for f in failures:
        print(f"   FAILED {f['demo_id']}: {f['reason']}")


if __name__ == "__main__":
    main()
