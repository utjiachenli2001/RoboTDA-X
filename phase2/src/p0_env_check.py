"""P0 environment verification: env reset + OPEN-LOOP action replay of 1 demo (must succeed),
then a 50-step tiny training run through Phase-1's OWN trainer (src/train.py).
"""
import json
import os
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, "/mnt/sdb/ljc/RoboTDA-X/src")
import bootstrap  # noqa: F401  (pins GPUs 4-7 + LIBERO paths BEFORE torch)
from bootstrap import DATA, ROOT  # noqa: E402
import libero_env as LE  # noqa: E402

out = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}

# ---------------------------------------------------------------- open-loop replay of 1 demo
SUITE, TASK, DEMO = "libero_goal", "open_the_middle_drawer_of_the_cabinet", "demo_0"
sdir = os.path.join(DATA, SUITE)
cands = [f for f in os.listdir(sdir) if f.startswith(TASK) and f.endswith(".hdf5")]
h5 = os.path.join(sdir, cands[0])

with h5py.File(h5, "r") as f:
    actions = f["data"][DEMO]["actions"][:].astype(np.float32)
    init_state = f["data"][DEMO]["states"][0]

t0 = time.time()
env = LE.make_env(LE.get_bddl_path(SUITE, TASK), horizon=1000, seed=0)
env.reset()
env.env.sim.set_state_from_flattened(init_state)   # exact demo start state
env.env.sim.forward()

success, step_success = False, None
for t in range(actions.shape[0]):
    env.step(actions[t])                            # OPEN-LOOP: recorded actions, no policy
    if LE.check_success(env):
        success, step_success = True, t + 1
        break
wall = time.time() - t0
try:
    env.close()
except Exception:
    pass

out["replay"] = {"suite": SUITE, "task": TASK, "demo": DEMO,
                 "n_actions": int(actions.shape[0]), "success": bool(success),
                 "success_at_step": step_success, "wall_s": round(wall, 2)}
print(f"[replay] {TASK}/{DEMO}: {actions.shape[0]} recorded actions replayed open-loop -> "
      f"success={success} (step {step_success}) in {wall:.1f}s", flush=True)

# ------------------------------------------------- 50-step tiny training run via src/train.py
import torch  # noqa: E402
import train as T  # noqa: E402
import dataset  # noqa: E402

cfg = T.load_cfg()
cfg = dict(cfg); cfg["total_steps"] = 50; cfg["n_ckpt"] = 1
demos = json.load(open(os.path.join(ROOT, "runs/stage_G/G000_s401/demos.json")))["demos"]

rd = os.path.join(ROOT, "phase2/runs/_envcheck_tiny")
os.makedirs(rd, exist_ok=True)
t0 = time.time()
T.train(rd, demos, seed=999, cfg=cfg, device="cuda")
wall2 = time.time() - t0

log = [json.loads(l) for l in open(os.path.join(rd, "train_log.jsonl"))]
losses = [r["loss"] for r in log]
out["tiny_train"] = {
    "device": "cuda", "torch": torch.__version__,
    "state_dim": int(dataset.state_dim()),
    "steps": 50, "n_log_recs": len(log),
    "loss_first": losses[0], "loss_last": losses[-1],
    "all_finite": bool(np.all(np.isfinite(losses))),
    "wall_s": round(wall2, 2),
}
print(f"[train] 50 steps: loss {losses[0]:.4f} -> {losses[-1]:.4f} in {wall2:.1f}s "
      f"finite={out['tiny_train']['all_finite']}", flush=True)

out["PASS"] = bool(success and out["tiny_train"]["all_finite"])
json.dump(out, open(os.path.join(ROOT, "phase2/results/p0_env_check.json"), "w"), indent=1)
print("PASS:", out["PASS"])
sys.exit(0 if out["PASS"] else 2)
