"""P8a INSTRUMENT CHECK -- prove the init/order factorization is real before spending 48 retrains.

Three trainings on the SAME data (Stage-G mask G000's 68 demos), compared at the level of raw
parameter tensors:

  A. src/train.py            --seed 401                        (the Phase-1 trainer, stock flags)
  B. train_factorial.py      --init_seed 401 --order_seed 401  --no_deterministic
  C. train_factorial.py      --init_seed 401 --order_seed 401  (cuDNN deterministic, as P8a runs)
  C'. a REPEAT of C

ASSERTIONS
  1. B == A  bit-for-bit  ->  splitting the seed into (init, order) reproduces the Phase-1 RNG
     stream exactly when both are set to the same value. This is what licenses calling INIT and
     ORDER separable factors: the factorial trainer IS the Phase-1 trainer, re-parameterized.
     A failure here means the decomposition would be measuring an artifact of my rewrite.
  2. C' == C bit-for-bit  ->  the trainer P8a actually runs is reproducible, so any difference
     between two P8a cells is caused by the seeds, not by nondeterminism.

Also reported (informational, NOT an assertion): whether A reproduces the ARCHIVED Phase-1 run
runs/stage_G/G000_s401/final.pt. If it does not, Phase-1 training was not bit-reproducible across
time on this host -- which would be worth knowing, but does not invalidate P8a, because P8a's
contrast is between runs made HERE, under identical flags.
"""
import json
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3lib as L
from p3lib import P3_RESULTS, P3_RUNS, RESULTS, RUNS

PY = "/home/ljc/miniconda/envs/robotda_x/bin/python"
SRC = os.path.join(L.ROOT, "src")
P3SRC = os.path.dirname(os.path.abspath(__file__))
BC = os.path.join(P3_RUNS, "_bitcheck")


def weights(path):
    return torch.load(path, map_location="cpu", weights_only=False)["model"]


def identical(p1, p2):
    a, b = weights(p1), weights(p2)
    if set(a) != set(b):
        return False, {"key_mismatch": True}
    stats = {"n_tensors": len(a), "n_differing": 0, "max_abs_diff": 0.0}
    for k in a:
        d = (a[k].float() - b[k].float()).abs().max().item()
        if d > 0:
            stats["n_differing"] += 1
            stats["max_abs_diff"] = max(stats["max_abs_diff"], d)
    return stats["n_differing"] == 0, stats


def launch(cmd, gpu, log):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    with open(log, "w") as f:
        return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=P3SRC, env=env)


def main():
    idle = L.gpu_idle()
    print(f"[bitcheck] idle GPUs: {idle}")
    if len(idle) < 3:
        raise SystemExit("need >= 3 idle GPUs (hard rule: 4-7 only)")

    gman = json.load(open(os.path.join(RESULTS, "demo_mask_manifest.json")))["masks"]
    demos = [m for m in gman if m["mask_id"] == "G000"][0]["demos"]
    os.makedirs(BC, exist_ok=True)
    dj = os.path.join(BC, "demos.json")
    json.dump({"demos": demos, "seed": 401}, open(dj, "w"), indent=1)

    arms = {
        "A_stock_train":      [PY, os.path.join(SRC, "train.py"), "--run_dir",
                               os.path.join(BC, "A_stock_train"), "--demos", dj, "--seed", "401"],
        "B_factorial_nodet":  [PY, os.path.join(P3SRC, "train_factorial.py"), "--run_dir",
                               os.path.join(BC, "B_factorial_nodet"), "--demos", dj,
                               "--init_seed", "401", "--order_seed", "401", "--no_deterministic"],
        "C_factorial_det":    [PY, os.path.join(P3SRC, "train_factorial.py"), "--run_dir",
                               os.path.join(BC, "C_factorial_det"), "--demos", dj,
                               "--init_seed", "401", "--order_seed", "401"],
    }
    procs = {}
    for (name, cmd), g in zip(arms.items(), idle[:3]):
        log = os.path.join(L.P3_LOGS, f"bitcheck_{name}.log")
        procs[name] = launch(cmd, g, log)
        print(f"[bitcheck] launch {name} on gpu{g}")
        time.sleep(3)
    for n, p in procs.items():
        rc = p.wait()
        print(f"[bitcheck] {n} rc={rc}")
        if rc != 0:
            raise SystemExit(f"{n} FAILED -- see phase3/logs/bitcheck_{n}.log")

    # repeat of C, on the first idle GPU, to test reproducibility of the P8a trainer itself
    g = L.gpu_idle()[0]
    cmd = [PY, os.path.join(P3SRC, "train_factorial.py"), "--run_dir",
           os.path.join(BC, "Cp_factorial_det_repeat"), "--demos", dj,
           "--init_seed", "401", "--order_seed", "401"]
    print(f"[bitcheck] launch C'_repeat on gpu{g}")
    rc = launch(cmd, g, os.path.join(L.P3_LOGS, "bitcheck_Cp.log")).wait()
    if rc != 0:
        raise SystemExit("C' FAILED")

    A = os.path.join(BC, "A_stock_train", "final.pt")
    B = os.path.join(BC, "B_factorial_nodet", "final.pt")
    C = os.path.join(BC, "C_factorial_det", "final.pt")
    Cp = os.path.join(BC, "Cp_factorial_det_repeat", "final.pt")
    ARCH = os.path.join(RUNS, "stage_G", "G000_s401", "final.pt")

    ok_BA, st_BA = identical(B, A)
    ok_CpC, st_CpC = identical(Cp, C)
    ok_CA, st_CA = identical(C, A)
    ok_AR, st_AR = identical(A, ARCH)

    out = {
        "stage": "P8a instrument check (init/order factorization)",
        "data": "Stage-G mask G000 (68 demos)", "seed": 401,
        "ASSERTION_1_factorial_equals_stock_when_init_eq_order": {
            "arms": "B (train_factorial init=order=401, no cuDNN flags) vs A (src/train.py seed=401)",
            "IDENTICAL": bool(ok_BA), "stats": st_BA,
            "meaning": ("If IDENTICAL, splitting one seed into (init, order) reproduces the "
                        "Phase-1 RNG stream EXACTLY. This is what licenses treating INIT and ORDER "
                        "as separable factors of the same trainer."),
        },
        "ASSERTION_2_p8a_trainer_is_reproducible": {
            "arms": "C' vs C (train_factorial, cuDNN deterministic, run twice)",
            "IDENTICAL": bool(ok_CpC), "stats": st_CpC,
            "meaning": ("If IDENTICAL, any difference between two P8a cells is caused by the "
                        "seeds, not by nondeterminism."),
        },
        "INFORMATIONAL_cudnn_flag_effect": {
            "arms": "C (deterministic) vs A (stock flags)", "IDENTICAL": bool(ok_CA),
            "stats": st_CA,
            "meaning": ("Whether enabling cuDNN determinism changes the trained weights. Either "
                        "way P8a is unbiased: ALL 48 P8a runs use the same flags."),
        },
        "INFORMATIONAL_phase1_reproducible_across_time": {
            "arms": "A (fresh src/train.py seed=401) vs the ARCHIVED runs/stage_G/G000_s401",
            "IDENTICAL": bool(ok_AR), "stats": st_AR,
            "meaning": ("Whether a Phase-1 training run reproduces bit-for-bit today. NOT an "
                        "assertion -- P8a compares runs made here under identical conditions."),
        },
        "GATE": bool(ok_BA and ok_CpC),
    }
    L.atomic_write_json(os.path.join(P3_RESULTS, "p8_bitcheck.json"), out)

    print("\n" + "=" * 88)
    print("P8a INSTRUMENT CHECK")
    print("=" * 88)
    print(f"  1. factorial(init=401,order=401) == src/train.py(401)   : {ok_BA}  {st_BA}")
    print(f"  2. P8a trainer reproducible (C' == C)                   : {ok_CpC}  {st_CpC}")
    print(f"  -  cuDNN-deterministic changes weights (C vs A)         : {not ok_CA}  {st_CA}")
    print(f"  -  Phase-1 run reproduces across time (A vs archive)    : {ok_AR}  {st_AR}")
    print(f"\n  GATE (assertions 1 AND 2): {out['GATE']}")
    print("=" * 88)
    if not out["GATE"]:
        print("INSTRUMENT DEFECT -- do NOT run P8a. Write PHASE3_DEFECT.md.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
