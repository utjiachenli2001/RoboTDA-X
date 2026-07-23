"""Import-first bootstrap: pins LIBERO paths to THIS env and sets headless MuJoCo.

Every RoboTDA-X module imports this before importing libero/robosuite.
Rationale: LIBERO resolves its bddl/init_files/assets through a *global* ~/.libero/config.yaml,
which on this host points at a different conda env (dcvla). We do not modify the global file
(other projects depend on it); we point LIBERO at a project-local config instead.
"""
import os

ROOT = "/mnt/sdb/ljc/RoboTDA-X"

os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.join(ROOT, "configs/libero_cfg"))
os.environ.setdefault("MUJOCO_GL", "osmesa")   # no GPU rendering; we never render
# NB: do NOT set MUJOCO_EGL_DEVICE_ID -- robosuite asserts it is a digit inside
# CUDA_VISIBLE_DEVICES whenever CUDA_VISIBLE_DEVICES is non-empty (binding_utils.py:29).
os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
# keep per-process CPU usage to 1 thread: we parallelize with processes, not threads
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

DATA = os.path.join(ROOT, "data/libero")
PROC = os.path.join(ROOT, "data/proc")
RESULTS = os.path.join(ROOT, "results")
RUNS = os.path.join(ROOT, "runs")
LOGS = os.path.join(ROOT, "logs")
FIGURES = os.path.join(ROOT, "figures")
CONFIGS = os.path.join(ROOT, "configs")

# ---------------------------------------------------------------------------------------
# HARD RULE: only GPUs 4,5,6,7 may ever be used. 0-3 belong to another user.
#
# Orchestrator-launched jobs get CUDA_VISIBLE_DEVICES from the orchestrator. But scripts run
# BARE (attribution.py, stage_d.py, stage_c.py's intrusion sweep, gate0_diag.py) inherit no
# CUDA_VISIBLE_DEVICES, and torch then defaults to cuda:0 -- i.e. GPU 0, someone else's job.
# That actually happened once (attribution.py died with OOM on GPU 0; it allocated nothing, but
# it should never have looked). So: if CUDA_VISIBLE_DEVICES is not already set, pin it here to
# an allowed GPU BEFORE torch is ever imported. This makes the rule structural rather than a
# thing every caller has to remember.
# ---------------------------------------------------------------------------------------
ALLOWED_GPUS = (4, 5, 6, 7)


def _gpu_state():
    import subprocess
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits", "-i", ",".join(map(str, ALLOWED_GPUS))],
        capture_output=True, text=True, check=True).stdout.strip()
    st = {}
    for line in out.splitlines():
        i, m, u = [x.strip() for x in line.split(",")]
        st[int(i)] = (int(m), int(u))
    return st


def _pin_allowed_gpu():
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        return                      # a caller (e.g. the orchestrator) already chose; respect it
    try:
        st = _gpu_state()
    except Exception:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(ALLOWED_GPUS[0])
        return
    idle = [g for g in ALLOWED_GPUS if st[g][0] < 1000 and st[g][1] < 10]
    pick = idle[0] if idle else min(ALLOWED_GPUS, key=lambda g: st[g][0])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(pick)
    print(f"[bootstrap] CUDA_VISIBLE_DEVICES unset -> pinned to GPU {pick} "
          f"(allowed {ALLOWED_GPUS}; idle={idle})", flush=True)


_pin_allowed_gpu()

for _d in (RESULTS, RUNS, LOGS, FIGURES):
    os.makedirs(_d, exist_ok=True)


def done_marker(run_dir, name="done"):
    return os.path.join(run_dir, f"{name}.marker")


def is_done(run_dir, name="done"):
    return os.path.exists(done_marker(run_dir, name))


def write_done(run_dir, payload="done\n", name="done"):
    """Atomic completion marker: write tmp then rename (survives a kill mid-write)."""
    os.makedirs(run_dir, exist_ok=True)
    tmp = os.path.join(run_dir, f".{name}.tmp")
    with open(tmp, "w") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, done_marker(run_dir, name))
