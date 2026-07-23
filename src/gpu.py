"""GPU gating (HARD RULE: only indices 4,5,6,7; never touch 0-3).

A GPU is idle iff memory.used < 1000 MiB AND utilization.gpu < 10%.
Never queue onto a busy GPU; if none are idle, sleep 300 s and re-check.
"""
import subprocess
import time

ALLOWED = (4, 5, 6, 7)
MEM_LIMIT_MIB = 1000
UTIL_LIMIT = 10
SLEEP_S = 300


def query(gpus=ALLOWED):
    """-> {index: (mem_used_MiB, util_pct)} for the allowed GPUs only."""
    ids = ",".join(str(g) for g in gpus)
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits", "-i", ids],
        capture_output=True, text=True, check=True).stdout.strip()
    st = {}
    for line in out.splitlines():
        i, m, u = [x.strip() for x in line.split(",")]
        st[int(i)] = (int(m), int(u))
    return st


def is_idle(gpu, st=None):
    if gpu not in ALLOWED:
        raise ValueError(f"GPU {gpu} is OUT OF BOUNDS (allowed: {ALLOWED})")
    st = st or query()
    m, u = st[gpu]
    return m < MEM_LIMIT_MIB and u < UTIL_LIMIT


def idle_gpus():
    st = query()
    return [g for g in ALLOWED if is_idle(g, st)]


def wait_for_idle(n=1, verbose=True):
    """Block (sleeping 300 s between checks) until >= n allowed GPUs are idle."""
    while True:
        free = idle_gpus()
        if len(free) >= n:
            return free
        if verbose:
            st = query()
            print(f"[gpu] no idle GPU (need {n}); state={st}; sleeping {SLEEP_S}s", flush=True)
        time.sleep(SLEEP_S)


if __name__ == "__main__":
    st = query()
    for g in ALLOWED:
        m, u = st[g]
        print(f"GPU {g}: {m} MiB, {u}%  -> {'IDLE' if is_idle(g, st) else 'BUSY'}")
