"""Phase-3 shared library: the AUDIT-HARDENING fixes (P6) + common paths.

This module is the ONLY ingestion path any Phase-3 reader may use. It exists because the
external audit found three latent defects in the Phase-1/2 pipeline. Phase-1/2 source is
READ-ONLY, so the fixes live here as hardened re-implementations.

DEFECT 1 -- MARKER-GATED INGESTION (P6.1)
    src/rollout.py writes outcomes.json BEFORE it raises on rollout errors and BEFORE it
    writes probe.marker:

        json.dump({"outcomes": out, ...}, open(run_dir/"outcomes.json", "w"))   # <-- written
        if info["n_errors"]:
            ...
            raise RuntimeError(...)                                             # <-- then dies
        write_done(run_dir, ..., name="probe")                                  # <-- never reached

    So a run whose rollouts partially errored leaves a COMPLETE-LOOKING outcomes.json with no
    marker. Every Phase-1/2 analysis reader ingests on file existence only (e.g.
    phase2/src/p1_analyze.py:collect_new does `if not os.path.exists(p): continue`), so such a
    file would be silently ingested as if it were a clean run.

    It never fired -- the audit found zero artifact-without-marker dirs and p6_audit.py
    re-confirms that on this host. But it is a live trap for every future run.

    FIX (here): (a) writes are atomic (tmp + os.replace), so a killed process can never leave a
    torn artifact; (b) readers REQUIRE the completion marker. read_outcomes() refuses to ingest
    an unmarked run.

DEFECT 2 -- Q490 PROBE LEAK (P6.2)
    Stage-C's Q=490 training pool contains libero_goal demo_45..demo_49 -- which are exactly the
    Phase-2 per-task probe demos. Harmless today (no Q490 model is ever scored against them),
    but nothing PREVENTED it. assert_no_probe_leak() makes it structurally impossible: it
    refuses to compute attribution for any model whose demos.json intersects the probe set.

DEFECT 3 -- HARDCODED EPISODE COUNT (P6.3)
    src/analysis.py:32 hardcodes N_EPISODES=30 for the logit clamp, then applies it to every
    row. logit_success_rowwise() reads the row's own n_episodes instead. All existing rows have
    n_episodes == 30, so this changes nothing today (proved in p6_no_change.json) -- but the
    clamp would have been silently wrong the moment any stage used a different rollout count.
"""
import functools
import hashlib
import json
import os
import sys

import numpy as np

ROOT = "/mnt/sdb/ljc/RoboTDA-X"
sys.path.insert(0, os.path.join(ROOT, "src"))

import bootstrap  # noqa: F401,E402  (structural GPU pinning -- never regress this)
from bootstrap import RESULTS, RUNS  # noqa: E402

P3 = os.path.join(ROOT, "phase3")
P3_RESULTS = os.path.join(P3, "results")
P3_RUNS = os.path.join(P3, "runs")
P3_FIGURES = os.path.join(P3, "figures")
P3_LOGS = os.path.join(P3, "logs")
P2_RESULTS = os.path.join(ROOT, "phase2", "results")
P2_RUNS = os.path.join(ROOT, "phase2", "runs")

for _d in (P3_RESULTS, P3_RUNS, P3_FIGURES, P3_LOGS):
    os.makedirs(_d, exist_ok=True)

# Which marker gates which artifact (Phase-1 marker names, retained -- see PHASE2_REPORT §7.3).
ARTIFACT_MARKER = {
    "outcomes.json": "probe",
    "cluster_eval.json": "clustereval",
    "suite_outcomes.json": "suite",
}


# ---------------------------------------------------------------- P6.1: atomic writes
def atomic_write_json(path, obj, indent=1):
    """Write JSON atomically: tmp in the same dir, fsync, then os.replace.

    os.replace is atomic within a filesystem, so a reader either sees the OLD file or the
    COMPLETE new one -- never a torn one, even if the process is killed mid-write.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent, default=float)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_obj(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=float).encode()).hexdigest()


# ---------------------------------------------------------------- P6.1: marker-gated reading
def is_marked(run_dir, marker):
    return os.path.exists(os.path.join(run_dir, f"{marker}.marker"))


def read_artifact(run_dir, artifact="outcomes.json", required=True):
    """MARKER-GATED ingestion. Returns the parsed artifact, or None.

    An artifact present WITHOUT its completion marker is a partially-failed run (see the
    module docstring). It is REFUSED, loudly -- never silently ingested.
    """
    marker = ARTIFACT_MARKER[artifact]
    apath = os.path.join(run_dir, artifact)
    has_a, has_m = os.path.exists(apath), is_marked(run_dir, marker)
    if has_a and not has_m:
        raise RuntimeError(
            f"MARKER-GATE VIOLATION: {apath} exists but {marker}.marker does not. "
            f"This is a partially-failed run (rollout.py writes the artifact before it raises). "
            f"Refusing to ingest. Investigate the run before using it.")
    if not has_a:
        if required:
            raise FileNotFoundError(f"missing artifact: {apath}")
        return None
    return json.load(open(apath))


def read_outcomes(run_dir, required=True):
    """The probe battery's outcomes.json, marker-gated. -> {cluster: {...}} or None."""
    r = read_artifact(run_dir, "outcomes.json", required=required)
    return None if r is None else r["outcomes"]


def scan_marker_gate(run_dirs):
    """Sweep for artifact-without-marker. -> {'violations': [...], 'n_scanned': int, ...}"""
    viol, ok, empty = [], 0, []
    for d in run_dirs:
        if not os.path.isdir(d):
            continue
        found = False
        for artifact, marker in ARTIFACT_MARKER.items():
            apath = os.path.join(d, artifact)
            if os.path.exists(apath):
                found = True
                if not is_marked(d, marker):
                    viol.append({"run_dir": d, "artifact": artifact, "marker": marker,
                                 "artifact_bytes": os.path.getsize(apath)})
                else:
                    ok += 1
        if not found:
            empty.append(d)
    return {"n_run_dirs_scanned": len(run_dirs),
            "n_artifact_marker_pairs_ok": ok,
            "n_violations": len(viol),
            "violations": viol,
            "n_dirs_with_no_eval_artifact": len(empty),
            "dirs_with_no_eval_artifact": sorted(empty)}


# ---------------------------------------------------------------- P6.2: probe-leak guard
@functools.lru_cache(maxsize=1)
def phase2_probe_ids():
    """The 135 Phase-2 per-task probe demos (27 tasks x demo_45..demo_49), as a frozenset.

    NB the probe entries live under the "probes" key, NOT at the top level (the top level also
    holds "rule", "n_tasks", ... metadata). An earlier version of this function walked only the
    top level and returned an EMPTY set -- which made assert_no_probe_leak() pass VACUOUSLY.
    The post-condition below makes that failure mode impossible: an empty or wrong-sized probe
    set now raises instead of silently disarming the guard.
    """
    p = json.load(open(os.path.join(P2_RESULTS, "per_task_probes.json")))
    ids = set()

    def walk(o):
        if isinstance(o, dict):
            if "demo_ids" in o and isinstance(o["demo_ids"], list):
                ids.update(o["demo_ids"])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(p)
    n_tasks = int(p["n_tasks"])
    n_per = int(p["n_demos_per_task"])
    expect = n_tasks * n_per
    if len(ids) != expect:
        raise RuntimeError(
            f"phase2_probe_ids(): extracted {len(ids)} probe ids but per_task_probes.json "
            f"declares {n_tasks} tasks x {n_per} demos = {expect}. A probe set that is empty or "
            f"the wrong size would DISARM assert_no_probe_leak(), so this is fatal.")
    return frozenset(ids)


def run_demos(run_dir):
    """The demo ids a run was TRAINED on, from its demos.json (the authoritative record)."""
    p = os.path.join(run_dir, "demos.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"no demos.json in {run_dir}")
    return set(json.load(open(p))["demos"])


def assert_no_probe_leak(run_dirs, probe_ids, context=""):
    """HARD GUARD (P6.2). Refuse to compute attribution for any model whose training demos
    intersect the probe set being used as the test-side functional.

    Called by EVERY Phase-3 attribution entry point. Raises on any intersection.
    """
    probe_ids = frozenset(probe_ids)
    bad = []
    for d in run_dirs:
        inter = run_demos(d) & probe_ids
        if inter:
            bad.append({"run_dir": d, "n_leaked": len(inter),
                        "examples": sorted(inter)[:5]})
    if bad:
        raise RuntimeError(
            f"PROBE LEAK REFUSED{(' [' + context + ']') if context else ''}: "
            f"{len(bad)} model(s) were trained on demos that are in the probe set being "
            f"attributed against. Attribution on such a model is invalid. Offenders: {bad}")
    return {"n_models_checked": len(run_dirs), "n_probe_ids": len(probe_ids), "clean": True}


# ---------------------------------------------------------------- P6.3: episode-count fix
def logit_success_rowwise(p, n_episodes):
    """logit(clamp(p, 1/(2n), 1-1/(2n))) with n READ PER ROW (P6.3).

    src/analysis.py hardcodes n = 30. Here n comes from the row. Vectorized.
    """
    p = np.asarray(p, dtype=float)
    n = np.asarray(n_episodes, dtype=float)
    lo = 1.0 / (2.0 * n)
    p = np.clip(p, lo, 1.0 - lo)
    return np.log(p / (1.0 - p))


# ---------------------------------------------------------------- convenience
def all_phase12_run_dirs():
    """Every Phase-1 and Phase-2 run dir on this host."""
    out = []
    for base in (RUNS, P2_RUNS):
        for stage in sorted(os.listdir(base)):
            sp = os.path.join(base, stage)
            if not os.path.isdir(sp):
                continue
            for r in sorted(os.listdir(sp)):
                rp = os.path.join(sp, r)
                if os.path.isdir(rp):
                    out.append(rp)
    return out


def gpu_idle(gpus=(4, 5, 6, 7)):
    """HARD RULE: idle <=> mem < 1000 MiB AND util < 10%. Verified before EVERY launch."""
    import subprocess
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits", "-i", ",".join(map(str, gpus))],
        capture_output=True, text=True, check=True).stdout.strip()
    idle = []
    for line in out.splitlines():
        i, m, u = [int(x.strip()) for x in line.split(",")]
        if m < 1000 and u < 10:
            idle.append(i)
    return idle
