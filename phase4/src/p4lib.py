"""Phase-4 shared library.

Carries the Phase-3 audit hardening forward VERBATIM (marker-gated atomic reads, the probe-leak
guard, the row-wise logit clamp) -- Phase-3 source is read-only, so the hardened readers are
re-exported here rather than imported across a frozen phase boundary. Never regress to
existence-only reads.

Adds the two things Phase 4 needs on top:

  * EXACT RATIONAL CRITERION ARITHMETIC (meets_half_ceiling). The criterion is
    `rho >= 0.5 * ceiling`. Evaluated in binary floating point this is a comparison of two
    rounded quantities, and Phase-4's verdicts turn on it. fractions.Fraction(float) is the
    EXACT binary value of the double, so Fraction(rho) >= Fraction(1,2)*Fraction(ceiling) is
    the exact comparison of the two doubles actually computed. No epsilon, no tie-breaking.

  * SPEARMAN-BROWN helpers, and the k-vs-k split-half ceiling estimator generalized over the
    aggregator (mean for the BC arm, MEDIAN for the diffusion arm -- P13's preregistered
    remedy for the heavy tail).

SB CAVEAT, stated here because P13 depends on it: Spearman-Brown is EXACT for means of
exchangeable replicates and only APPROXIMATE for medians. P13's preregistration discloses this
and reports both the uncorrected split-half and the SB-corrected value.
"""
import functools
import hashlib
import itertools
import json
import os
import sys
from fractions import Fraction

import numpy as np

ROOT = "/mnt/sdb/ljc/RoboTDA-X"
sys.path.insert(0, os.path.join(ROOT, "src"))

import bootstrap  # noqa: F401,E402  (structural GPU pinning -- never regress this)
from bootstrap import RESULTS, RUNS  # noqa: E402
from lds import spearman  # noqa: E402

P4 = os.path.join(ROOT, "phase4")
P4_RESULTS = os.path.join(P4, "results")
P4_RUNS = os.path.join(P4, "runs")
P4_FIGURES = os.path.join(P4, "figures")
P4_LOGS = os.path.join(P4, "logs")
P2_RESULTS = os.path.join(ROOT, "phase2", "results")
P2_RUNS = os.path.join(ROOT, "phase2", "runs")
P3_RESULTS = os.path.join(ROOT, "phase3", "results")
P3_RUNS = os.path.join(ROOT, "phase3", "runs")
P3_SRC = os.path.join(ROOT, "phase3", "src")

for _d in (P4_RESULTS, P4_RUNS, P4_FIGURES, P4_LOGS):
    os.makedirs(_d, exist_ok=True)

PREREG = os.path.join(P4, "preregistration_phase4.json")
PREREG_SHA = "fca37f54804c0173f5a9029e99b220de3db2a97ca2af7b3cf3c1c1a5372ecb80"

ARTIFACT_MARKER = {
    "outcomes.json": "probe",
    "cluster_eval.json": "clustereval",
    "suite_outcomes.json": "suite",
}


# ---------------------------------------------------------------- preregistration integrity
def assert_prereg_locked():
    """Every Phase-4 verdict script calls this FIRST. The prereg must be on disk, unmodified."""
    got = sha256_file(PREREG)
    if got != PREREG_SHA:
        raise RuntimeError(
            f"PREREGISTRATION MISMATCH: {PREREG} hashes to {got}, expected {PREREG_SHA}. "
            f"The preregistration was locked before any Phase-4 training or verdict. A change "
            f"to it after the fact invalidates every confirmatory claim. Refusing to proceed.")
    return got


# ---------------------------------------------------------------- atomic writes (P6.1)
def atomic_write_json(path, obj, indent=1):
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


# ---------------------------------------------------------------- marker-gated reading (P6.1)
def is_marked(run_dir, marker):
    return os.path.exists(os.path.join(run_dir, f"{marker}.marker"))


def read_artifact(run_dir, artifact="outcomes.json", required=True):
    """MARKER-GATED ingestion. An artifact without its completion marker is a partially-failed
    run (rollout.py writes the artifact BEFORE it raises). REFUSED, loudly."""
    marker = ARTIFACT_MARKER[artifact]
    apath = os.path.join(run_dir, artifact)
    has_a, has_m = os.path.exists(apath), is_marked(run_dir, marker)
    if has_a and not has_m:
        raise RuntimeError(
            f"MARKER-GATE VIOLATION: {apath} exists but {marker}.marker does not. "
            f"Partially-failed run. Refusing to ingest.")
    if not has_a:
        if required:
            raise FileNotFoundError(f"missing artifact: {apath}")
        return None
    return json.load(open(apath))


def read_outcomes(run_dir, required=True):
    r = read_artifact(run_dir, "outcomes.json", required=required)
    return None if r is None else r["outcomes"]


def scan_marker_gate(run_dirs):
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
    return {"n_run_dirs_scanned": len(run_dirs), "n_artifact_marker_pairs_ok": ok,
            "n_violations": len(viol), "violations": viol,
            "n_dirs_with_no_eval_artifact": len(empty)}


# ---------------------------------------------------------------- probe-leak guard (P6.2)
@functools.lru_cache(maxsize=1)
def phase2_probe_ids():
    """The 135 Phase-2 per-task probe demos. An empty/wrong-sized set would DISARM the guard,
    so the post-condition below is fatal (Phase-3 incident 1: the guard once passed vacuously)."""
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
    expect = int(p["n_tasks"]) * int(p["n_demos_per_task"])
    if len(ids) != expect:
        raise RuntimeError(
            f"phase2_probe_ids(): extracted {len(ids)} ids, manifest declares {expect}. "
            f"A wrong-sized probe set DISARMS assert_no_probe_leak(). Fatal.")
    return frozenset(ids)


def run_demos(run_dir):
    p = os.path.join(run_dir, "demos.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"no demos.json in {run_dir}")
    return set(json.load(open(p))["demos"])


def assert_no_probe_leak(run_dirs, probe_ids, context=""):
    """HARD GUARD. Refuse attribution for any model trained on demos in the probe/test set."""
    probe_ids = frozenset(probe_ids)
    if not probe_ids:
        raise RuntimeError("assert_no_probe_leak called with an EMPTY probe set -- that would "
                           "pass vacuously. Fatal.")
    bad = []
    for d in run_dirs:
        inter = run_demos(d) & probe_ids
        if inter:
            bad.append({"run_dir": d, "n_leaked": len(inter), "examples": sorted(inter)[:5]})
    if bad:
        raise RuntimeError(f"PROBE LEAK REFUSED [{context}]: {len(bad)} model(s) trained on "
                           f"demos in the probe set being attributed against. Offenders: {bad}")
    return {"n_models_checked": len(run_dirs), "n_probe_ids": len(probe_ids), "clean": True}


# ---------------------------------------------------------------- episode-count fix (P6.3)
def logit_success_rowwise(p, n_episodes):
    """logit(clamp(p, 1/(2n), 1-1/(2n))), n READ PER ROW."""
    p = np.asarray(p, dtype=float)
    n = np.asarray(n_episodes, dtype=float)
    lo = 1.0 / (2.0 * n)
    p = np.clip(p, lo, 1.0 - lo)
    return np.log(p / (1.0 - p))


# ---------------------------------------------------------------- EXACT criterion arithmetic
def meets_half_ceiling(rho, ceiling):
    """EXACT: Fraction(rho) >= 1/2 * Fraction(ceiling), on the doubles actually computed.

    Fraction(float) is the exact binary value of the double, so this compares the two numbers
    that exist -- no epsilon, no rounding, no tie ambiguity. A criterion that decides a paper's
    verdict is not evaluated in approximate arithmetic.
    """
    if not (np.isfinite(rho) and np.isfinite(ceiling)):
        return False
    return Fraction(float(rho)) >= Fraction(1, 2) * Fraction(float(ceiling))


def ratio_exact_str(rho, ceiling):
    """The ratio as an exact rational, for the audit trail (also returned as a float)."""
    if not (np.isfinite(rho) and np.isfinite(ceiling)) or ceiling == 0:
        return None
    return str(Fraction(float(rho)) / Fraction(float(ceiling)))


# ---------------------------------------------------------------- Spearman-Brown
def sb(r, k):
    """Spearman-Brown: reliability of a k-fold aggregate given the 1-fold reliability r."""
    if not np.isfinite(r):
        return np.nan
    den = 1.0 + (k - 1) * r
    return float(k * r / den) if den != 0 else np.nan


def sb_half_to_full(r_half):
    """The x2 correction: reliability of a full-length aggregate from its split-half value."""
    return sb(r_half, 2)


def split_half_ceiling(piv, seeds, agg="mean", max_splits=None):
    """Mean Spearman over ALL distinct k-vs-k seed split-halves (k = len(seeds)//2).

    piv: DataFrame indexed by mask_id, columns = seeds, values = the outcome.
    agg: 'mean' (BC) or 'median' (P13's preregistered heavy-tail remedy).

    Returns (r_half, r_full_SB, per_split, n_splits). r_full_SB = 2*r_half/(1+r_half).
    """
    seeds = [s for s in seeds if s in piv.columns]
    piv = piv[seeds].dropna()
    k = len(seeds) // 2
    if len(seeds) < 2 or k < 1 or piv.shape[0] < 4:
        return np.nan, np.nan, [], 0
    f = (lambda d: d.mean(axis=1)) if agg == "mean" else (lambda d: d.median(axis=1))
    seen, vals = set(), []
    for half in itertools.combinations(seeds, k):
        other = tuple(s for s in seeds if s not in half)
        if len(other) != k:
            continue                      # odd seed count: no balanced split
        key = frozenset([half, other])
        if key in seen:
            continue
        seen.add(key)
        r = spearman(f(piv[list(half)]).values, f(piv[list(other)]).values)
        if np.isfinite(r):
            vals.append(float(r))
    if max_splits is not None and len(vals) > max_splits:
        vals = vals[:max_splits]
    if not vals:
        return np.nan, np.nan, [], 0
    r_half = float(np.mean(vals))
    return r_half, sb_half_to_full(r_half), vals, len(vals)


def mean_pairwise_1seed_r(piv, seeds):
    """r1: mean Spearman between every pair of SINGLE seeds. Aggregator-free (no averaging),
    so it is the correct base for a Spearman-Brown extrapolation to any depth, for the mean
    AND for the median arm."""
    seeds = [s for s in seeds if s in piv.columns]
    piv = piv[seeds].dropna()
    vals = [spearman(piv[a].values, piv[b].values) for a, b in itertools.combinations(seeds, 2)]
    vals = [v for v in vals if np.isfinite(v)]
    return (float(np.mean(vals)) if vals else np.nan), vals


# ---------------------------------------------------------------- per-member normalization
def normalized_ensemble_scores(per_member_df, attributor, target, demo_ids, members,
                               normalize=True):
    """The PREREGISTERED aggregation: per-member scale normalization (unit L2), then MEAN.

    per_member_df: columns [attributor, target, demo_id, member, score] (functional already
    filtered to 'plain' by the caller if the frame carries one).
    -> {demo_id: score}.  Raises if any (member, demo) cell is missing -- a silently short
    ensemble would be a different estimator.
    """
    sub = per_member_df[(per_member_df.attributor == attributor)
                        & (per_member_df.target == target)]
    piv = sub.pivot_table(index="demo_id", columns="member", values="score")
    missing_m = [m for m in members if m not in piv.columns]
    if missing_m:
        raise RuntimeError(f"normalized_ensemble_scores: members missing for {attributor}/"
                           f"{target}: {missing_m}")
    piv = piv.reindex(index=demo_ids)[list(members)]
    if piv.isna().any().any():
        raise RuntimeError(f"normalized_ensemble_scores: NaN cells for {attributor}/{target} "
                           f"({int(piv.isna().sum().sum())} of {piv.size})")
    M = piv.values.astype(float)                       # (n_demos, n_members)
    if normalize:
        nrm = np.linalg.norm(M, axis=0, keepdims=True)  # per-member L2 over the demo vector
        if np.any(nrm == 0):
            raise RuntimeError(f"zero-norm member score vector for {attributor}/{target}")
        M = M / nrm
    return {d: float(v) for d, v in zip(demo_ids, M.mean(axis=1))}


# ---------------------------------------------------------------- convenience
def gpu_idle(gpus=(4, 5, 6, 7)):
    """HARD RULE: idle <=> mem < 1000 MiB AND util < 10%."""
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
