"""Phase-5 shared library.

Carries the Phase-3/4 audit hardening forward VERBATIM (marker-gated atomic reads, the probe-leak
guard, the row-wise logit clamp, EXACT rational criterion arithmetic, the SB / split-half ceiling
helpers generalized over the aggregator). Prior-phase source is read-only, so the hardened readers
are re-exported here rather than imported across a frozen phase boundary. Never regress to
existence-only reads.

The only thing Phase 5 adds is a thin convenience: a single-seed-reliability -> SB-consistency
helper keyed by an explicit seed SUBSET, because P15's stricter gate extrapolates from S=4 (seeds
601-604) AND from S=8 (601-608) to S=10 -- both bases are aggregator-free single-seed reliabilities
and therefore correct SB bases for either the mean or the median arm.
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

P5 = os.path.join(ROOT, "phase5")
P5_RESULTS = os.path.join(P5, "results")
P5_RUNS = os.path.join(P5, "runs")
P5_FIGURES = os.path.join(P5, "figures")
P5_LOGS = os.path.join(P5, "logs")
P2_RESULTS = os.path.join(ROOT, "phase2", "results")
P3_RESULTS = os.path.join(ROOT, "phase3", "results")
P3_RUNS = os.path.join(ROOT, "phase3", "runs")
P3_SRC = os.path.join(ROOT, "phase3", "src")
P4_RESULTS = os.path.join(ROOT, "phase4", "results")
P4_RUNS = os.path.join(ROOT, "phase4", "runs")

for _d in (P5_RESULTS, P5_RUNS, P5_FIGURES, P5_LOGS):
    os.makedirs(_d, exist_ok=True)

PREREG = os.path.join(P5, "preregistration_phase5.json")
PREREG_SHA = "b4883cb1a7048aee1a47410291573c3e27401c7e08be7852f67c58f42f2922b8"

ARTIFACT_MARKER = {
    "outcomes.json": "probe",
    "cluster_eval.json": "clustereval",
    "suite_outcomes.json": "suite",
}


# ---------------------------------------------------------------- preregistration integrity
def assert_prereg_locked():
    """Every Phase-5 verdict script calls this FIRST. The prereg must be on disk, unmodified."""
    got = sha256_file(PREREG)
    if got != PREREG_SHA:
        raise RuntimeError(
            f"PREREGISTRATION MISMATCH: {PREREG} hashes to {got}, expected {PREREG_SHA}. "
            f"The preregistration was locked before any Phase-5 training or verdict. A change "
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
    """EXACT: Fraction(rho) >= 1/2 * Fraction(ceiling), on the doubles actually computed."""
    if not (np.isfinite(rho) and np.isfinite(ceiling)):
        return False
    return Fraction(float(rho)) >= Fraction(1, 2) * Fraction(float(ceiling))


def ratio_exact_str(rho, ceiling):
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
    return sb(r_half, 2)


def split_half_ceiling(piv, seeds, agg="mean", max_splits=None):
    """Mean Spearman over ALL distinct k-vs-k seed split-halves (k = len(seeds)//2).

    piv: DataFrame indexed by mask_id, columns = seeds, values = the outcome.
    agg: 'mean' (BC) or 'median' (the diffusion heavy-tail remedy).
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
            continue
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
    """r1: mean Spearman between every pair of SINGLE seeds. Aggregator-free."""
    seeds = [s for s in seeds if s in piv.columns]
    piv = piv[seeds].dropna()
    vals = [spearman(piv[a].values, piv[b].values) for a, b in itertools.combinations(seeds, 2)]
    vals = [v for v in vals if np.isfinite(v)]
    return (float(np.mean(vals)) if vals else np.nan), vals


def sb_consistency(piv, base_seeds, k_target):
    """SB-consistency BASE: single-seed reliability over `base_seeds` extrapolated to depth
    k_target. Returns (r1, n_pairs, predicted_k). The gate compares |predicted_k - measured|."""
    r1, pairs = mean_pairwise_1seed_r(piv, base_seeds)
    return r1, len(pairs), sb(r1, k_target)


# ---------------------------------------------------------------- per-member normalization
def normalized_ensemble_scores(per_member_df, attributor, target, demo_ids, members,
                               normalize=True):
    """PREREGISTERED aggregation: per-member scale normalization (unit L2), then MEAN.
    Raises if any (member, demo) cell is missing -- a silently short ensemble is a different
    estimator."""
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
    M = piv.values.astype(float)
    if normalize:
        nrm = np.linalg.norm(M, axis=0, keepdims=True)
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
