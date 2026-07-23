"""P6 audit sweeps: marker gate (6.1), probe leak (6.2), episode count (6.3), G6 integrity (6.4).

Zero retrains, zero GPU. Every check is a machine-check against artifacts on disk; each writes
its own JSON. A failure in any of these is an INSTRUMENT DEFECT -> stop and write
PHASE3_DEFECT.md before anything else.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3lib as L
from p3lib import P3_RESULTS, RESULTS, P2_RESULTS, RUNS, P2_RUNS


# ---------------------------------------------------------------- 6.1 marker gate
def sweep_marker_gate():
    dirs = L.all_phase12_run_dirs()
    r = L.scan_marker_gate(dirs)
    r["defect"] = ("rollout.py writes outcomes.json/cluster_eval.json/suite_outcomes.json "
                   "BEFORE raising on rollout errors and BEFORE writing the completion marker; "
                   "all Phase-1/2 readers ingest on file-existence only.")
    r["fix"] = ("phase3/src/p3lib.py: atomic_write_json (tmp+os.replace) for writes; "
                "read_artifact() REFUSES an artifact whose marker is absent. Every Phase-3 "
                "reader uses it.")
    r["VERDICT"] = "CLEAN" if r["n_violations"] == 0 else "DEFECT"
    L.atomic_write_json(os.path.join(P3_RESULTS, "p6_marker_sweep.json"), r)
    print(f"[6.1] scanned {r['n_run_dirs_scanned']} run dirs, "
          f"{r['n_artifact_marker_pairs_ok']} artifact+marker pairs OK, "
          f"{r['n_violations']} VIOLATIONS -> {r['VERDICT']}")
    if r["n_violations"]:
        for v in r["violations"][:10]:
            print("   VIOLATION:", v)
    return r


# ---------------------------------------------------------------- 6.2 probe leak
def sweep_probe_leak():
    """Machine-check that no influence artifact was ever computed from a model trained on the
    probe demos it was (or could have been) scored against.

    The probe set at issue is Phase 2's per-task probe set (demo_45..demo_49, 27 tasks). It is
    used as the TEST-SIDE functional by phase2/results/per_task_influence_stage{B,E}*.parquet.
    Every other influence artifact uses the 9 clusters' 10 held-out demos as the test side.
    We check BOTH probe sets against BOTH the ensembles that produced each artifact.
    """
    probes = L.phase2_probe_ids()

    # the held-out pool (the OTHER test-side functional used by Phase-1 influence tables)
    sys.path.insert(0, os.path.join(L.ROOT, "src"))
    import dataset
    heldout = set(dataset.heldout_pool()[0])

    # map each influence artifact -> the ensemble run dirs whose gradients define it
    artifacts = {
        "results/influence_table.parquet": {
            "runs": sorted(glob.glob(os.path.join(RUNS, "stage_E", "ens_s*"))),
            "test_side": "9 clusters x 10 held-out demos", "probe_set": heldout},
        "results/influence_table_per_member.parquet": {
            "runs": sorted(glob.glob(os.path.join(RUNS, "stage_E", "ens_s*"))),
            "test_side": "9 clusters x 10 held-out demos", "probe_set": heldout},
        "results/stage_D_influence_C1.parquet": {
            "runs": sorted(glob.glob(os.path.join(RUNS, "stage_E", "ens_s*"))),
            "test_side": "C1 10 held-out demos", "probe_set": heldout},
        "phase2/results/per_task_influence_stageE.parquet": {
            "runs": sorted(glob.glob(os.path.join(RUNS, "stage_E", "ens_s*"))),
            "test_side": "27 tasks x demo_45..49 (Phase-2 per-task probes)", "probe_set": probes},
        "phase2/results/per_task_influence_stageB.parquet": {
            "runs": sorted(glob.glob(os.path.join(RUNS, "stage_B", "*cotrain*"))),
            "test_side": "27 tasks x demo_45..49 (Phase-2 per-task probes)", "probe_set": probes},
        "phase2/results/p3_influence_Q15.parquet": {
            "runs": sorted(glob.glob(os.path.join(P2_RUNS, "P3", "Q15full_s*"))),
            "test_side": "C1 10 held-out demos", "probe_set": heldout},
        "phase2/results/p3_influence_Q50.parquet": {
            "runs": sorted(glob.glob(os.path.join(P2_RUNS, "P3", "Q50full_s*"))),
            "test_side": "C1 10 held-out demos", "probe_set": heldout},
        "phase2/results/p3_influence_Q150.parquet": {
            "runs": sorted(glob.glob(os.path.join(P2_RUNS, "P3", "Q150full_s*"))),
            "test_side": "C1 10 held-out demos", "probe_set": heldout},
    }

    rows, total_bad = [], 0
    for art, spec in artifacts.items():
        apath = os.path.join(L.ROOT, art)
        if not os.path.exists(apath):
            rows.append({"artifact": art, "status": "ABSENT", "n_models": 0})
            continue
        runs = [r for r in spec["runs"] if os.path.exists(os.path.join(r, "demos.json"))]
        # NON-VACUITY GUARD: a check over zero models, or against an empty probe set, would
        # "pass" while checking nothing. Both are fatal here.
        if not runs:
            raise RuntimeError(f"probe-leak check for {art} resolved ZERO model dirs -- the run "
                               f"glob is wrong. A check over no models passes vacuously.")
        if not spec["probe_set"]:
            raise RuntimeError(f"probe-leak check for {art} has an EMPTY probe set.")
        bad = []
        for r in runs:
            inter = L.run_demos(r) & spec["probe_set"]
            if inter:
                bad.append({"run": os.path.basename(r), "n_leaked": len(inter),
                            "examples": sorted(inter)[:3]})
        total_bad += len(bad)
        rows.append({"artifact": art, "status": "CHECKED", "n_models": len(runs),
                     "test_side": spec["test_side"], "n_leaking_models": len(bad),
                     "leaks": bad, "clean": len(bad) == 0})
        print(f"[6.2] {art}: {len(runs)} models, test-side = {spec['test_side']}, "
              f"leaks = {len(bad)} {'CLEAN' if not bad else 'LEAK!'}")

    # the loaded gun itself: DO the Q490 training sets contain the probe demos?
    q490 = sorted(glob.glob(os.path.join(RUNS, "stage_C", "*490*")))
    gun = []
    for r in q490:
        if os.path.exists(os.path.join(r, "demos.json")):
            inter = L.run_demos(r) & probes
            gun.append({"run": os.path.basename(r), "n_demos": len(L.run_demos(r)),
                        "n_probe_demos_in_training_set": len(inter),
                        "examples": sorted(inter)[:3]})
    scored = any(a.get("n_leaking_models", 0) for a in rows)

    out = {
        "defect": ("Stage-C Q=490 training sets contain the Phase-2 per-task probe demos "
                   "(demo_45..demo_49). Harmless today because no Q490 model is ever scored "
                   "against them, but nothing PREVENTED it."),
        "fix": ("p3lib.assert_no_probe_leak() -- a hard assertion in every Phase-3 attribution "
                "entry point. It REFUSES to compute attribution for any model whose demos.json "
                "intersects the probe set being used."),
        "loaded_gun_confirmed": gun,
        "loaded_gun_note": ("These Q490 models DO contain probe demos in their training sets. "
                            "The check below is whether any of them was ever ATTRIBUTED against "
                            "those probes. Expected: no -- no influence artifact is derived from "
                            "a Q490 model."),
        "per_artifact_checks": rows,
        "n_leaking_model_artifact_pairs": total_bad,
        "VERDICT": "CLEAN" if total_bad == 0 else "DEFECT",
        "conclusion": ("No influence artifact in the study was computed from a model whose "
                       "training set intersects its test-side probe set."
                       if total_bad == 0 else "LEAK FOUND -- see per_artifact_checks."),
    }
    L.atomic_write_json(os.path.join(P3_RESULTS, "p6_probe_leak_check.json"), out)
    print(f"[6.2] loaded gun: {len(gun)} Q490 runs, "
          f"{sum(g['n_probe_demos_in_training_set'] for g in gun)} probe-demo inclusions total")
    print(f"[6.2] VERDICT: {out['VERDICT']} ({total_bad} leaking model-artifact pairs)")
    return out


# ---------------------------------------------------------------- 6.3 episode count
def check_episode_counts():
    """The P6.3 fix is a no-op iff every table consumed by a HARDCODING reader has n=30.

    The question is not 'does every table in the study have n_episodes == 30' -- it does not:
    phase2/results/p3_outcomes.parquet has n = 200. The question is whether any reader that
    HARDCODES the clamp is ever pointed at a table whose n != 30. So the check is scoped by the
    reader -> table map, established by grepping every call site of lds.logit_success:

      HARDCODING readers (the defect):
        src/analysis.py:44      lds.logit_success(row["success_rate"], N_EPISODES=30)
                                consumes stage_F_outcomes.parquet, stage_G_outcomes.parquet
        src/stage_d.py:85       lds.logit_success(sc, 30)
                                consumes runs/stage_D/*/outcomes.json
      ROW-WISE readers (already correct, not a defect):
        phase2/src/p1_analyze.py:85, p3_analyze.py:89, p4_analyze.py:89
                                pass df.n_episodes.values per row
    """
    HARDCODING = {
        "src/analysis.py:44 (N_EPISODES=30)": ["results/stage_F_outcomes.parquet",
                                               "results/stage_G_outcomes.parquet"],
        "src/stage_d.py:85 (literal 30)": ["runs/stage_D/*/outcomes.json"],
    }
    ROWWISE = {
        "phase2/src/p1_analyze.py:85": ["phase2/results/stage_G6_outcomes.parquet"],
        "phase2/src/p3_analyze.py:89": ["phase2/results/p3_outcomes.parquet"],
        "phase2/src/p4_analyze.py:89": ["phase2/runs/P4/*/ (per-episode subsampling)"],
    }

    def episodes_of(t):
        if t.endswith(".parquet"):
            df = pd.read_parquet(os.path.join(L.ROOT, t))
            return sorted(int(x) for x in df.n_episodes.unique()), len(df)
        ns, n = set(), 0
        for r in sorted(glob.glob(os.path.join(L.ROOT, t))):
            for c, v in json.load(open(r))["outcomes"].items():
                ns.add(int(v["n_episodes"]))
                n += 1
        return sorted(ns), n

    hard, all30 = {}, True
    for reader, tabs in HARDCODING.items():
        for t in tabs:
            u, n = episodes_of(t)
            ok = (u == [30])
            all30 &= ok
            hard[f"{reader} -> {t}"] = {"n_episodes_unique": u, "n_rows": n,
                                        "clamp_correct": ok}
            print(f"[6.3] HARDCODING {reader}\n            -> {t}: {n} rows, n_episodes={u} "
                  f"{'OK (30 == 30, inert)' if ok else 'MISMATCH -- CLAMP WAS WRONG'}")

    soft = {}
    for reader, tabs in ROWWISE.items():
        for t in tabs:
            if t.endswith(".parquet"):
                u, n = episodes_of(t)
                soft[f"{reader} -> {t}"] = {"n_episodes_unique": u, "n_rows": n,
                                            "reads_n_episodes_per_row": True}
                print(f"[6.3] ROW-WISE   {reader}\n            -> {t}: {n} rows, "
                      f"n_episodes={u} (handled correctly per row)")
            else:
                soft[f"{reader} -> {t}"] = {"reads_n_episodes_per_row": True}

    res = {
        "defect": ("src/analysis.py:32/44 hardcodes N_EPISODES=30 for the logit clamp; "
                   "src/stage_d.py:85 hardcodes the literal 30."),
        "fix": "phase3/src/p3lib.py:logit_success_rowwise reads the row's own n_episodes.",
        "scoping_note": ("phase2/results/p3_outcomes.parquet carries n_episodes = 200, NOT 30. "
                         "It is consumed ONLY by phase2/src/p3_analyze.py, which already passes "
                         "df.n_episodes.values ROW-WISE and is therefore correct. No hardcoding "
                         "reader is ever pointed at a table whose n != 30."),
        "hardcoding_readers": hard,
        "rowwise_readers_already_correct": soft,
        "every_table_read_by_a_hardcoding_reader_has_n_30": bool(all30),
        "VERDICT": "NO-OP CONFIRMED" if all30 else "NUMBERS WOULD MOVE",
        "note": ("Every table a hardcoding reader consumes has n_episodes == 30, so the clamp it "
                 "applied was the correct one and the fix is provably numerically inert on all "
                 "existing data. That is exactly the claim: a LATENT defect, not an active "
                 "error. p6_no_change.json proves the headline table is unchanged."
                 if all30 else
                 "A HARDCODING READER CONSUMED A TABLE WITH n != 30 -- the clamp WAS wrong and "
                 "real numbers move. This is an active error and must be reported as one."),
    }
    L.atomic_write_json(os.path.join(P3_RESULTS, "p6_episode_count_check.json"), res)
    print(f"[6.3] VERDICT: {res['VERDICT']}")
    return res


# ---------------------------------------------------------------- 6.4 G6 integrity
def check_g6_integrity():
    """96 run dirs intact; parquet re-derives from them; SHA-256 recorded for future copies."""
    base = os.path.join(P2_RUNS, "stage_G6")
    dirs = sorted(os.listdir(base))
    need = ["final.pt", "outcomes.json", "train.marker", "probe.marker", "demos.json",
            "train_meta.json"]
    manifest, bad = {}, []
    for d in dirs:
        p = os.path.join(base, d)
        files = {f: os.path.getsize(os.path.join(p, f)) for f in sorted(os.listdir(p))}
        manifest[d] = files
        miss = [f for f in need if f not in files]
        zero = [f for f, s in files.items() if s == 0]
        if miss or zero:
            bad.append({"run": d, "missing": miss, "zero_byte": zero})

    # --- re-derive the parquet from the raw outcomes.json (MARKER-GATED)
    jobs = json.load(open(os.path.join(P2_RESULTS, "p1_jobs.json")))
    rederived = []
    for j in jobs:
        rd = j["run_dir"]
        oc = L.read_outcomes(rd, required=True)          # <-- marker-gated read
        for c, v in oc.items():
            rederived.append({"run": os.path.basename(rd), "mask_id": j["mask_id"],
                              "seed": j["seed"], "target": c,
                              "success_rate": v["success_rate"], "n_episodes": v["n_episodes"],
                              "plain_loss": v["plain_loss"],
                              "transport_loss": v["transport_loss"],
                              "interaction_loss": v["interaction_loss"]})
    RD = pd.DataFrame(rederived)

    ppath = os.path.join(P2_RESULTS, "stage_G6_outcomes.parquet")
    PQ = pd.read_parquet(ppath)
    # the parquet ALSO contains the 2 Phase-1 seeds (401,402) from stage_G; the 96 G6 runs are
    # seeds 403-406. Compare on the overlap.
    new_seeds = sorted(RD.seed.unique())
    PQn = PQ[PQ.seed.isin(new_seeds)]

    key = ["run", "mask_id", "seed", "target"]
    M = PQn.merge(RD, on=key, suffixes=("_pq", "_raw"))
    cols = ["success_rate", "plain_loss", "transport_loss", "interaction_loss"]
    maxdiff = {c: float(np.abs(M[f"{c}_pq"] - M[f"{c}_raw"]).max()) for c in cols}
    coverage = {
        "n_raw_rows_from_run_dirs": len(RD),
        "n_parquet_rows_for_those_seeds": len(PQn),
        "n_merged": len(M),
        "n_masks": int(PQ.mask_id.nunique()),
        "n_seeds_total": int(PQ.seed.nunique()),
        "n_targets": int(PQ.target.nunique()),
        "expected_total_rows_24x6x9": 24 * 6 * 9,
        "actual_total_parquet_rows": len(PQ),
        "coverage_complete": len(PQ) == 24 * 6 * 9,
    }

    # --- spot-recompute 5 rows straight from the raw JSON (independent of the merge above)
    rng = np.random.default_rng(1234)
    spot = []
    idx = rng.choice(len(PQn), size=5, replace=False)
    for i in idx:
        r = PQn.iloc[int(i)]
        raw = json.load(open(os.path.join(base, r["run"], "outcomes.json")))["outcomes"][r["target"]]
        spot.append({
            "run": r["run"], "seed": int(r["seed"]), "target": r["target"],
            "parquet": {c: float(r[c]) for c in cols},
            "raw_outcomes_json": {c: float(raw[c]) for c in cols},
            "match": all(abs(float(r[c]) - float(raw[c])) == 0.0 for c in cols),
        })

    out = {
        "defect_context": ("On a COPY of this project made to another machine, 41/96 "
                           "phase2/runs/stage_G6 run dirs were zero-byte -- a truncated file "
                           "transfer, not a run failure. This check verifies the runs ON THIS "
                           "HOST and records hashes so any future copy is checkable."),
        "host_check": {
            "n_dirs": len(dirs),
            "n_bad_dirs": len(bad),
            "bad_dirs": bad,
            "required_files": need,
            "total_bytes": int(sum(sum(f.values()) for f in manifest.values())),
            "all_intact": len(bad) == 0 and len(dirs) == 96,
        },
        "parquet_rederivation": {
            "coverage": coverage,
            "max_abs_diff_parquet_vs_raw": maxdiff,
            "rederives_exactly": all(v == 0.0 for v in maxdiff.values()),
        },
        "spot_recomputed_rows": spot,
        "n_spot_rows_matching": sum(s["match"] for s in spot),
        "sha256_stage_G6_outcomes_parquet": L.sha256_file(ppath),
        "sha256_run_dir_size_manifest": L.sha256_obj(manifest),
        "run_dir_size_manifest": manifest,
        "VERDICT": None,
    }
    out["VERDICT"] = ("INTACT" if (out["host_check"]["all_intact"]
                                   and out["parquet_rederivation"]["rederives_exactly"]
                                   and out["n_spot_rows_matching"] == 5
                                   and coverage["coverage_complete"]) else "DEFECT")
    L.atomic_write_json(os.path.join(P3_RESULTS, "p6_g6_integrity.json"), out)
    print(f"[6.4] {len(dirs)} dirs, {len(bad)} bad | parquet rows {len(PQ)} "
          f"(expect {24*6*9}) | max|diff| {max(maxdiff.values())} | "
          f"spot {out['n_spot_rows_matching']}/5 | VERDICT {out['VERDICT']}")
    print(f"[6.4] sha256(stage_G6_outcomes.parquet) = {out['sha256_stage_G6_outcomes_parquet']}")
    return out


def main():
    print("=" * 88)
    print("P6 AUDIT-HARDENING SWEEPS")
    print("=" * 88)
    r1 = sweep_marker_gate()
    print()
    r2 = sweep_probe_leak()
    print()
    r3 = check_episode_counts()
    print()
    r4 = check_g6_integrity()
    print()
    verdicts = {"6.1_marker_gate": r1["VERDICT"], "6.2_probe_leak": r2["VERDICT"],
                "6.3_episode_count": r3["VERDICT"], "6.4_g6_integrity": r4["VERDICT"]}
    ok = (r1["VERDICT"] == "CLEAN" and r2["VERDICT"] == "CLEAN"
          and r3["VERDICT"] == "NO-OP CONFIRMED" and r4["VERDICT"] == "INTACT")
    print("=" * 88)
    for k, v in verdicts.items():
        print(f"  {k:22s} {v}")
    print(f"  ALL CHECKS PASS: {ok}")
    print("=" * 88)
    L.atomic_write_json(os.path.join(P3_RESULTS, "p6_sweeps_summary.json"),
                        {"verdicts": verdicts, "ALL_PASS": bool(ok)})
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
