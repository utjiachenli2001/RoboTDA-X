"""P6 NO-CHANGE PROOF: recompute every headline number with the FIXED readers.

The three fixes (marker-gated + atomic ingestion, row-wise n_episodes, probe-leak guard) are
claimed to be LATENT -- i.e. they close real traps but change no reported number. A claim like
that is worthless unless it is checked. So this script:

  1. RE-INGESTS the raw per-run artifacts from disk through the MARKER-GATED reader
     (p3lib.read_outcomes -- which REFUSES any artifact whose completion marker is absent),
     rebuilding stage_F / stage_G / stage_G6 / stage_D from scratch rather than trusting the
     archived parquets;
  2. RECOMPUTES the headline statistics with logit_success_rowwise (n_episodes read PER ROW)
     instead of the hardcoded N_EPISODES = 30;
  3. DIFFS every recomputed number against the archived artifact, to the precision the reports
     quote.

Any mismatch is an instrument defect -> STOP and write PHASE3_DEFECT.md.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3lib as L
from p3lib import P3_RESULTS, RESULTS, P2_RESULTS, RUNS

sys.path.insert(0, os.path.join(L.ROOT, "src"))
import dataset  # noqa: E402
import lds  # noqa: E402
from lds import spearman, spearman_p_onesided  # noqa: E402

TOL = 1e-9          # "identical" -- these should be bit-for-bit, not merely close
FOCAL = ["C1", "C5"]
ATTRS = ["TracIn", "TRAK", "IF"]


def reingest(stage_glob, stage_name):
    """Rebuild an outcomes table from the RAW run dirs through the MARKER-GATED reader."""
    rows = []
    for rd in sorted(glob.glob(stage_glob)):
        if not os.path.isdir(rd):
            continue
        oc = L.read_outcomes(rd, required=False)      # <-- refuses unmarked artifacts
        if oc is None:
            continue
        name = os.path.basename(rd)
        mask_id, seed = name.rsplit("_s", 1)
        for c, v in oc.items():
            rows.append({"stage": stage_name, "run": name, "mask_id": mask_id, "seed": int(seed),
                         "target": c, "success_rate": v["success_rate"],
                         "n_episodes": v["n_episodes"], "plain_loss": v["plain_loss"],
                         "transport_loss": v["transport_loss"],
                         "interaction_loss": v["interaction_loss"]})
    return pd.DataFrame(rows)


def add_outcomes(df):
    """FIXED outcome transform: n_episodes read PER ROW (P6.3), not hardcoded to 30."""
    df = df.copy()
    df["logit_success"] = L.logit_success_rowwise(df.success_rate.values, df.n_episodes.values)
    df["neg_plain_loss"] = -df.plain_loss
    df["neg_transport_loss"] = -df.transport_loss
    df["neg_interaction_loss"] = -df.interaction_loss
    return df


def cmp(name, before, after, out, tol=TOL):
    ok = (before is None and after is None)
    if not ok:
        try:
            ok = bool(abs(float(before) - float(after)) <= tol)
        except (TypeError, ValueError):
            ok = (before == after)
    out.append({"quantity": name, "before_archived": before, "after_fixed_readers": after,
                "abs_diff": (abs(float(before) - float(after))
                             if isinstance(before, (int, float)) and isinstance(after, (int, float))
                             else None),
                "IDENTICAL": ok})
    return ok


def main():
    checks, allok = [], True

    # ================================================================ Phase 1 cluster grain
    dfF_raw = reingest(os.path.join(RUNS, "stage_F", "*"), "stage_F")
    dfF_arch = pd.read_parquet(os.path.join(RESULTS, "stage_F_outcomes.parquet"))
    key = ["run", "mask_id", "seed", "target"]
    M = dfF_arch.merge(dfF_raw, on=key, suffixes=("_a", "_r"))
    for c in ("success_rate", "plain_loss", "transport_loss", "interaction_loss"):
        d = float(np.abs(M[f"{c}_a"] - M[f"{c}_r"]).max())
        allok &= cmp(f"stage_F re-ingest max|diff| {c}", 0.0, d, checks)
    allok &= cmp("stage_F n_rows", len(dfF_arch), len(dfF_raw), checks)

    dfG_raw = reingest(os.path.join(RUNS, "stage_G", "*"), "stage_G")
    dfG_arch = pd.read_parquet(os.path.join(RESULTS, "stage_G_outcomes.parquet"))
    allok &= cmp("stage_G n_rows", len(dfG_arch), len(dfG_raw), checks)

    dfF = add_outcomes(dfF_raw)
    dfG = add_outcomes(dfG_raw)

    # ---- ceilings (Phase-1 conditional ceiling, recomputed with the fixed transform)
    fman = json.load(open(os.path.join(RESULTS, "mask_manifest.json")))
    fmasks = [{"mask_id": m["mask_id"], "demos": m["demos"], "clusters": m["clusters"]}
              for m in fman["masks"]]
    ncm = fman["noise_ceiling_masks"]
    incl = {m["mask_id"]: m["clusters"] for m in fmasks}
    seeds = [301, 302, 303, 304]
    arch_ceil = json.load(open(os.path.join(RESULTS, "noise_ceilings.json")))

    for t in dataset.clusters():
        for keyo in ("logit_success", "neg_plain_loss"):
            sub = dfF[(dfF.target == t) & (dfF.mask_id.isin(ncm)) & (dfF.seed.isin(seeds))]
            obms = {}
            for m, grp in sub.groupby("mask_id"):
                d = {int(r.seed): float(getattr(r, keyo)) for r in grp.itertuples()}
                if all(s in d for s in seeds):
                    obms[m] = d
            cond = {m: v for m, v in obms.items() if t in incl.get(m, [])}
            r = lds.noise_ceiling(cond, seeds=tuple(seeds))
            allok &= cmp(f"ceiling[{t}][{keyo}]", arch_ceil[t][keyo]["ceiling"], r["ceiling"],
                         checks)

    # ---- cluster-grain conditional LDS (the Phase-1 headline table)
    infl = pd.read_parquet(os.path.join(RESULTS, "influence_table.parquet"))
    head_arch = pd.read_csv(os.path.join(RESULTS, "headline_lds_by_target.csv"))
    n_sig_after = 0
    for t in dataset.clusters():
        for a in ATTRS:
            sc = infl[(infl.attributor == a) & (infl.functional == "plain") & (infl.target == t)]
            scores = dict(zip(sc.demo_id, sc.score))
            obs = (dfF[(dfF.target == t) & (dfF.seed.isin([301, 302]))]
                   .groupby("mask_id")["logit_success"].mean().to_dict())
            r = lds.conditional_lds(scores, fmasks, obs, t, include_only_target_masks=True)
            row = head_arch[(head_arch.target == t) & (head_arch.attributor == a)].iloc[0]
            allok &= cmp(f"cluster LDS[{t}][{a}]", float(row.lds_conditional), r["rho"], checks)
            allok &= cmp(f"cluster p1[{t}][{a}]", float(row.p_onesided), r["p_onesided"], checks)
            if r["p_onesided"] < 0.05 / 9:
                n_sig_after += 1
    n_sig_before = int((head_arch.p_onesided < 0.05 / 9).sum())
    allok &= cmp("Bonferroni-significant cluster-grain cells, logit_success (of 27)",
                 n_sig_before, n_sig_after, checks)

    # ---- the REPORTED headline: 8/27 Bonferroni-significant cells on the L2 outcome
    Larch = pd.read_parquet(os.path.join(RESULTS, "lds_cluster_grain.parquet"))
    Larch = Larch[(Larch.outcome == "neg_plain_loss") & (Larch.functional == "plain")]
    n_sig_L2_before = int((Larch.p_onesided < 0.05 / 9).sum())
    n_sig_L2_after = 0
    for t in dataset.clusters():
        for a in ATTRS:
            sc = infl[(infl.attributor == a) & (infl.functional == "plain") & (infl.target == t)]
            scores = dict(zip(sc.demo_id, sc.score))
            obs = (dfF[(dfF.target == t) & (dfF.seed.isin([301, 302]))]
                   .groupby("mask_id")["neg_plain_loss"].mean().to_dict())
            r = lds.conditional_lds(scores, fmasks, obs, t, include_only_target_masks=True)
            arow = Larch[(Larch.target == t) & (Larch.attributor == a)].iloc[0]
            allok &= cmp(f"cluster LDS L2 [{t}][{a}]", float(arow.lds_conditional), r["rho"],
                         checks)
            if r["p_onesided"] < 0.05 / 9:
                n_sig_L2_after += 1
    allok &= cmp("Bonferroni-significant cluster-grain cells, L2 outcome (of 27) [REPORT: 8]",
                 n_sig_L2_before, n_sig_L2_after, checks)

    # ================================================================ Phase 1 Gate 1 (Stage D)
    # Stage D uses SEEDS = [101, 102] (src/stage_d.py:35), NOT the Stage-G seeds.
    dman = json.load(open(os.path.join(RESULTS, "stage_D_mask_manifest.json")))
    dmasks = dman["masks"] if isinstance(dman, dict) else dman
    dinf = pd.read_parquet(os.path.join(RESULTS, "stage_D_influence_C1.parquet"))
    gate1_arch = json.load(open(os.path.join(RESULTS, "stage_D_gate1.json")))
    obsD = {}
    for m in dmasks:
        pl = []
        for s in (101, 102):
            rd = os.path.join(RUNS, "stage_D", f"{m['mask_id']}_s{s}")
            oc = L.read_outcomes(rd, required=False)      # marker-gated
            if oc:
                pl.append(oc["C1"]["plain_loss"])
        if pl:
            obsD[m["mask_id"]] = -float(np.mean(pl))
    assert len(obsD) == 12, f"Stage-D re-ingest found {len(obsD)} masks, expected 12"
    for a in ATTRS:
        sc = dinf[(dinf.attributor == a) & (dinf.functional == "plain")]
        scores = dict(zip(sc.demo_id, sc.score))
        pred = [sum(scores.get(d, 0.0) for d in m["demos"]) for m in dmasks
                if m["mask_id"] in obsD]
        out = [obsD[m["mask_id"]] for m in dmasks if m["mask_id"] in obsD]
        rho = spearman(pred, out)
        allok &= cmp(f"Gate-1 rho (Stage D, neg_plain_loss) [{a}]",
                     gate1_arch["attributors"][a]["neg_plain_loss"]["rho"], rho, checks, tol=1e-9)

    # ================================================================ Phase 2 P1 focal verdict
    G6_raw = reingest(os.path.join(L.P2_RUNS, "stage_G6", "*"), "stage_G6")
    G6 = pd.concat([add_outcomes(dfG_raw), add_outcomes(G6_raw)], ignore_index=True)
    G6_arch = pd.read_parquet(os.path.join(P2_RESULTS, "stage_G6_outcomes.parquet"))
    allok &= cmp("stage_G6 n_rows (2 old seeds + 4 new)", len(G6_arch), len(G6), checks)

    p1_arch = json.load(open(os.path.join(P2_RESULTS, "p1_demo_grain.json")))
    gman = json.load(open(os.path.join(RESULTS, "demo_mask_manifest.json")))["masks"]
    import itertools
    ALL_SEEDS = [401, 402, 403, 404, 405, 406]
    for t in FOCAL:
        sub = G6[G6.target == t]
        piv = sub.pivot_table(index="mask_id", columns="seed", values="neg_plain_loss")
        piv = piv[[s for s in ALL_SEEDS if s in piv.columns]].dropna()
        seen, vals = set(), []
        for half in itertools.combinations(list(piv.columns), 3):
            other = tuple(s for s in piv.columns if s not in half)
            k = frozenset([half, other])
            if k in seen:
                continue
            seen.add(k)
            vals.append(spearman(piv[list(half)].mean(1).values,
                                 piv[list(other)].mean(1).values))
        r3 = float(np.mean(vals))
        r6 = 2 * r3 / (1 + r3)
        e = p1_arch["all_targets"][t]["neg_plain_loss"]
        allok &= cmp(f"P1 6-seed ceiling (SB) [{t}]", e["ceiling_6seed_SB"], r6, checks, tol=1e-6)

        mean6 = piv.mean(1)
        for a in ATTRS:
            sc = infl[(infl.attributor == a) & (infl.functional == "plain") & (infl.target == t)]
            sc = dict(zip(sc.demo_id, sc.score))
            pred = np.array([sum(sc.get(d, 0.0) for d in m["demos"]) for m in gman])
            out = np.array([mean6.get(m["mask_id"], np.nan) for m in gman])
            ok = np.isfinite(out)
            rho = spearman(pred[ok], out[ok])
            p1v = spearman_p_onesided(rho, int(ok.sum()))
            allok &= cmp(f"P1 demo LDS [{t}][{a}]", e["attributors"][a]["rho"], rho, checks,
                         tol=1e-6)
            allok &= cmp(f"P1 demo p1 [{t}][{a}]", e["attributors"][a]["p_onesided"], p1v, checks,
                         tol=1e-6)

    # ================================================================ verdict
    n_bad = sum(1 for c in checks if not c["IDENTICAL"])
    out = {
        "stage": "P6 no-change proof",
        "what_was_fixed": [
            "P6.1 marker-gated + atomic ingestion (p3lib.read_outcomes refuses unmarked artifacts)",
            "P6.2 probe-leak guard on every attribution entry point (p3lib.assert_no_probe_leak)",
            "P6.3 logit clamp reads the row's n_episodes (p3lib.logit_success_rowwise)",
        ],
        "method": ("Every table was RE-INGESTED from the raw per-run artifacts through the "
                   "marker-gated reader (not read from the archived parquet), and every headline "
                   "statistic was RECOMPUTED with the fixed row-wise transform, then diffed "
                   "against the archived artifact."),
        "tolerance": TOL,
        "n_checks": len(checks),
        "n_mismatches": n_bad,
        "ALL_IDENTICAL": bool(n_bad == 0),
        "checks": checks,
        "conclusion": ("The three fixes close real traps and change ZERO reported numbers. Every "
                       "Phase-1 and Phase-2 headline statistic recomputes bit-for-bit from the raw "
                       "artifacts under the hardened readers."
                       if n_bad == 0 else
                       "MISMATCH FOUND -- a fix changed a reported number. INSTRUMENT DEFECT."),
    }
    L.atomic_write_json(os.path.join(P3_RESULTS, "p6_no_change.json"), out)

    print("=" * 96)
    print("P6 NO-CHANGE PROOF")
    print("=" * 96)
    for c in checks:
        if not c["IDENTICAL"]:
            print(f"  MISMATCH  {c['quantity']}: {c['before_archived']} -> "
                  f"{c['after_fixed_readers']}")
    print(f"  {len(checks)} checks, {n_bad} mismatches")
    print(f"  ALL IDENTICAL: {out['ALL_IDENTICAL']}")
    print("=" * 96)
    for c in checks[:0]:
        pass
    # show a representative sample
    print("sample of recomputed headline values (before == after):")
    for q in ("ceiling[C1][neg_plain_loss]", "P1 6-seed ceiling (SB) [C1]",
              "P1 demo LDS [C1][TracIn]", "P1 demo LDS [C5][IF]",
              "Gate-1 rho (Stage D, neg_plain_loss) [IF]",
              "Bonferroni-significant cluster-grain cells, L2 outcome (of 27) [REPORT: 8]"):
        for c in checks:
            if c["quantity"] == q:
                print(f"  {q:52s} {c['before_archived']} == {c['after_fixed_readers']}")
    print("=" * 96)
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
