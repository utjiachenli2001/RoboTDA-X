"""STAGE H -- attribution analysis: LDS vs noise ceiling, headline stats, RQ3, moderators.

Every number here is computed from artifact files written by actual runs:
  results/stage_F_outcomes.parquet   (168 cluster-grain retrains)
  results/stage_G_outcomes.parquet   (48 demo-grain retrains)
  results/influence_table.parquet    (+ _per_member for jackknife CIs)
  results/mask_manifest.json, demo_mask_manifest.json, similarity.npz

Outcome convention: all outcomes are ORIENTED SO THAT HIGHER = BETTER, so a faithful
attributor (positive score = "this demo helps the target") should give a POSITIVE Spearman.
  success  -> logit(clamp(p, 1/60, 59/60))
  losses   -> NEGATIVE loss ("utility")
"""
import os
import sys
import json
import itertools
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bootstrap  # noqa: F401
from bootstrap import RESULTS
import dataset
import masks as MK
import lds

F_SEEDS = [301, 302]
F_NC_SEEDS = [303, 304]
G_SEEDS = [401, 402]
N_EPISODES = 30                 # 3 probe tasks x 10 rollouts
ATTRS = ["TracIn", "TRAK", "IF"]
FUNCS = ["plain", "transport", "interaction"]
OUTCOMES = ["logit_success", "neg_plain_loss", "neg_transport_loss", "neg_interaction_loss"]
BONF_ALPHA = 0.05 / 9           # 9 targets -> 0.00556
FOCAL = ["C1", "C5"]
BONF2_ALPHA = 0.05 / 2          # focal demo-grain, one-sided


# ---------------------------------------------------------------- outcomes
def outcome_value(row, key):
    if key == "logit_success":
        return float(lds.logit_success(row["success_rate"], N_EPISODES))
    return -float(row[key.replace("neg_", "")])


def outcome_table(parquet):
    df = pd.read_parquet(parquet)
    for k in OUTCOMES:
        df[k] = df.apply(lambda r: outcome_value(r, k), axis=1)
    return df


def seedmean_outcomes(df, target, key, seeds):
    """{mask_id: mean outcome over the given seeds} for one target."""
    sub = df[(df.target == target) & (df.seed.isin(seeds))]
    g = sub.groupby("mask_id")[key].mean()
    return {m: float(v) for m, v in g.items()}


# ---------------------------------------------------------------- noise ceiling
def ceilings(dfF, nc_masks, fmasks):
    """Per target, per outcome: the noise ceiling from the 12 replicate masks x 4 seeds.

    TWO ceilings are computed and BOTH are reported:

      ceiling_all12       -- over all 12 replicate masks (the spec's literal instruction).
      ceiling_conditional -- over ONLY the target-INCLUDED subset of those 12 (n = 6-8 by the
                             stratified design).  <-- this is the one overlaid on the primary
                             (conditional) LDS.

    Why the distinction matters: a mask that EXCLUDES the target produces ~0 success for every
    seed, so it is trivially reproducible across seeds. Mixing those masks into the ceiling
    inflates it with the include/exclude split -- a split the conditional LDS is never allowed
    to exploit, because it only ever sees target-included masks. Judging a conditional LDS
    against an all-12 ceiling would therefore compare against an unattainable bar.
    """
    incl = {m["mask_id"]: m["clusters"] for m in fmasks}
    out = {}
    seeds = F_SEEDS + F_NC_SEEDS
    for t in dataset.clusters():
        out[t] = {}
        for key in OUTCOMES:
            sub = dfF[(dfF.target == t) & (dfF.mask_id.isin(nc_masks)) & (dfF.seed.isin(seeds))]
            obms = {}
            for m, grp in sub.groupby("mask_id"):
                d = {int(r.seed): float(getattr(r, key)) for r in grp.itertuples()}
                if all(s in d for s in seeds):
                    obms[m] = d
            cond = {m: v for m, v in obms.items() if t in incl.get(m, [])}
            all12 = lds.noise_ceiling(obms, seeds=tuple(seeds))
            conditional = lds.noise_ceiling(cond, seeds=tuple(seeds))
            out[t][key] = {
                "ceiling": conditional["ceiling"],          # PRIMARY comparator
                "ceiling_sb": conditional["ceiling_sb"],
                "ci95": conditional["ci95"],
                "n_masks": conditional["n_masks"],
                "per_pairing": conditional["per_pairing"],
                "ceiling_all12_INFLATED_BY_EXCLUSION": all12["ceiling"],
                "n_masks_all12": all12["n_masks"],
            }
    return out


# ---------------------------------------------------------------- LDS
def cluster_grain_lds(dfF, infl, fmasks):
    rows = []
    for t in dataset.clusters():
        for attr in ATTRS:
            for func in FUNCS:
                sc = infl[(infl.attributor == attr) & (infl.functional == func)
                          & (infl.target == t)]
                scores = dict(zip(sc.demo_id, sc.score))
                if not scores:
                    continue
                for key in OUTCOMES:
                    obs = seedmean_outcomes(dfF, t, key, F_SEEDS)
                    cond = lds.conditional_lds(scores, fmasks, obs, t,
                                               include_only_target_masks=True)
                    full = lds.conditional_lds(scores, fmasks, obs, t,
                                               include_only_target_masks=False)
                    rows.append({
                        "grain": "cluster", "target": t, "attributor": attr,
                        "functional": func, "outcome": key,
                        "lds_conditional": cond["rho"], "n_cond": cond["n_masks"],
                        "p_onesided": cond["p_onesided"],
                        "ci_lo": cond["ci95"][0], "ci_hi": cond["ci95"][1],
                        "lds_full72_INFLATED": full["rho"], "n_full": full["n_masks"],
                    })
    return pd.DataFrame(rows)


def demo_grain_lds(dfG, infl, gmasks):
    from scipy import stats as st
    _, by_c = dataset.train_pool()
    rows = []
    for t in dataset.clusters():
        in_t = set(by_c[t])
        n_in = {m["mask_id"]: sum(1 for d in m["demos"] if d in in_t) for m in gmasks}
        for attr in ATTRS:
            for func in FUNCS:
                sc = infl[(infl.attributor == attr) & (infl.functional == func)
                          & (infl.target == t)]
                scores = dict(zip(sc.demo_id, sc.score))
                if not scores:
                    continue
                for key in OUTCOMES:
                    obs = seedmean_outcomes(dfG, t, key, G_SEEDS)
                    r = lds.conditional_lds(scores, gmasks, obs, t,
                                            include_only_target_masks=False)
                    # partial Spearman controlling for in-target demo count (7 vs 8)
                    ids = r["mask_ids"]
                    if len(ids) >= 5:
                        x = st.rankdata(r["pred"])
                        y = st.rankdata(r["outcome"])
                        z = st.rankdata([n_in[m] for m in ids])
                        def resid(v):
                            A = np.vstack([np.ones_like(z), z]).T
                            b = np.linalg.lstsq(A, v, rcond=None)[0]
                            return v - A @ b
                        rp = lds.spearman(resid(x), resid(y))
                    else:
                        rp = np.nan
                    rows.append({
                        "grain": "demo", "target": t, "attributor": attr, "functional": func,
                        "outcome": key, "lds": r["rho"], "n_masks": r["n_masks"],
                        "p_onesided": r["p_onesided"],
                        "ci_lo": r["ci95"][0], "ci_hi": r["ci95"][1],
                        "partial_rho_ctrl_in_target_count": rp,
                        "focal": t in FOCAL,
                    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- headline stats
def intrusion(scores, target, by_c):
    """(a) fraction of the 120 outsiders above the median insider, and above the p75 insider."""
    ins = np.array([scores[d] for d in by_c[target] if d in scores])
    out = np.array([scores[d] for c in by_c if c != target for d in by_c[c] if d in scores])
    if len(ins) == 0 or len(out) == 0:
        return {}
    med, p75 = float(np.median(ins)), float(np.percentile(ins, 75))
    # insider-advantage AUC = P(random insider outranks a random outsider)
    auc = float((out[None, :] < ins[:, None]).mean() + 0.5 * (out[None, :] == ins[:, None]).mean())
    return {
        "n_insiders": int(len(ins)), "n_outsiders": int(len(out)),
        "intrusion_above_median": float((out > med).mean()),
        "intrusion_above_p75": float((out > p75).mean()),
        "insider_advantage_auc": auc,
        "any_outsider_above_p75": bool((out > p75).any()),
    }


def headline(infl, per_member, best_attr_by_target, func="plain"):
    _, by_c = dataset.train_pool()
    res = {}
    for t in dataset.clusters():
        attr = best_attr_by_target.get(t)
        if attr is None:
            continue
        sc = infl[(infl.attributor == attr) & (infl.functional == func) & (infl.target == t)]
        scores = dict(zip(sc.demo_id, sc.score))
        base = intrusion(scores, t, by_c)

        # top-15 composition
        top = sc.nlargest(15, "score")
        n_out = int((top.cluster_of_demo != t).sum())
        base["top15_outsiders"] = n_out
        base["top15_outsider_clusters"] = (top[top.cluster_of_demo != t]
                                           .cluster_of_demo.value_counts().to_dict())
        base["top15_demo_ids"] = list(top.demo_id)

        # delete-1 jackknife over the E=10 ensemble members
        pm = per_member[(per_member.attributor == attr) & (per_member.functional == func)
                        & (per_member.target == t)]
        members = sorted(pm.member.unique())
        jk = {"intrusion_above_median": [], "intrusion_above_p75": [],
              "insider_advantage_auc": [], "top15_outsiders": []}
        for m in members:
            sub = pm[pm.member != m].groupby("demo_id")["score"].mean()
            s = dict(sub)
            r = intrusion(s, t, by_c)
            for k in ("intrusion_above_median", "intrusion_above_p75", "insider_advantage_auc"):
                jk[k].append(r[k])
            tt = sub.nlargest(15)
            jk["top15_outsiders"].append(
                int(sum(1 for d in tt.index
                        if dataset.cluster_of_task()[dataset.parse_did(d)[1]] != t)))
        n = len(members)
        for k, v in jk.items():
            v = np.array(v, dtype=float)
            se = float(np.sqrt((n - 1) / n * ((v - v.mean()) ** 2).sum())) if n > 1 else np.nan
            base[f"{k}_jackknife_se"] = se
            base[f"{k}_jackknife_range"] = [float(v.min()), float(v.max())]
        base["attributor_used"] = attr
        base["n_ensemble_members"] = n
        res[t] = base
    return res


def transfer_matrix(infl, best_attr_by_target, func="plain"):
    """9x9: mean influence of cluster-i demos toward target j."""
    cl = dataset.clusters()
    M = np.full((9, 9), np.nan)
    for j, t in enumerate(cl):
        attr = best_attr_by_target.get(t)
        if attr is None:
            continue
        sc = infl[(infl.attributor == attr) & (infl.functional == func) & (infl.target == t)]
        for i, c in enumerate(cl):
            v = sc[sc.cluster_of_demo == c]["score"]
            if len(v):
                M[i, j] = float(v.mean())
    return M, cl


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate1_pass", default=None,
                    help="override; default reads results/stage_D_gate1.json")
    a = ap.parse_args()

    fman = MK.cluster_mask_manifest()
    gman = MK.demo_mask_manifest()
    fmasks = [{"mask_id": m["mask_id"], "demos": m["demos"], "clusters": m["clusters"]}
              for m in fman["masks"]]
    gmasks = [{"mask_id": m["mask_id"], "demos": m["demos"]} for m in gman["masks"]]
    nc = fman["noise_ceiling_masks"]

    dfF = outcome_table(os.path.join(RESULTS, "stage_F_outcomes.parquet"))
    infl = pd.read_parquet(os.path.join(RESULTS, "influence_table.parquet"))
    per_member = pd.read_parquet(os.path.join(RESULTS, "influence_table_per_member.parquet"))

    # ---- noise ceilings
    ceil = ceilings(dfF, nc, fmasks)
    json.dump(ceil, open(os.path.join(RESULTS, "noise_ceilings.json"), "w"), indent=1,
              default=float)

    # ---- cluster-grain LDS
    L = cluster_grain_lds(dfF, infl, fmasks)
    L.to_parquet(os.path.join(RESULTS, "lds_cluster_grain.parquet"), index=False)

    # headline row set: outcome = logit_success, functional = plain
    head = L[(L.outcome == "logit_success") & (L.functional == "plain")].copy()
    head["ceiling"] = head.target.map(lambda t: ceil[t]["logit_success"]["ceiling"])
    head["bonferroni_sig"] = head.p_onesided < BONF_ALPHA
    head.to_csv(os.path.join(RESULTS, "headline_lds_by_target.csv"), index=False)

    # ---- best attributor per target.
    # Selected on the conditional LDS against the HELD-OUT L2 outcome, not against success.
    # Reason (measured, not assumed): the conditional noise ceiling of the SUCCESS outcome is
    # low and sometimes negative (C2 0.078, C7 0.117, C8 -0.095) -- i.e. even an oracle cannot
    # predict it -- so ranking attributors by their success LDS would be ranking them on noise.
    # The L2 outcome's ceilings are far higher (up to 0.84). Both rankings are saved.
    # NB Gate 1 FAILED, so NO attributor is "LDS-validated"; this is the best-performing one,
    # and every statistic derived from it inherits that caveat.
    lossrows = L[(L.outcome == "neg_plain_loss") & (L.functional == "plain")]
    best, best_by_success = {}, {}
    for t in dataset.clusters():
        sl = lossrows[lossrows.target == t].dropna(subset=["lds_conditional"])
        if len(sl):
            best[t] = str(sl.loc[sl.lds_conditional.idxmax(), "attributor"])
        ss = head[head.target == t].dropna(subset=["lds_conditional"])
        if len(ss):
            best_by_success[t] = str(ss.loc[ss.lds_conditional.idxmax(), "attributor"])
    json.dump(best, open(os.path.join(RESULTS, "best_attributor_by_target.json"), "w"), indent=1)
    json.dump({"selected_on": "conditional LDS vs held-out L2 (success ceilings are too low "
                              "to rank on)", "by_loss": best, "by_success": best_by_success,
               "gate1_passed": False},
              open(os.path.join(RESULTS, "best_attributor_selection.json"), "w"), indent=1)

    # ---- headline statistics with jackknife CIs
    H = headline(infl, per_member, best, func="plain")
    json.dump(H, open(os.path.join(RESULTS, "headline_stats.json"), "w"), indent=1, default=float)

    M, cl = transfer_matrix(infl, best, func="plain")
    np.savez(os.path.join(RESULTS, "transfer_matrix.npz"), M=M, clusters=np.array(cl))

    # ---- demo grain (Stage G)
    gpath = os.path.join(RESULTS, "stage_G_outcomes.parquet")
    if os.path.exists(gpath):
        dfG = outcome_table(gpath)
        DG = demo_grain_lds(dfG, infl, gmasks)
        DG.to_parquet(os.path.join(RESULTS, "lds_demo_grain.parquet"), index=False)
        focal = DG[(DG.outcome == "logit_success") & (DG.functional == "plain")
                   & (DG.target.isin(FOCAL))]
        focal_pass = {}
        for t in FOCAL:
            sub = focal[focal.target == t].dropna(subset=["lds"])
            focal_pass[t] = bool((sub.p_onesided < BONF2_ALPHA).any()) if len(sub) else False
        dg_verdict = {"focal_targets": FOCAL, "focal_pass": focal_pass,
                      "both_focal_failed": not any(focal_pass.values()),
                      "downgrade_rule": "if BOTH focal targets fail, all per-demo claims "
                                        "downgrade to cluster grain"}
        json.dump(dg_verdict, open(os.path.join(RESULTS, "demo_grain_verdict.json"), "w"),
                  indent=1)
    else:
        dg_verdict = {"note": "Stage G did not run"}

    # ---- RQ3 phase contrast (per target: LDS + intrusion under each functional)
    _, by_c = dataset.train_pool()
    rq3 = []
    for t in dataset.clusters():
        attr = best.get(t)
        if attr is None:
            continue
        for func in FUNCS:
            sc = infl[(infl.attributor == attr) & (infl.functional == func) & (infl.target == t)]
            scores = dict(zip(sc.demo_id, sc.score))
            r = L[(L.target == t) & (L.attributor == attr) & (L.functional == func)
                  & (L.outcome == "logit_success")]
            intr = intrusion(scores, t, by_c)
            rq3.append({"target": t, "attributor": attr, "functional": func,
                        "lds_conditional_success": float(r.lds_conditional.iloc[0]) if len(r) else np.nan,
                        **{k: intr.get(k) for k in ("intrusion_above_median",
                                                    "intrusion_above_p75",
                                                    "insider_advantage_auc")}})
    pd.DataFrame(rq3).to_csv(os.path.join(RESULTS, "rq3_phase_contrast.csv"), index=False)

    # ---- print summary
    print("\n" + "=" * 78)
    print("STAGE H -- CLUSTER-GRAIN CONDITIONAL LDS (outcome = logit success, functional = plain)")
    print("=" * 78)
    print(f"{'target':>6} {'attr':>7} {'LDS':>8} {'95% CI':>16} {'p':>9} {'ceiling':>8} "
          f"{'sig(Bonf)':>10}")
    for t in dataset.clusters():
        sub = head[head.target == t]
        for r in sub.itertuples():
            print(f"{r.target:>6} {r.attributor:>7} {r.lds_conditional:>+8.3f} "
                  f"[{r.ci_lo:>+5.2f},{r.ci_hi:>+5.2f}] {r.p_onesided:>9.4f} "
                  f"{r.ceiling:>8.3f} {str(r.bonferroni_sig):>10}")
    print(f"\nBonferroni alpha = {BONF_ALPHA:.5f} (9 targets, one-sided)")
    print(f"best attributor per target: {best}")
    print(f"\nheadline intrusion (best attributor, plain functional):")
    for t, v in H.items():
        print(f"  {t}: above-median {v['intrusion_above_median']:.3f} "
              f"(jk SE {v['intrusion_above_median_jackknife_se']:.3f}), "
              f"above-p75 {v['intrusion_above_p75']:.3f}, "
              f"top15 outsiders {v['top15_outsiders']}/15, AUC {v['insider_advantage_auc']:.3f}")
    print(f"\ndemo-grain verdict: {dg_verdict}")


if __name__ == "__main__":
    main()
