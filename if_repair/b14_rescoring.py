"""W4 -- Phi-hygiene rescoring family.

The one pass-4 idea that is evaluable against the TRUE cached-Gram baseline (0.593 on C1) for
ZERO GPU: RelatIF. Every gradient estimator so far normalizes a member's scores by a single
scalar (mean diag G_m, dmean; or a per-column unit-L2). RelatIF instead divides each demo's
raw dot product K[d,t] by that demo's OWN self-influence -- a THETA-relative influence:

    relatif_sqrt   score[d,t] = K_m[d,t] / sqrt(G_m[d,d])       (Barshan et al. 2020, exact)
    relatif_lin    score[d,t] = K_m[d,t] / G_m[d,d]             (full self-influence norm)

both aggregated over members. This is a distinct normalization from the two archived ones
(dmean = global-scalar; unitL2 = per-column L2), so it is a genuine new estimator, and it is
the ONLY one runnable on the cached E=20 Gram. Everything else in W4 (per-query aggregation,
frame-robust demo gradients, sigma-clamped NLL gradients) needs the regenerated E=5 gradients
and is deferred to the regen arm.

Baseline is recomputed here through build_scores so the 0.593 anchor is reproduced in the same
run and RelatIF is judged against it, never against the paper's 0.513 (BLOCKERS #1).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402
from if_repair.eval import evaluate, build_scores  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

D.add_repo_paths()
from lds import spearman, spearman_p_onesided  # noqa: E402


def relatif_scores(Z, power, aggregate="mean_over_members", eps=1e-30):
    """RelatIF on a cached (or regenerated) Gram.

    power: 0.5 -> divide by sqrt(G_dd) ; 1.0 -> divide by G_dd.
    aggregate:
      mean_over_members   score[d,t] = mean_m K_m[d,t] / G_m[d,d]**power
      unitl2_then_mean    per-member column unit-L2 normalized before the mean (matches the
                          paper's champion aggregation, applied to the RelatIF scores)
    -> {target: {demo_id: score}}
    """
    G, K = np.asarray(Z["G"], float), np.asarray(Z["K"], float)   # (M,N,N),(M,N,T)
    M, N, T = K.shape
    tids, tgts = list(Z["train_ids"]), list(Z["targets"])
    S = np.empty((M, N, T))
    for m in range(M):
        gdd = np.clip(np.diagonal(G[m]), eps, None)               # (N,)
        denom = gdd ** power
        S[m] = K[m] / denom[:, None]
    if aggregate == "unitl2_then_mean":
        nrm = np.linalg.norm(S, axis=1, keepdims=True)            # per (member,target) column
        S = S / np.clip(nrm, eps, None)
    Sag = S.mean(axis=0)                                          # (N,T)
    return {tgts[j]: {tids[i]: float(Sag[i, j]) for i in range(N)}
            for j in range(T)}


def _draw(draw):
    """-> (masks[{mask_id,demos}], campaign) for G/H/I. Same-construction campaign outcomes."""
    from if_repair import retrain as RT
    if draw == "G":
        return [{"mask_id": m["mask_id"], "demos": m["demos"]} for m in D.demo_masks()], "A"
    if draw == "H":
        return [{"mask_id": m["mask_id"], "demos": m["demos"]}
                for m in RT.fresh_demo_masks(seed=RT.FRESH_MASK_SEED, prefix="H")[0]], "B"
    if draw == "I":
        return [{"mask_id": m["mask_id"], "demos": m["demos"]}
                for m in RT.fresh_demo_masks(seed=RT.FRESH_MASK_SEED_I, prefix="I")[0]], "I"
    raise KeyError(draw)


def _mask_pred(scores, masks):
    return np.array([sum(scores.get(d, 0.0) for d in m["demos"]) for m in masks])


def paired_bootstrap(px, pg, out, n_boot=5000, seed=0):
    """One-sided: is rho(X)-rho(GradDot) > 0 on these masks? Shared-mask draw cancels."""
    rng = np.random.default_rng(seed)
    px, pg, out = map(lambda a: np.asarray(a, float), (px, pg, out))
    n = len(out)
    d0 = spearman(px, out) - spearman(pg, out)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        rx, rg = spearman(px[i], out[i]), spearman(pg[i], out[i])
        if np.isfinite(rx) and np.isfinite(rg):
            diffs.append(rx - rg)
    diffs = np.array(diffs)
    return d0, (float((diffs <= 0).mean()) if len(diffs) else np.nan)


def paired_across_draws(Z, targets):
    """The out-of-sample test of the RelatIF C5 win: cached-Gram RelatIF vs cached-Gram
    GradDot_dmean, evaluated PAIRED on each of G/H/I and pooled. Both estimators come from the
    SAME E=20 cached Gram (self-consistent; BLOCKERS #6 is about mixing regen with cached, not
    about reusing outcomes -- BLOCKERS #12). Outcomes are the campaign per-frame plain tables."""
    from if_repair import functionals as F
    xsc = relatif_scores(Z, 1.0, aggregate="unitl2_then_mean")     # the best dev cell
    gsc = build_scores({"kind": "GradDot", "normalize": "dmean", "aggregator": "mean"}, "bc_s10")
    rows = []
    pooled = {t: {"px": [], "pg": [], "out": []} for t in targets}
    for draw in ("G", "H", "I"):
        masks, campaign = _draw(draw)
        for t in targets:
            raw = F.campaign_outcomes(campaign, "plain", targets=(t,))[t]
            ceil = F.split_half_ceiling(raw)["ceiling"]
            obs = F.seed_mean(raw)
            out = np.array([obs.get(m["mask_id"], np.nan) for m in masks])
            ok = np.isfinite(out)
            px, pg = _mask_pred(xsc[t], masks)[ok], _mask_pred(gsc[t], masks)[ok]
            o = out[ok]
            rx, rg = spearman(px, o), spearman(pg, o)
            d0, pp = paired_bootstrap(px, pg, o)
            rows.append({"draw": draw, "target": t, "n": int(ok.sum()), "ceiling": ceil,
                         "relatif_lds": rx, "graddot_lds": rg,
                         "relatif_ratio": rx / ceil, "graddot_ratio": rg / ceil,
                         "paired_delta_rho": d0, "paired_p": pp})
            pooled[t]["px"] += list(px); pooled[t]["pg"] += list(pg); pooled[t]["out"] += list(o)
    for t in targets:
        px = np.array(pooled[t]["px"]); pg = np.array(pooled[t]["pg"])
        o = np.array(pooled[t]["out"])
        rx, rg = spearman(px, o), spearman(pg, o)
        d0, pp = paired_bootstrap(px, pg, o)
        rows.append({"draw": "POOLED_GHI", "target": t, "n": len(o), "ceiling": np.nan,
                     "relatif_lds": rx, "graddot_lds": rg, "relatif_ratio": np.nan,
                     "graddot_ratio": np.nan, "paired_delta_rho": d0, "paired_p": pp})
    return pd.DataFrame(rows)


def main():
    tier = "bc_s10"
    Z = D.cache_for(tier)
    targets = list(D.DEV_TARGETS)     # C1, C5 (dev)

    # ---- baselines recomputed on THIS cache (reproduce the anchors)
    rows = []
    for name, spec in [
        ("GradDot_dmean", {"kind": "GradDot", "normalize": "dmean", "aggregator": "mean"}),
        ("GradDot_unitL2", {"kind": "GradDot", "normalize": "unitL2", "aggregator": "mean"}),
    ]:
        sc = build_scores(spec, tier)
        for t in targets:
            r = evaluate(sc[t], t, tier)
            r["estimator"] = name
            rows.append(r)

    # ---- RelatIF variants
    for power, pname in [(0.5, "relatif_sqrt"), (1.0, "relatif_lin")]:
        for agg in ("mean_over_members", "unitl2_then_mean"):
            sc = relatif_scores(Z, power, aggregate=agg)
            for t in targets:
                r = evaluate(sc[t], t, tier)
                r["estimator"] = f"{pname}/{agg}"
                rows.append(r)

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "b14_relatif_cached.csv"), index=False)

    base = {t: df[(df.estimator == "GradDot_dmean") & (df.target == t)].lds.iloc[0]
            for t in targets}
    print("=" * 96)
    print("W4 -- RelatIF on the CACHED E=20 Gram (tier bc_s10, demo grain n=24). "
          "Bar = GradDot_dmean.")
    print("=" * 96)
    show = df[["estimator", "target", "lds", "ceiling", "ratio", "p", "passed"]].copy()
    show["d_vs_dmean"] = [df.iloc[i].lds - base[df.iloc[i].target] for i in range(len(df))]
    print(show.round(4).to_string(index=False))
    print(f"\nGradDot_dmean anchors reproduced: "
          f"C1={base.get('C1', float('nan')):.4f} (expect 0.5930), "
          f"C5={base.get('C5', float('nan')):.4f} (expect 0.3904)")
    print("\nRelatIF beats GradDot_dmean on:")
    for t in targets:
        b = base[t]
        winners = df[(df.target == t) & (df.estimator.str.startswith("relatif")) & (df.lds > b)]
        print(f"  {t}: " + (", ".join(f"{w.estimator} ({w.lds:.4f}, +{w.lds-b:.4f})"
                                      for _, w in winners.iterrows()) or "none"))

    # ---- the out-of-sample test: relatif_lin/unitL2 vs GradDot paired on G/H/I
    print("\n" + "=" * 96)
    print("W4 -- best cell (relatif_lin/unitl2) PAIRED vs GradDot_dmean on each mask draw "
          "(cached E=20 Gram)")
    print("=" * 96)
    pdf = paired_across_draws(Z, targets)
    pdf.to_csv(os.path.join(RESULTS, "b14_relatif_paired_ghi.csv"), index=False)
    print(pdf.round(4).to_string(index=False))
    print("\nKILL-RULE / prereg check (W4 needs paired Delta >= +0.15 pooled to enter PREREG_J):")
    for t in targets:
        r = pdf[(pdf.draw == "POOLED_GHI") & (pdf.target == t)].iloc[0]
        verdict = "CANDIDATE" if (r.paired_delta_rho >= 0.15 and r.paired_p < 0.05) else "no"
        print(f"  {t}: pooled paired Delta_rho = {r.paired_delta_rho:+.4f}  "
              f"(p={r.paired_p:.4f})  -> {verdict}")


if __name__ == "__main__":
    main()
