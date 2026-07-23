"""Figures (spec §10). Every figure saves the data behind it to results/.

Design rules (see vizstyle.py):
  * the QUANTITY figure plots two measures of different scale (margin in success points, and
    intrusion rate in [0,1]) -> TWO PANELS, never a dual y-axis;
  * the transfer matrix is a SIGNED quantity -> diverging blue<->red with a gray midpoint;
  * insider/outsider identity uses two fixed categorical hues everywhere;
  * the noise ceiling is a REFERENCE LINE, not a series.
"""
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bootstrap  # noqa: F401
from bootstrap import RESULTS, FIGURES
import vizstyle as V
import dataset

V.apply()
import matplotlib.pyplot as plt

BONF = 0.05 / 9


def _save(fig, name):
    p = os.path.join(FIGURES, name)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {p}")


# ---------------------------------------------------------------- 1. transfer matrix
def fig_transfer_matrix():
    z = np.load(os.path.join(RESULTS, "transfer_matrix.npz"), allow_pickle=True)
    M, cl = z["M"], [str(c) for c in z["clusters"]]
    head = pd.read_csv(os.path.join(RESULTS, "headline_lds_by_target.csv"))
    best = json.load(open(os.path.join(RESULTS, "best_attributor_by_target.json")))
    lds_of = {}
    for t in cl:
        r = head[(head.target == t) & (head.attributor == best.get(t))]
        lds_of[t] = float(r.lds_conditional.iloc[0]) if len(r) else np.nan

    # per-column z-score so columns (targets) are comparable; raw values are in the npz
    Z = np.full_like(M, np.nan, dtype=float)
    for j in range(M.shape[1]):
        col = M[:, j]
        if np.isfinite(col).sum() > 1 and np.nanstd(col) > 0:
            Z[:, j] = (col - np.nanmean(col)) / np.nanstd(col)
    v = np.nanmax(np.abs(Z)) if np.isfinite(Z).any() else 1.0

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    im = ax.imshow(Z, cmap=V.DIV, vmin=-v, vmax=v)
    ax.set_xticks(range(9))
    ax.set_yticks(range(9))
    ax.set_xticklabels([f"{t}\nLDS {lds_of[t]:+.2f}" if np.isfinite(lds_of[t]) else f"{t}\nLDS n/a"
                        for t in cl], fontsize=7)
    ax.set_yticklabels(cl)
    ax.set_xlabel("target cluster  (conditional LDS badge = is this column trustworthy?)")
    ax.set_ylabel("source cluster of the demos")
    ax.set_title("Mean per-demo influence of source cluster $i$ toward target $j$\n"
                 "(column z-scored; signed → diverging scale)", loc="left")
    ax.grid(False)
    for i in range(9):
        for j in range(9):
            if np.isfinite(Z[i, j]):
                ax.text(j, i, f"{Z[i,j]:+.1f}", ha="center", va="center", fontsize=6,
                        color=V.INK if abs(Z[i, j]) < 0.6 * v else V.SURFACE)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("influence (column z-score)", fontsize=8)
    cb.outline.set_visible(False)
    _save(fig, "fig1_transfer_matrix.png")
    pd.DataFrame(M, index=cl, columns=cl).to_csv(os.path.join(RESULTS, "fig1_transfer_matrix.csv"))


# ---------------------------------------------------------------- 2. insider/outsider dists
def fig_insider_outsider():
    infl = pd.read_parquet(os.path.join(RESULTS, "influence_table.parquet"))
    best = json.load(open(os.path.join(RESULTS, "best_attributor_by_target.json")))
    H = json.load(open(os.path.join(RESULTS, "headline_stats.json")))
    dgv = json.load(open(os.path.join(RESULTS, "demo_grain_verdict.json"))) \
        if os.path.exists(os.path.join(RESULTS, "demo_grain_verdict.json")) else {}
    focal_ok = dgv.get("focal_pass", {})
    cl = dataset.clusters()

    fig, axes = plt.subplots(3, 3, figsize=(12, 8.5), sharey=False)
    rows = []
    for k, t in enumerate(cl):
        ax = axes[k // 3][k % 3]
        attr = best.get(t)
        sub = infl[(infl.attributor == attr) & (infl.functional == "plain") & (infl.target == t)]
        ins = sub[sub.cluster_of_demo == t].score.values
        out = sub[sub.cluster_of_demo != t].score.values
        rng = np.random.default_rng(0)
        for x, vals, col, lab in ((0, ins, V.INSIDER, "insider"),
                                  (1, out, V.OUTSIDER, "outsider")):
            if len(vals) == 0:
                continue
            ax.scatter(x + rng.uniform(-0.13, 0.13, len(vals)), vals, s=11, color=col,
                       alpha=0.75, linewidths=0.5, edgecolors=V.SURFACE,
                       label=lab if k == 0 else None)
            ax.hlines(np.median(vals), x - 0.26, x + 0.26, color=V.INK2, linewidth=1.6)
        h = H.get(t, {})
        ir = h.get("intrusion_above_median", np.nan)
        se = h.get("intrusion_above_median_jackknife_se", np.nan)
        # grain licence: demo-grain claims are only licensed on focal targets that passed
        lic = "demo-grain licensed" if focal_ok.get(t) else "cluster-grain only"
        ax.set_title(f"{t} · {attr} · intrusion {ir:.2f} ± {se:.2f}\n[{lic}]", fontsize=8,
                     loc="left")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["insider", "outsider"])
        ax.axhline(0, color=V.MUTED, linewidth=0.6, linestyle=":")
        V.despine(ax)
        rows.append({"target": t, "attributor": attr, "intrusion_above_median": ir,
                     "jackknife_se": se, "licence": lic,
                     "median_insider": float(np.median(ins)) if len(ins) else np.nan,
                     "median_outsider": float(np.median(out)) if len(out) else np.nan})
    axes[0][0].set_ylabel("per-demo influence toward the target")
    fig.legend(loc="upper right", ncol=2, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Insider vs outsider influence per target (best LDS-validated attributor, "
                 "plain functional)\nintrusion = fraction of the 120 outsiders scoring above "
                 "the median insider; ± = delete-1 jackknife SE over the E=10 ensemble",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "fig2_insider_outsider.png")
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "fig2_intrusion.csv"), index=False)


# ---------------------------------------------------------------- 3. LDS vs ceiling
def fig_lds_vs_ceiling():
    L = pd.read_parquet(os.path.join(RESULTS, "lds_cluster_grain.parquet"))
    ceil = json.load(open(os.path.join(RESULTS, "noise_ceilings.json")))
    cl = dataset.clusters()
    probes = [("logit_success", "success (logit)"), ("neg_plain_loss", "held-out loss (utility)")]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True)
    rows = []
    for pi, (key, lab) in enumerate(probes):
        ax = axes[pi]
        sub = L[(L.outcome == key) & (L.functional == "plain")]
        attrs = ["TracIn", "TRAK", "IF"]
        w = 0.26
        for ai, attr in enumerate(attrs):
            vals, los, his = [], [], []
            for t in cl:
                r = sub[(sub.target == t) & (sub.attributor == attr)]
                v = float(r.lds_conditional.iloc[0]) if len(r) else np.nan
                lo = float(r.ci_lo.iloc[0]) if len(r) else np.nan
                hi = float(r.ci_hi.iloc[0]) if len(r) else np.nan
                vals.append(v)
                los.append(v - lo if np.isfinite(lo) else 0)
                his.append(hi - v if np.isfinite(hi) else 0)
                if len(r):
                    rows.append({"outcome": key, "target": t, "attributor": attr,
                                 "lds": v, "ci_lo": lo, "ci_hi": hi,
                                 "p_onesided": float(r.p_onesided.iloc[0]),
                                 "ceiling": ceil[t][key]["ceiling"]})
            x = np.arange(9) + (ai - 1) * w
            ax.bar(x, vals, w * 0.92, color=V.CAT[ai], label=attr, linewidth=0)
            ax.errorbar(x, vals, yerr=[np.abs(los), np.abs(his)], fmt="none",
                        ecolor=V.INK2, elinewidth=0.8, capsize=1.5)
        # noise ceiling = a REFERENCE LINE per target, not a series
        for i, t in enumerate(cl):
            c = ceil[t][key]["ceiling"]
            if np.isfinite(c):
                ax.hlines(c, i - 0.42, i + 0.42, color=V.INK, linewidth=1.8, linestyle="--",
                          zorder=5, label="noise ceiling" if (i == 0 and pi == 0) else None)
        ax.axhline(0, color=V.MUTED, linewidth=0.8)
        ax.set_xticks(range(9))
        ax.set_xticklabels(cl)
        ax.set_title(f"outcome: {lab}", loc="left")
        ax.set_xlabel("target cluster")
        V.despine(ax)
    axes[0].set_ylabel("conditional LDS (Spearman, 40 target-included masks)")
    axes[0].legend(loc="best", ncol=2)
    fig.suptitle("Conditional LDS vs its noise ceiling (judge against the ceiling, never 1.0)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, "fig3_lds_vs_ceiling.png")
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "fig3_lds_vs_ceiling.csv"), index=False)


# ---------------------------------------------------------------- 4. quantity (TWO PANELS)
def fig_quantity():
    q = json.load(open(os.path.join(RESULTS, "stage_C_quantity.json")))
    Qs = q["Q_values"]
    marg = [q["by_Q"][str(x)]["margin_pts_mean"] for x in Qs]
    msd = [q["by_Q"][str(x)]["margin_pts_sd"] or 0 for x in Qs]
    ip = os.path.join(RESULTS, "stage_C_intrusion.json")
    intr = json.load(open(ip)) if os.path.exists(ip) else {}

    # margin (success points) and intrusion (a rate) are DIFFERENT SCALES -> two panels.
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    ax = axes[0]
    ax.errorbar(Qs, marg, yerr=msd, marker="o", markersize=7, color=V.BLUE, linewidth=2,
                capsize=3, ecolor=V.INK2, elinewidth=0.9)
    ax.axhline(0, color=V.MUTED, linewidth=0.8, linestyle=":")
    ax.set_xscale("log")
    ax.set_xticks(Qs)
    ax.set_xticklabels([str(x) for x in Qs])
    ax.minorticks_off()                      # log minor labels collide with the Q labels
    ax.set_xlabel("in-target quantity Q (goal demos)")
    ax.set_ylabel("co-train − target-only (success pts)")
    ax.set_title("co-train margin vs Q", loc="left")
    for x, y in zip(Qs, marg):
        if y is not None:
            ax.annotate(f"{y:+.1f}", (x, y), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8, color=V.INK2)
    V.despine(ax)

    ax = axes[1]
    if intr:
        ys = [intr.get(str(x), {}).get("intrusion_above_median", np.nan) for x in Qs]
        sd = [intr.get(str(x), {}).get("intrusion_sd", 0) or 0 for x in Qs]
        ax.errorbar(Qs, ys, yerr=sd, marker="s", markersize=7, color=V.ORANGE, linewidth=2,
                    capsize=3, ecolor=V.INK2, elinewidth=0.9)
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color=V.MUTED, linewidth=0.8, linestyle=":")
        for x, y in zip(Qs, ys):
            if np.isfinite(y):
                ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=8, color=V.INK2)
    else:
        ax.text(0.5, 0.5, "intrusion did not run", ha="center", va="center",
                transform=ax.transAxes, color=V.MUTED)
    ax.set_xscale("log")
    ax.set_xticks(Qs)
    ax.set_xticklabels([str(x) for x in Qs])
    ax.minorticks_off()
    ax.set_xlabel("in-target quantity Q (goal demos)")
    ax.set_ylabel("outsider intrusion rate")
    ax.set_title("outsider intrusion vs Q (TracIn → C1)", loc="left")
    V.despine(ax)
    fig.tight_layout()
    _save(fig, "fig4_quantity.png")


# ---------------------------------------------------------------- 5. influence vs DTW scatter
def fig_influence_vs_dtw():
    import moderators as MO
    infl = pd.read_parquet(os.path.join(RESULTS, "influence_table.parquet"))
    best = json.load(open(os.path.join(RESULTS, "best_attributor_by_target.json")))
    z = MO.build()
    ids = [str(x) for x in z["demo_ids"]]
    D = z["dtw"]
    _, by_c = dataset.train_pool()
    cl = dataset.clusters()
    from scipy import stats as st

    fig, axes = plt.subplots(3, 3, figsize=(12, 8.5))
    rows = []
    for k, t in enumerate(cl):
        ax = axes[k // 3][k % 3]
        attr = best.get(t)
        sub = infl[(infl.attributor == attr) & (infl.functional == "plain") & (infl.target == t)]
        sc = dict(zip(sub.demo_id, sub.score))
        tgt_idx = [ids.index(d) for d in by_c[t]]
        xs, ys, cols = [], [], []
        for i, d in enumerate(ids):
            if d not in sc:
                continue
            cols_i = [j for j in tgt_idx if j != i]
            xs.append(float(D[i, cols_i].mean()))
            ys.append(sc[d])
            cols.append(V.INSIDER if d in set(by_c[t]) else V.OUTSIDER)
        ax.scatter(xs, ys, s=12, c=cols, alpha=0.8, linewidths=0.4, edgecolors=V.SURFACE)
        r = st.spearmanr(xs, ys).statistic if len(xs) > 3 else np.nan
        ax.set_title(f"{t} · {attr} · Spearman(influence, −DTW) = {-r:+.2f}", fontsize=8,
                     loc="left")
        V.despine(ax)
        rows.append({"target": t, "attributor": attr, "spearman_influence_vs_dtw": float(r)})
    for a in axes[-1]:
        a.set_xlabel("mean DTW distance to the target's training demos")
    for a in axes[:, 0] if hasattr(axes, "shape") else []:
        a.set_ylabel("influence")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="", color=V.INSIDER, label="insider"),
                        Line2D([], [], marker="o", linestyle="", color=V.OUTSIDER, label="outsider")],
               loc="upper right", ncol=2, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Influence vs trajectory-space (DTW) similarity to the target — per target",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, "fig7_influence_vs_dtw.png")
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "fig7_influence_vs_dtw.csv"), index=False)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    figs = {"transfer": fig_transfer_matrix, "insider": fig_insider_outsider,
            "lds": fig_lds_vs_ceiling, "quantity": fig_quantity, "dtw": fig_influence_vs_dtw}
    for name, fn in figs.items():
        if a.only and a.only != name:
            continue
        try:
            fn()
        except FileNotFoundError as e:
            print(f"[fig] SKIP {name}: missing input ({e})")


if __name__ == "__main__":
    main()
