"""Phase-2 figures. Every number is read from an artifact file; nothing is recomputed here."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter, NullLocator  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, "/mnt/sdb/ljc/RoboTDA-X/src")
import bootstrap  # noqa: F401
from bootstrap import ROOT  # noqa: E402

P2 = os.path.join(ROOT, "phase2")
FIG = os.path.join(P2, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 140, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})
C_CEIL, C_LDS = "#1f77b4", "#d62728"


def _logx(ax, QS):
    """log x-axis showing ONLY the three Q values (matplotlib's minor decade labels collide)."""
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(QS)
    ax.set_xticklabels([str(q) for q in QS])


def fig_seed_ladder():
    """P1: the ceiling climbs, the LDS does not."""
    d = json.load(open(f"{P2}/results/p1_seed_ladder.json"))["ladder"]
    p1 = json.load(open(f"{P2}/results/p1_demo_grain.json"))
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), sharey=True)
    for ax, t in zip(axes, ["C1", "C5"]):
        S = sorted(int(k) for k in d[t])
        ceil = [d[t][str(s)]["ceiling_SvS"] for s in S]
        best = [d[t][str(s)]["best_lds_mean"] for s in S]
        ok = [i for i, c in enumerate(ceil) if np.isfinite(c)]
        ax.plot([S[i] for i in ok], [ceil[i] for i in ok], "o-", color=C_CEIL,
                label="noise ceiling (S-vs-S split-half)")
        c6 = p1["all_targets"][t]["neg_plain_loss"]["ceiling_6seed_SB"]
        ax.plot([6], [c6], "o", mfc="white", mec=C_CEIL, ms=7)
        ax.annotate(f"{c6:.2f}\n(SB)", (6, c6), textcoords="offset points", xytext=(-4, 6),
                    color=C_CEIL, fontsize=7, ha="center")
        ax.plot(S, best, "s-", color=C_LDS, label="best attributor LDS")
        bar = 0.5 * c6
        ax.axhline(bar, ls="--", lw=1, color="gray")
        ax.annotate(f"bar = ½·ceiling = {bar:.2f}", (1.05, bar), fontsize=7, color="gray",
                    va="bottom")
        ax.set_title(f"{t}  (focal)", fontsize=10)
        ax.set_xlabel("training seeds averaged into ground truth (S)")
        ax.set_xticks(S)
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("Spearman ρ")
    axes[0].legend(fontsize=7, loc="lower right", frameon=False)
    fig.suptitle("P1 — seed-ensembling raises the CEILING, not the ATTRIBUTION\n"
                 "held-out L2, same 24 Stage-G demo masks", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{FIG}/p1_seed_ladder.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote p1_seed_ladder.png")


def fig_regime_boundary():
    """P3: the headline regime-boundary figure."""
    r = json.load(open(f"{P2}/results/p3_regime_boundary.json"))
    QS = r["Q_values"]
    tab = pd.read_csv(f"{P2}/results/p3_lds_table.csv")
    snr = r["DESCRIPTIVE_signal_vs_seed_noise"]["per_Q"]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))

    ax = axes[0]
    ax.plot(QS, [r["per_Q"][str(q)]["outcomes"]["neg_plain_loss"]["ceiling_4seed_SB"] for q in QS],
            "o-", color=C_CEIL, label="held-out L2")
    ax.plot(QS, [r["per_Q"][str(q)]["outcomes"]["logit_success"]["ceiling_4seed_SB"] for q in QS],
            "s-", color="#ff7f0e", label="closed-loop success")
    ax.axhline(0, color="gray", lw=0.8)
    _logx(ax, QS)
    ax.set_xlabel("Q (in-target demos)"); ax.set_ylabel("noise ceiling (4-seed, SB)")
    ax.set_title("(a) ground-truth ceiling vs data scale", fontsize=9)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    for a, mk in zip(["IF", "TRAK", "TracIn"], ["o", "s", "^"]):
        s = tab[(tab.attributor == a) & (tab.outcome == "neg_plain_loss")].sort_values("Q")
        ax.plot(s.Q, s.rho, mk + "-", label=a, ms=4)
    ax.axhline(0, color="gray", lw=0.8)
    _logx(ax, QS)
    ax.set_xlabel("Q (in-target demos)"); ax.set_ylabel("LDS (Spearman) on held-out L2")
    ax.set_title("(b) attribution LDS vs data scale", fontsize=9)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[2]
    b = [snr[str(q)]["success"]["between_mask_sd"] for q in QS]
    w = [snr[str(q)]["success"]["within_mask_seed_sd"] for q in QS]
    ax.plot(QS, b, "o-", color="#2ca02c", label="between-mask sd (data signal)")
    ax.plot(QS, w, "s-", color="#9467bd", label="within-mask seed sd (noise)")
    _logx(ax, QS)
    ax.set_xlabel("Q (in-target demos)"); ax.set_ylabel("sd of closed-loop success")
    ax.set_title("(c) WHY: the data signal shrinks, the seed noise does not", fontsize=9)
    ax.legend(fontsize=7, frameon=False)

    fig.suptitle("P3 — REGIME BOUNDARY: reliability does NOT improve with data scale; "
                 "for closed-loop success it gets WORSE", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{FIG}/regime_boundary.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote regime_boundary.png")


def fig_success_reliability():
    """P4: the reliability curve + the hard 50-episode instrument wall."""
    if not os.path.exists(f"{P2}/results/p4_reliability.csv"):
        print("p4: did not run -- skipping figure")
        return
    t = pd.read_csv(f"{P2}/results/p4_reliability.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5), sharey=True)
    for ax, c in zip(axes, ["C1", "C2"]):
        s = t[t.cluster == c]
        for S in sorted(s.seeds_averaged.unique()):
            g = s[s.seeds_averaged == S].sort_values("episodes_per_estimate")
            ex = bool(g.is_extrapolation.iloc[0]) if "is_extrapolation" in g else False
            ax.plot(g.episodes_per_estimate, g.reliability,
                    ("--" if ex else "-") + "o", ms=4,
                    label=f"S={S}" + (" (extrap.)" if ex else ""))
        ax.axhline(0.5, ls=":", color="gray", lw=1)
        ax.axhline(0.8, ls=":", color="gray", lw=1)
        ax.axvline(150, color="crimson", lw=1)
        ax.annotate("instrument wall\n(50 init states/task)", (150, 0.02), fontsize=6.5,
                    color="crimson", ha="right", rotation=90, va="bottom")
        ax.set_title(f"{c} ({'near-floor' if c == 'C1' else 'mid-range'})", fontsize=9)
        ax.set_xlabel("distinct episodes per success estimate")
    axes[0].set_ylabel("split-half reliability of success")
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("P4 — what closed-loop success costs as demo-grain ground truth\n"
                 "(dotted = the 0.5 / 0.8 reliability targets; dashed = Spearman-Brown extrapolation)",
                 fontsize=9.5)
    fig.tight_layout()
    fig.savefig(f"{FIG}/success_reliability.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote success_reliability.png")


def fig_p2_scatter():
    """P2: predicted transfer benefit vs measured per-task margin (the primary estimator)."""
    f = f"{P2}/results/p2_pred_vs_meas_stageE_IF.csv"
    if not os.path.exists(f):
        return
    d = pd.read_csv(f)
    st = json.load(open(f"{P2}/results/p2_attribution_stability.json"))
    res = json.load(open(f"{P2}/results/p2_transfer_sign.json"))
    h = res["PREREGISTERED_PRIMARY"]

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5))
    ax = axes[0]
    for c, col in zip(["C1", "C2", "C5"], ["#1f77b4", "#2ca02c", "#ff7f0e"]):
        s = d[d.cluster == c]
        ax.scatter(s.pred_sum_outsider, s.margin_pts, s=26, color=col, label=c, alpha=.85)
    ax.axhline(0, color="gray", lw=.8); ax.axvline(0, color="gray", lw=.8)
    ax.set_xlabel("predicted benefit (Σ outsider attribution)")
    ax.set_ylabel("measured co-train margin (pts)")
    ax.set_title(f"(a) primary: IF / Stage-E  ρ={h['rho']:+.3f}, p={h['p_onesided']:.3f} → FAIL",
                 fontsize=9)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    sp = st["subensemble_spread"]
    xs = ["IF", "TRAK", "TracIn"]
    ax.bar(xs, [sp[a]["pooled_rho_mean"] for a in xs],
           yerr=[[sp[a]["pooled_rho_mean"] - sp[a]["pooled_rho_min"] for a in xs],
                 [sp[a]["pooled_rho_max"] - sp[a]["pooled_rho_mean"] for a in xs]],
           color="#8c8c8c", capsize=4, alpha=.85)
    ax.axhline(0.323, ls="--", color="crimson", lw=1)
    ax.annotate("critical ρ (α=.05, n=27)", (2.42, 0.333), fontsize=6.5, color="crimson", ha="right")
    ax.axhline(0, color="gray", lw=.8)
    ax.set_ylabel("pooled ρ over 5-member sub-ensembles")
    ax.set_title("(b) same 135 demos, different SEEDS:\nthe statistic swings across the bar",
                 fontsize=9)
    fig.suptitle("P2 — attribution does not predict per-task transfer, and its 'significance' "
                 "is a seed lottery", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{FIG}/p2_transfer_sign.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote p2_transfer_sign.png")


if __name__ == "__main__":
    fig_seed_ladder()
    fig_regime_boundary()
    fig_p2_scatter()
    fig_success_reliability()
