"""Phase-3 figures. Reuses the Phase-1 shared style (src/vizstyle.py) unchanged.

  grain_ladder.png       -- P7's preregistered deliverable: LDS/ceiling vs grain g
  lambda_sweep.png       -- P6.5: LDS vs ridge, with the analytic lambda->inf limit
  seed_anatomy.png       -- P8a variance decomposition + P8b repairs
  ensemble_cost_law.png  -- P9: the two-sided cost law
  diffusion.png          -- P10: the replication
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3lib as L
from p3lib import P3_RESULTS, P3_FIGURES

sys.path.insert(0, os.path.join(L.ROOT, "src"))
import vizstyle as V  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

V.apply()
FOCAL = ["C1", "C5"]


def fig_grain_ladder():
    T = pd.read_csv(os.path.join(P3_RESULTS, "p7_grain_ladder.csv"))
    R = json.load(open(os.path.join(P3_RESULTS, "p7_grain_ladder.json")))
    grains = sorted(T.g.unique())

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))

    # (a) focal targets, rule (a), ratio vs g
    ax = axes[0]
    for t, c in zip(FOCAL, (V.BLUE, V.ORANGE)):
        s = T[(T.rule == "random") & (T.target == t)]
        best = [s[s.g == g].ratio.max() for g in grains]
        ax.plot(grains, best, "o-", color=c, lw=1.6, ms=5, label=t)
    ax.axhline(0.5, color=V.RED, ls="--", lw=1.2)
    ax.text(15, 0.52, "half-ceiling bar", color=V.RED, fontsize=7, ha="right")
    ax.axhline(0, color=V.MUTED, lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(grains)
    ax.set_xticklabels([str(g) for g in grains])
    ax.set_xlabel("grain g (demos per group)")
    ax.set_ylabel("best LDS / ceiling")
    ax.set_title("(a) focal targets — no g qualifies", loc="left")
    ax.legend(loc="upper right")
    V.despine(ax)

    # (b) all 9 targets
    ax = axes[1]
    for t in sorted(T.target.unique()):
        s = T[(T.rule == "random") & (T.target == t)]
        best = [s[s.g == g].ratio.max() for g in grains]
        col = V.BLUE if t == "C1" else (V.ORANGE if t == "C5" else V.GRID)
        lw = 1.8 if t in FOCAL else 1.0
        z = 3 if t in FOCAL else 1
        ax.plot(grains, best, "o-", color=col, lw=lw, ms=4, zorder=z,
                label=t if t in FOCAL else None)
    for t, c in (("C2", V.AQUA), ("C9", V.VIOLET)):
        s = T[(T.rule == "random") & (T.target == t)]
        ax.plot(grains, [s[s.g == g].ratio.max() for g in grains], "o-", color=c, lw=1.2, ms=4,
                zorder=2, label=t)
    ax.axhline(0.5, color=V.RED, ls="--", lw=1.2)
    ax.axhline(0, color=V.MUTED, lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(grains)
    ax.set_xticklabels([str(g) for g in grains])
    ax.set_xlabel("grain g (demos per group)")
    ax.set_ylabel("best LDS / ceiling")
    ax.set_title("(b) all 9 targets — C2/C9 do clear the bar", loc="left")
    ax.legend(loc="upper right", ncol=2)
    V.despine(ax)

    # (c) DTW grouping minus random grouping
    ax = axes[2]
    dm = R["dtw_minus_random_mean_over_9_targets"]
    vals = [dm[str(g)] for g in grains]
    cols = [V.GREEN if v > 0 else V.MUTED for v in vals]
    ax.bar([str(g) for g in grains], vals, color=cols, width=0.6)
    ax.axhline(0, color=V.INK2, lw=0.8)
    ax.set_xlabel("grain g")
    ax.set_ylabel("Δ LDS (DTW-grouped − random)")
    ax.set_title("(c) group STRUCTURE carries signal", loc="left")
    for i, g in enumerate(grains):
        n = R["dtw_beats_random_n_targets_of_9"][str(g)]
        ax.text(i, vals[i] + (0.002 if vals[i] >= 0 else -0.004), f"{n}/9",
                ha="center", fontsize=7, color=V.INK2)
    V.despine(ax)

    fig.tight_layout()
    p = os.path.join(P3_FIGURES, "grain_ladder.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"[fig] {p}")


def fig_lambda():
    T = pd.read_csv(os.path.join(P3_RESULTS, "p6_lambda_sweep_extended.csv"))
    J = json.load(open(os.path.join(P3_RESULTS, "p6_lambda_sweep_extended.json")))
    fin = T[np.isfinite(T.ridge_rel)]
    grid = sorted(fin.ridge_rel.unique())

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, t, c in zip(axes, FOCAL, (V.BLUE, V.ORANGE)):
        for a, ls in (("IF", "-"), ("TRAK", "--")):
            s = fin[(fin.target == t) & (fin.attributor == a)].sort_values("ridge_rel")
            ax.plot(s.ridge_rel, s.demo_lds, ls, color=c, lw=1.6, ms=4, marker="o",
                    label=a, alpha=1.0 if a == "IF" else 0.55)
        lim = T[(T.attributor == "GradDot_limit") & (T.target == t)].demo_lds.iloc[0]
        ax.axhline(lim, color=V.GREEN, ls=":", lw=1.4)
        ax.text(grid[0], lim + 0.02, "λ→∞ limit = gradient dot product (no preconditioner)",
                color=V.GREEN, fontsize=6.5)
        bar = T[(T.target == t)].demo_bar_half_ceiling.iloc[0]
        ax.axhline(bar, color=V.RED, ls="--", lw=1.2)
        ax.text(grid[-1], bar + 0.02, "half-ceiling bar", color=V.RED, fontsize=7, ha="right")
        ax.axvline(1e-2, color=V.MUTED, lw=1.0)
        ax.text(1e-2, ax.get_ylim()[0], " Phase-2 default λ", color=V.INK2, fontsize=7,
                rotation=90, va="bottom")
        ax.axhline(0, color=V.MUTED, lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("ridge (relative to mean diag G)")
        ax.set_ylabel("demo-grain LDS")
        ax.set_title(f"{t}", loc="left")
        ax.legend(loc="lower right")
        V.despine(ax)
    fig.suptitle("P6.5 — the exact preconditioner HURTS: LDS is maximized where it is switched off",
                 fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(P3_FIGURES, "lambda_sweep.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"[fig] {p}")


def fig_seed_anatomy():
    pa = os.path.join(P3_RESULTS, "p8a_variance_decomposition.json")
    if not os.path.exists(pa):
        print("[fig] P8a not done; skipping seed_anatomy")
        return
    A = json.load(open(pa))
    pb = os.path.join(P3_RESULTS, "p8b_variance_reduction.json")
    B = json.load(open(pb)) if os.path.exists(pb) else None

    n = 2 if B else 1
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 3.6))
    axes = np.atleast_1d(axes)

    ax = axes[0]
    ocs = ["neg_plain_loss", "logit_success"]
    labs = ["held-out L2", "logit success"]
    comps = ["INIT", "ORDER", "INTERACTION_PLUS_RESIDUAL"]
    cols = [V.BLUE, V.ORANGE, V.GRID]
    x = np.arange(len(ocs))
    bot = np.zeros(len(ocs))
    for c, col in zip(comps, cols):
        v = [A["results"][o]["variance_fraction"][c] for o in ocs]
        ax.bar(x, v, bottom=bot, color=col, width=0.55,
               label=c.replace("_PLUS_", "+").replace("INTERACTION+RESIDUAL", "inter+resid"))
        for i, (vv, bb) in enumerate(zip(v, bot)):
            if vv > 0.06:
                ax.text(i, bb + vv / 2, f"{vv:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if col != V.GRID else V.INK)
        bot += np.array(v)
    ax.axhline(0.70, color=V.RED, ls="--", lw=1.2)
    ax.text(1.45, 0.71, "70% dominance", color=V.RED, fontsize=7, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.set_ylabel("share of seed variance")
    ax.set_title("(a) P8a — what IS the seed noise?", loc="left")
    ax.legend(loc="upper right", ncol=1)
    V.despine(ax)

    if B:
        ax = axes[1]
        r = B["results"]["neg_plain_loss"]
        names = ["single seed\n(S=1)", "ckpt-avg\n(S=1, FREE)", "outcome mean\n(S=3)",
                 "action ens.\n(S=3)"]
        vals = [r["baseline_S1_single_seed"], r["arm_i_S1_checkpoint_averaged"],
                r["baseline_S3_outcome_mean"], r["arm_ii_S3_action_ensemble"]]
        cols2 = [V.MUTED, V.GREEN, V.MUTED, V.VIOLET]
        ax.bar(names, vals, color=cols2, width=0.6)
        ax.axhline(0, color=V.INK2, lw=0.8)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01, f"{v:+.2f}", ha="center", fontsize=8, color=V.INK2)
        ax.set_ylabel("split-half mask-ranking reliability")
        ax.set_title("(b) P8b — do the cheap repairs work?", loc="left")
        V.despine(ax)

    fig.tight_layout()
    p = os.path.join(P3_FIGURES, "seed_anatomy.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"[fig] {p}")


def fig_cost_law():
    p9 = os.path.join(P3_RESULTS, "p9_ensemble_cost_law.json")
    if not os.path.exists(p9):
        print("[fig] P9 not done; skipping cost_law")
        return
    J = json.load(open(p9))
    T = pd.DataFrame(J["table"])
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))

    ax = axes[0]
    for a, c in zip(["IF", "TRAK", "TracIn"], (V.BLUE, V.ORANGE, V.GREEN)):
        s = T[T.attributor == a].sort_values("E")
        ax.plot(s.E, s.split_half_reliability_mean_over_9_targets, "o-", color=c, lw=1.6, ms=5,
                label=a)
    ax.axhline(0.8, color=V.RED, ls="--", lw=1.2)
    ax.text(20, 0.81, "r = 0.8 target", color=V.RED, fontsize=7, ha="right")
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 20])
    ax.set_xticklabels(["2", "5", "10", "20"])
    ax.set_xlabel("ensemble size E (attribution side)")
    ax.set_ylabel("split-half per-demo ranking reliability")
    ax.set_title("(a) P9 — the attribution side", loc="left")
    ax.legend(loc="lower right")
    V.despine(ax)

    ax = axes[1]
    for a, c in zip(["IF", "TRAK", "TracIn"], (V.BLUE, V.ORANGE, V.GREEN)):
        s = T[T.attributor == a].sort_values("E")
        ax.plot(s.E, s.top15_jaccard_mean_across_subensembles, "o-", color=c, lw=1.6, ms=5,
                label=a)
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 20])
    ax.set_xticklabels(["2", "5", "10", "20"])
    ax.set_xlabel("ensemble size E")
    ax.set_ylabel("top-15 Jaccard across sub-ensembles")
    ax.set_title("(b) does the headline top-15 even stabilize?", loc="left")
    ax.legend(loc="lower right")
    V.despine(ax)

    fig.tight_layout()
    p = os.path.join(P3_FIGURES, "ensemble_cost_law.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"[fig] {p}")


def fig_diffusion():
    """P10: the CEILING is the story. Panel (a) the broken instrument; (b) the repaired one."""
    ps6 = os.path.join(P3_RESULTS, "p10_verdict_S6.json")
    pmd = os.path.join(P3_RESULTS, "p10_diagnostic_median.json")
    if not (os.path.exists(ps6) and os.path.exists(pmd)):
        print("[fig] P10 not done; skipping diffusion")
        return
    S6 = json.load(open(ps6))
    MD = json.load(open(pmd))
    p1 = json.load(open(os.path.join(L.P2_RESULTS, "p1_demo_grain.json")))
    cl = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    # (a) the broken instrument: mean-aggregated ceilings collapse (some negative)
    ax = axes[0]
    cm = [S6["all_targets"][t]["neg_plain_loss"]["ceiling_6seed_SB"] for t in cl]
    cd = [MD["all_targets"][t]["ceiling_median_6seed_SB"] for t in cl]
    x = np.arange(len(cl))
    ax.bar(x - 0.2, cm, 0.4, color=V.MUTED, label="seed MEAN (preregistered)")
    ax.bar(x + 0.2, cd, 0.4, color=V.GREEN, label="seed MEDIAN")
    ax.axhline(0, color=V.INK2, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(cl, fontsize=7)
    ax.set_ylabel("measured 6-seed ceiling")
    ax.set_title("(a) the instrument broke — and the median fixes it", loc="left", fontsize=9)
    ax.legend(loc="upper left", fontsize=7)
    V.despine(ax)

    # (b) why: heavy tails. seed spread of held-out L2, BC vs diffusion
    ax = axes[1]
    import pandas as pd
    dfd = pd.read_parquet(os.path.join(P3_RESULTS, "p10b_outcomes_S6.parquet"))
    g6 = pd.read_parquet(os.path.join(L.P2_RESULTS, "stage_G6_outcomes.parquet"))
    for name, d, c in (("BC-Transformer", g6, V.BLUE), ("Diffusion", dfd, V.VIOLET)):
        s = d[d.target == "C1"]
        piv = s.pivot_table(index="mask_id", columns="seed", values="plain_loss")
        r = (piv.max(1) / piv.min(1)).values
        ax.hist(r, bins=np.linspace(1, 5, 17), alpha=0.65, color=c, label=name)
    ax.set_xlabel("max/min held-out L2 across seeds (per mask, C1)")
    ax.set_ylabel("masks")
    ax.set_title("(b) why: the diffusion L2 is heavy-tailed across seeds", loc="left", fontsize=9)
    ax.legend(loc="upper right", fontsize=7)
    V.despine(ax)

    # (c) the comparison, each judged against its OWN usable ceiling
    ax = axes[2]
    w = 0.36
    for i, t in enumerate(FOCAL):
        bc = p1["focal_verdict"][t]
        dp = MD["focal_verdict"][t]
        ax.bar(i - w / 2, bc["best_rho"], w, color=V.BLUE,
               label="BC-Transformer (S=6, mean)" if i == 0 else None)
        ax.bar(i + w / 2, dp["best_rho"], w, color=V.VIOLET,
               label="Diffusion (S=6, MEDIAN — post-hoc)" if i == 0 else None)
        ax.plot([i - w, i], [bc["bar_half_ceiling"]] * 2, color=V.RED, ls="--", lw=1.4)
        ax.plot([i, i + w], [dp["bar"]] * 2, color=V.RED, ls="--", lw=1.4)
        ax.text(i - w / 2, bc["best_rho"] + 0.015, f"{bc['best_ratio']:.2f}", ha="center",
                fontsize=7.5, color=V.INK2)
        ax.text(i + w / 2, dp["best_rho"] + 0.015, f"{dp['best_ratio']:.2f}", ha="center",
                fontsize=7.5, color=V.INK2)
    ax.axhline(0, color=V.INK2, lw=0.8)
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(FOCAL)
    ax.set_ylabel("demo-grain LDS (held-out L2)")
    ax.set_title("(c) MIXED: C1 crosses, C5 does not", loc="left", fontsize=9)
    ax.legend(loc="upper right", fontsize=7)
    V.despine(ax)

    fig.tight_layout()
    q = os.path.join(P3_FIGURES, "diffusion.png")
    fig.savefig(q)
    plt.close(fig)
    print(f"[fig] {q}")


if __name__ == "__main__":
    which = sys.argv[1:] or ["grain", "lambda", "seed", "cost", "diff"]
    if "grain" in which:
        fig_grain_ladder()
    if "lambda" in which:
        fig_lambda()
    if "seed" in which:
        fig_seed_anatomy()
    if "cost" in which:
        fig_cost_law()
    if "diff" in which:
        fig_diffusion()
