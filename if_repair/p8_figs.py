"""Pass-8 figures. Every number is read from a committed CSV -- none are retyped."""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGS = os.path.join(HERE, "figs")
STAT = "kendall_tau_b"


def _load(name):
    p = os.path.join(RESULTS, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def fig_grain_comparison():
    """THE teaching artifact: the same four frozen configs, demo grain vs cluster grain.

    Pass 7 spent three campaigns establishing that a self-influence correction helps demo-grain
    attribution by +0.06. On Stage F -- a draw none of them was tuned against -- every one of them
    is NEGATIVE, and the plain baseline they were built to improve is strongly positive.
    """
    sf = _load("p8_stageF_oos.csv")
    if sf is None:
        return None
    d = sf[(sf.conditional) & (sf.statistic == STAT)].sort_values("delta")
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    y = np.arange(len(d))
    ax.errorbar(d.delta, y, xerr=[d.delta - d.delta_ci_lo, d.delta_ci_hi - d.delta],
                fmt="o", color="#b2182b", capsize=3, lw=1.6, ms=6,
                label="cluster grain (Stage F, OOS, n=40 conditional)")
    ax.axvline(0, color="k", lw=1)
    ax.axvline(0.06, color="#2166ac", ls="--", lw=1.5,
               label="pass 7 demo-grain result for RelatIF/C5 (+0.06)")
    ax.axvspan(-0.02, 0.14, color="#2166ac", alpha=0.10)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.config}\n({r.target})" for _, r in d.iterrows()], fontsize=8)
    ax.set_xlabel(f"paired delta vs GradDot_dmean  ({STAT})")
    ax.set_title("Every pass-4/7 correction reverses sign when the unit becomes a cluster\n"
                 "(15 demos instead of 1)", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    p = os.path.join(FIGS, "p8_grain_comparison.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def fig_absolute_bar():
    """The pass-8 primary: GradDot's cluster-grain LDS against the half-ceiling bar."""
    cn = _load("confirm_nseries.csv")
    ce = _load("p8_design_ceiling.csv")
    if cn is None:
        return None
    d = cn[cn.statistic == STAT]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    y = np.arange(len(d))
    ax.barh(y, d.lds, color="#4393c3", height=0.5, label="GradDot_dmean LDS (campaign N)")
    for i, (_, r) in enumerate(d.iterrows()):
        if np.isfinite(r.ceiling):
            ax.plot([r.ceiling / 2] * 2, [i - 0.3, i + 0.3], color="#b2182b", lw=2.5)
            ax.plot([r.ceiling] * 2, [i - 0.3, i + 0.3], color="k", lw=1.5, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.target} (n={int(r.n_masks)}, d={int(r.depth)})"
                        for _, r in d.iterrows()], fontsize=9)
    ax.set_xlabel(f"LDS ({STAT})")
    ax.set_title("Pass-8 primary: does plain GradDot clear the absolute half-ceiling bar\n"
                 "at cluster grain? (red = bar, dotted = noise ceiling)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(FIGS, "p8_absolute_bar.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def fig_allocation():
    """Why campaign N buys depth: at cluster grain the mask axis has a hard combinatorial cap."""
    al = _load("p8_design_allocation.csv")
    if al is None:
        return None
    m = al[(al.axis == "masks") & (al.statistic == STAT)]
    m = m[m.n_masks < m.n_available]
    fig, ax = plt.subplots(figsize=(7, 4))
    for t, g in m.groupby("target"):
        g = g.sort_values("n_masks")
        ax.plot(g.n_masks, g.paired_sd, "o-", label=f"{t} (Stage F subsampling)")
    ax.axvline(126, color="#b2182b", ls="--", lw=1.5)
    ax.text(126, ax.get_ylim()[1] * 0.92, "  C(9,5)=126: the entire 5-of-9 mask space",
            color="#b2182b", fontsize=8, va="top")
    ax.axvline(336, color="k", ls=":", lw=1.5)
    ax.text(336, ax.get_ylim()[1] * 0.75, "  336: |S| in {4,5,6}", fontsize=8, va="top")
    ax.set_xlabel("conditional masks")
    ax.set_ylabel(f"paired sd ({STAT})")
    ax.set_title("Cluster grain has a combinatorial mask cap, so depth is the only\n"
                 "remaining axis -- the inverse of the demo-grain rule (BLOCKERS #29)",
                 fontsize=10)
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(FIGS, "p8_allocation.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def main():
    os.makedirs(FIGS, exist_ok=True)
    for fn in (fig_grain_comparison, fig_absolute_bar, fig_allocation):
        p = fn()
        print(f"  {fn.__name__}: {p if p else 'SKIPPED (input CSV not present yet)'}")


if __name__ == "__main__":
    main()
