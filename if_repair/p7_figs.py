"""Pass-7 figures. The forest plot of the paired contrast is the headline deliverable of W0.1:
it is a paper figure regardless of which way the effect goes.

Reads the committed CSVs written by p7_pooled_oos.py --stage contrast, so it is instant and
cannot silently disagree with the numbers in the tables.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
R, FG = os.path.join(HERE, "results"), os.path.join(HERE, "figs")

ROLE_STYLE = {"dev": ("#7f7f7f", "o", "dev (selection)"),
              "discovery": ("#d62728", "D", "discovery draw"),
              "oos": ("#1f77b4", "s", "out-of-sample")}
TITLE = {"relatif_C5": "RelatIF K/G_dd — C5", "surrogate_C5": "exact surrogate-LOO — C5",
         "ensemble_C5": "z-avg ensemble — C5", "leverage_C7": "leverage λ=1 β=1 — C7"}


def forest():
    d = pd.read_csv(os.path.join(R, "p7_per_draw.csv"))
    p = pd.read_csv(os.path.join(R, "p7_pooled_oos.csv"))
    h = pd.read_csv(os.path.join(R, "p7_heterogeneity.csv"))
    cfgs = list(TITLE)
    fig, axes = plt.subplots(1, len(cfgs), figsize=(4.6 * len(cfgs), 6.0), sharex=True)
    for ax, name in zip(np.atleast_1d(axes), cfgs):
        sub = d[d.config == name].reset_index(drop=True)
        ys, labels = [], []
        for i, r in sub.iterrows():
            y = len(sub) - i
            col, mk, _ = ROLE_STYLE[r.role]
            ax.plot([r.ci_lo, r.ci_hi], [y, y], color=col, lw=1.6, solid_capstyle="butt")
            ax.plot([r.paired_delta_rho], [y], mk, color=col, ms=7)
            ys.append(y)
            labels.append(f"{r.draw}  ({r.role})")
        base = min(ys)
        # pooled rows
        for j, (_, r) in enumerate(p[p.config == name].iterrows()):
            y = base - 1.2 - j
            ax.plot([r.ci_lo, r.ci_hi], [y, y], color="black", lw=2.4, solid_capstyle="butt")
            ax.plot([r.paired_delta_rho], [y], "*", color="black", ms=15)
            ys.append(y)
            tag = "pooled" if r.analysis == "full_information" else "pooled, honest"
            labels.append(f"{tag}  n={int(r.n_masks)} [{r.draws}]")
        hh = h[h.config == name]
        if len(hh):
            r = hh.iloc[0]
            y = min(ys) - 1.2
            ax.plot([r.RE_lo, r.RE_hi], [y, y], color="#2ca02c", lw=2.4, solid_capstyle="butt")
            ax.plot([r.RE], [y], "*", color="#2ca02c", ms=15)
            ys.append(y)
            labels.append(f"random-effects  (I²={r.I2:.0%})")
        ax.axvline(0, color="black", lw=1)
        ax.axvline(0.15, color="red", ls="--", lw=1.1)
        ax.set_yticks(ys)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_ylim(min(ys) - 0.8, max(ys) + 0.8)
        ax.set_xlabel("paired Δρ  vs  GradDot_dmean")
        ax.set_title(TITLE[name], fontsize=10)
        ax.grid(axis="x", alpha=0.25)
    hs = [plt.Line2D([], [], color=c, marker=m, ls="-", label=l)
          for c, m, l in ROLE_STYLE.values()]
    hs += [plt.Line2D([], [], color="black", marker="*", ls="-", ms=12, label="pooled mask statistic"),
           plt.Line2D([], [], color="#2ca02c", marker="*", ls="-", ms=12, label="random-effects summary"),
           plt.Line2D([], [], color="red", ls="--", label="Δρ = +0.15")]
    fig.legend(handles=hs, loc="lower center", ncol=6, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Pass 7 W0.1 — paired contrast vs GradDot_dmean across all six mask draws\n"
                 "(95% CI: mask bootstrap, stratified within draw)", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    out = os.path.join(FG, "p7_forest.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    os.makedirs(FG, exist_ok=True)
    forest()
