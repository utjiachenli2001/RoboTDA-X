"""Figures: k_sweep.png (LDS vs k, both endpoints marked) and spectrum.png."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D, spectral as S  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R, F = os.path.join(HERE, "results"), os.path.join(HERE, "figs")


def k_sweep_fig():
    null = json.load(open(os.path.join(R, "spectrum_null.json")))
    kstar = null["k_star_median"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for ax, t in zip(axes, ("C1", "C5")):
        d = json.load(open(os.path.join(R, f"k_sweep_{t}.json")))
        for nrm, col in (("dmean", "#1f77b4"), ("none", "#7f7f7f")):
            c = d[nrm]
            ax.plot([r["k"] for r in c], [r["lds"] for r in c], lw=1.2, color=col,
                    label=f"truncated_if ({nrm})")
        c = d["dmean"]
        ceil = c[0]["ceiling"]
        ax.axhline(ceil, ls="-", lw=1, color="green", label=f"ceiling {ceil:.3f}")
        ax.axhline(0.5 * ceil, ls="--", lw=1.2, color="red",
                   label=f"bar = ½·ceiling {0.5*ceil:.3f}")
        ax.axhline(c[0]["lds"], ls=":", lw=1.5, color="#d62728",
                   label=f"GradDot (k=0) {c[0]['lds']:+.3f}")
        ax.axhline(c[-1]["lds"], ls=":", lw=1.5, color="#9467bd",
                   label=f"exact IF (k=N) {c[-1]['lds']:+.3f}")
        ax.axvline(kstar, ls="-.", lw=1.2, color="orange", label=f"k* = {kstar}")
        best = max(c, key=lambda r: r["lds"])
        ax.plot([best["k"]], [best["lds"]], "o", ms=8, mfc="none", mec="black",
                label=f"argmax k={best['k']} ({best['lds']:+.3f})")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{t}: demo-grain LDS vs truncation rank k  (E=20, bc_s10, n=24)")
        ax.set_xlabel("k  (eigendirections inverted; rest = identity)")
        ax.set_ylabel("demo-grain LDS (Spearman)")
        ax.legend(fontsize=7, loc="lower left", ncol=2)
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(os.path.join(F, "k_sweep.png"), dpi=150)
    print("wrote figs/k_sweep.png")


def spectrum_fig():
    Z = D.gram_e20()
    sp = S.gram_spectrum(Z)
    null = json.load(open(os.path.join(R, "spectrum_null.json")))
    env = np.array(null["null_envelope_median_over_members"])
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for m in range(sp.shape[0]):
        ax.semilogy(sp[m], color="#1f77b4", alpha=.25, lw=.8)
    ax.semilogy(np.median(sp, 0), color="#1f77b4", lw=2, label="Gram eigenvalues (median)")
    ax.semilogy(env, color="orange", lw=2, ls="--",
                label=f"permutation null, {null['percentile']:.0f}th pct "
                      f"({null['n_perm']} perms)")
    ax.axvline(null["k_star_median"], color="red", ls="-.",
               label=f"k* = {null['k_star_median']}")
    ax.set_xlabel("eigenvalue index (descending)")
    ax.set_ylabel("eigenvalue")
    ax.set_title("Gram spectrum vs sampling-noise floor (E=20)")
    ax.legend(fontsize=8)
    ax.grid(alpha=.25, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(F, "spectrum.png"), dpi=150)
    print("wrote figs/spectrum.png")


if __name__ == "__main__":
    os.makedirs(F, exist_ok=True)
    k_sweep_fig()
    spectrum_fig()
