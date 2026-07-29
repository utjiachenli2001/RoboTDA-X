"""PASS 9 figures. Zero GPU.

Two panels, and the pairing is the argument of the pass:

LEFT -- the |S| correction. Campaign N's committed primary next to the same computation within each
stratum, with the permutation null drawn on each bar. The pooled bar clears the half-ceiling line;
none of the stratified bars do; and the pooled bar's null is enormous while the stratified nulls are
at zero. That is the whole finding in one picture: pooling credits the estimator for training-set
size.

RIGHT -- the grain curve at a fixed training-set size, with bootstrap CIs on the ratio. Every rung
beats its null and none clears the bar. k=3 and k=5 sit on top of each other; k=15's point estimate
is higher but its interval swallows both, which is why the crossover is reported as unlocated rather
than as a number.

Both panels are drawn on the rho/r scale that every historical number in this repo uses, with the
rho/sqrt(r) value annotated, because rho/r is inflated and #42 says so.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGS = os.path.join(HERE, "figs")
STAT = "kendall_tau_b"
BAR = 0.5


def _load():
    sc = pd.read_csv(os.path.join(RESULTS, "p9_stratum_control.csv"))
    co = pd.read_csv(os.path.join(RESULTS, "confirm_oseries.csv"))
    return sc[sc.statistic == STAT], co[co.statistic == STAT]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(FIGS, "p9_grain_and_size.png"))
    a = ap.parse_args()
    sc, co = _load()
    os.makedirs(FIGS, exist_ok=True)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))

    # ---------------- LEFT: the |S| correction
    order = ["POOLED (committed primary)", "within 4of9", "within 5of9", "within 6of9"]
    sub = sc.set_index("scope").loc[[o for o in order if o in set(sc.scope)]]
    x = range(len(sub))
    colors = ["#b03030" if "POOLED" in s else "#3b6ea5" for s in sub.index]
    axL.bar(x, sub["ratio"], color=colors, width=0.6, label="ratio = rho / ceiling")
    axL.errorbar(x, sub["ratio"],
                 yerr=[sub["ratio"] - sub["ratio_ci_lo"], sub["ratio_ci_hi"] - sub["ratio"]],
                 fmt="none", ecolor="black", capsize=4, lw=1.2)
    # the permutation null, expressed on the same ratio scale
    axL.plot(x, sub["perm_mean"] / sub["ceiling"], "kv", ms=9,
             label="permutation null (fixed estimator,\noutcomes shuffled within stratum)")
    axL.axhline(BAR, ls="--", c="grey", lw=1.4)
    axL.text(len(sub) - 0.45, BAR + 0.02, "half-ceiling bar", ha="right", fontsize=9, color="grey")
    axL.set_xticks(list(x))
    axL.set_xticklabels(["pooled\n(committed)", "4of9\n60 demos", "5of9\n75 demos",
                         "6of9\n90 demos"], fontsize=9)
    axL.set_ylabel(f"ratio to ceiling  ({STAT})")
    axL.set_title("Campaign N: the primary is pooled over |S|,\nand a third of it survives a shuffle",
                  fontsize=11)
    axL.legend(fontsize=8, loc="upper right")
    axL.set_ylim(-0.1, 1.05)
    for i, (s, r) in enumerate(zip(sub.index, sub["ratio"])):
        axL.text(i, -0.06, f"n={int(sub['n_masks'].iloc[i])}", ha="center", fontsize=8,
                 color="#444")

    # ---------------- RIGHT: the grain curve at a fixed training set
    co = co.copy()
    co["k"] = co["rung"].str.replace("k=", "", regex=False).astype(int)
    co = co.sort_values("k")
    axR.errorbar(co["k"], co["ratio"],
                 yerr=[co["ratio"] - co["ratio_ci_lo"], co["ratio_ci_hi"] - co["ratio"]],
                 fmt="o-", color="#3b6ea5", capsize=5, lw=1.6, ms=8, label="ratio (95% CI)")
    axR.plot(co["k"], co["perm_null_mean"] / co["ceiling"], "kv", ms=9,
             label="permutation null")
    axR.axhline(BAR, ls="--", c="grey", lw=1.4)
    axR.text(3.02, BAR + 0.07, "half-ceiling bar", ha="left", fontsize=9, color="grey")
    axR.set_xscale("log")
    axR.set_xticks(list(co["k"]))
    axR.set_xticklabels([str(v) for v in co["k"]])
    # a log axis emits 4x10^0 / 6x10^0 minor labels between the rungs; the x values are the three
    # grains and nothing else, so the minor ticks are noise
    axR.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axR.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axR.set_xlabel("grain k  (demos removed as one unit)")
    axR.set_ylabel(f"ratio to ceiling  ({STAT})")
    axR.set_title("Campaign O: training set fixed at 75 demos.\n"
                  "No grain clears the bar; the trend is unestablished", fontsize=11)
    for _, r in co.iterrows():
        axR.annotate(f"n={int(r['n_masks'])}\nrho/sqrt(r)={r['ratio_sqrt']:.2f}",
                     (r["k"], r["ratio"]), textcoords="offset points", xytext=(10, -30),
                     fontsize=8, color="#444")
    axR.legend(fontsize=8, loc="upper left")
    axR.set_ylim(-0.25, 2.5)

    fig.suptitle("Pass 9 -- the cluster-grain result was substantially a training-set-SIZE result",
                 fontsize=12.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(a.out, dpi=150)
    print(f"[p9/figs] -> {a.out}")


if __name__ == "__main__":
    main()
