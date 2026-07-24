"""The pass-7 summary figure: RelatIF/C5 vs GradDot_dmean across every mask draw the project has.

Everything is recomputed here under a single statistic per panel, so the figure is internally
consistent rather than splicing the Spearman per-draw CSV to the Kendall campaign-M row.

Rows: G/H/I (dev, selection), J (discovery), K/L (clean OOS at depth 10), M0..M5 (the six
campaign-M sub-draws at depth 2), then the pooled estimates. The story is meant to be legible
without the caption: every draw the effect was selected or reported on sits to the right, the
clean draws straddle zero, and the pooled interval is small and touches it.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import functionals as F  # noqa: E402
from if_repair import p7_pooled_oos as P7  # noqa: E402
from if_repair import confirm_mseries as CM  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FG, R = os.path.join(HERE, "figs"), os.path.join(HERE, "results")

DEV = ("G", "H", "I")
ROLE = {"G": "dev", "H": "dev", "I": "dev", "J": "discovery", "K": "oos", "L": "oos"}


def blocks():
    """-> {label: (px, pg, o)} for every draw, plus the M sub-draws."""
    cfg = P7.CONFIGS["relatif_C5"]
    est, base = cfg["scores"](), P7._graddot("cached")["C5"]
    out = {}
    for dr in ("G", "H", "I", "J", "K", "L"):
        ms = P7.masks_for(dr)
        obs = F.seed_mean(P7.raw_outcomes(dr, "C5"))
        v = np.array([obs.get(m["mask_id"], np.nan) for m in ms])
        k = np.isfinite(v)
        out[dr] = (P7.mask_pred(est, ms)[k], P7.mask_pred(base, ms)[k], v[k])
    ms = CM.m_masks()
    obs = F.seed_mean(F.campaign_outcomes("M", "plain", targets=("C5",))["C5"])
    v = np.array([obs.get(m["mask_id"], np.nan) for m in ms])
    for d in range(6):
        sel = np.array([m["subdraw"] == f"M{d}" for m in ms]) & np.isfinite(v)
        out[f"M{d}"] = (P7.mask_pred(est, ms)[sel], P7.mask_pred(base, ms)[sel], v[sel])
    return out


def row(px, pg, o, fn, strata=None, n_boot=4000, seed=7):
    strata = np.zeros(len(o)) if strata is None else strata
    return CM.stratified_bootstrap(px, pg, o, strata, fn, n_boot=n_boot, seed=seed)


def build(fn):
    B = blocks()
    rows = []
    for dr in ("G", "H", "I", "J", "K", "L"):
        d0, p, lo, hi = row(*B[dr], fn)
        rows.append({"label": f"{dr}  ({ROLE[dr]})", "role": ROLE[dr], "d": d0,
                     "lo": lo, "hi": hi, "n": len(B[dr][2])})
    for d in range(6):
        k = f"M{d}"
        d0, p, lo, hi = row(*B[k], fn)
        rows.append({"label": f"{k}  (oos, depth 2)", "role": "oos_m", "d": d0,
                     "lo": lo, "hi": hi, "n": len(B[k][2])})
    # pooled rows
    def pooled(keys, label, role):
        px = np.concatenate([B[k][0] for k in keys])
        pg = np.concatenate([B[k][1] for k in keys])
        o = np.concatenate([B[k][2] for k in keys])
        st = np.concatenate([[k] * len(B[k][2]) for k in keys])
        d0, p, lo, hi = row(px, pg, o, fn, st)
        rows.append({"label": label, "role": role, "d": d0, "lo": lo, "hi": hi, "n": len(o),
                     "p": p})
    pooled(["K", "L"], "pooled K+L  (pre-M, n=48)", "pool")
    pooled([f"M{d}" for d in range(6)], "pooled M  (virgin, n=144)", "pool")
    pooled(["K", "L"] + [f"M{d}" for d in range(6)],
           "POOLED ALL CLEAN OOS  (n=192)", "grand")
    return pd.DataFrame(rows)


STYLE = {"dev": ("#7f7f7f", "o"), "discovery": ("#d62728", "D"), "oos": ("#1f77b4", "s"),
         "oos_m": ("#17becf", "s"), "pool": ("black", "*"), "grand": ("#2ca02c", "*")}


def main():
    os.makedirs(FG, exist_ok=True)
    panels = [("kendall_tau_b", "Kendall tau_b  (primary)"),
              ("spearman", "Spearman  (secondary, comparable to all history)")]
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), sharey=True)
    for ax, (sname, title) in zip(axes, panels):
        df = build(CM.STATS[sname])
        df.to_csv(os.path.join(R, f"p7_forest_final_{sname}.csv"), index=False)
        ys = np.arange(len(df))[::-1]
        for y, (_, r) in zip(ys, df.iterrows()):
            c, m = STYLE[r.role]
            lw = 2.6 if r.role in ("pool", "grand") else 1.6
            ms = 16 if r.role in ("pool", "grand") else 7
            ax.plot([r.lo, r.hi], [y, y], color=c, lw=lw, solid_capstyle="butt")
            ax.plot([r.d], [y], m, color=c, ms=ms)
        ax.axvline(0, color="black", lw=1)
        ax.axvline(0.341, color="#d62728", ls=":", lw=1.4)
        ax.text(0.341, len(df) - 0.2, " campaign J\n claimed +0.341", color="#d62728", fontsize=7,
                va="top")
        ax.set_yticks(ys)
        ax.set_yticklabels(df.label, fontsize=8)
        ax.set_xlabel("paired  $\\Delta\\rho$  vs GradDot_dmean")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.25)
        ax.set_ylim(-0.8, len(df) - 0.2)
    fig.suptitle("RelatIF (K/G_dd) on C5 vs GradDot_dmean, every mask draw in the project\n"
                 "grey = draws the config was selected on   red = the draw it was reported on   "
                 "blue/cyan = clean out-of-sample   95% CI: mask bootstrap stratified within draw",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(FG, "p7_forest_final.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print("wrote", out)
    print(build(CM.STATS["kendall_tau_b"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
