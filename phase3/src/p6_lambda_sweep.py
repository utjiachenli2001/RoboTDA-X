"""P6.5 -- the IF/TRAK ridge sweep. Closes audit MAJOR-2 ("untuned lambda").

Phase 2's headline is a NULL: demo-grain attribution reaches only 0.40 of the seed-ensembled
ceiling. The audit's objection: the ridge lambda was never tuned, so "attribution was given its
best shot" is an assumption, not a measurement. This stage MEASURES it.

THE KEY ECONOMY -- why this costs 0 retrains and one gradient pass:

    IF and TRAK are both of the form  (G + c*lam*I)^{-1} K  for the SAME
        G = PHI PHI^T   (N x N Gram of the per-demo training gradients)
        K = PHI TG^T    (N x T train-vs-test gradient dots)
    NEITHER G NOR K DEPENDS ON lambda. Only the solve does. So we compute G and K ONCE per
    ensemble member (the expensive part: 135 per-demo gradients of a 19.2M-param model), cache
    them, and then sweep the ENTIRE lambda grid in closed form for free.

    TracIn has no ridge at all, so it is invariant to lambda by construction. It is reported
    unchanged as a control -- if TracIn's LDS moved, this script would be wrong.

SELF-VALIDATION: at the default ridge_rel = 1e-2 the recomputed scores must reproduce the
archived results/influence_table.parquet. That is asserted, not assumed.

PREREGISTERED READ-OUT (preregistration_phase3.json, P6_5):
    the MAX over the lambda grid of demo-grain LDS/ceiling on the focal targets C1/C5,
    plus a MANDATORY cross-validated check (tune on C1 -> evaluate frozen on C5, and vice
    versa). Per-target tuned maxima are an UPPER BOUND, labelled oracle-tuned, never a headline.
"""
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3lib as L
from p3lib import P3_RESULTS, RESULTS, P2_RESULTS, RUNS

sys.path.insert(0, os.path.join(L.ROOT, "src"))
import dataset  # noqa: E402
import evaluate as EV  # noqa: E402
import attribution as AT  # noqa: E402
from lds import spearman, spearman_p_onesided, bootstrap_spearman_ci  # noqa: E402

# ---- PREREGISTERED grid (preregistration_phase3.json: P6_5.ridge_rel_grid)
RIDGE_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
DEFAULT_RIDGE = 1e-2
FOCAL = ["C1", "C5"]
ATTRS_RIDGE = ["IF", "TRAK"]
ALPHA = 0.025
GK_CACHE = os.path.join(P3_RESULTS, "p6_gram_cache.npz")


# ---------------------------------------------------------------- gradient pass (GPU)
def build_gram_cache(force=False):
    """One pass over the E=10 Stage-E ensemble. Caches G (N,N) and K (N,T) per member."""
    if os.path.exists(GK_CACHE) and not force:
        print(f"[6.5] Gram cache exists -> {GK_CACHE}")
        return np.load(GK_CACHE, allow_pickle=True)

    ens = sorted(glob.glob(os.path.join(RUNS, "stage_E", "ens_s*")))
    ens = [d for d in ens if os.path.exists(os.path.join(d, "final.pt"))]
    assert len(ens) == 10, f"expected 10 Stage-E members, got {len(ens)}"

    # P6.2 GUARD: refuse attribution if any member was trained on the probe/test demos.
    heldout = set(dataset.heldout_pool()[0])
    L.assert_no_probe_leak(ens, heldout, context="P6.5 lambda sweep, test side = held-out pool")
    print(f"[6.5] probe-leak guard PASSED for {len(ens)} ensemble members")

    train_ids, _ = dataset.train_pool()
    tbank = dataset.Bank(train_ids)
    hbank = EV.heldout_bank("base")
    slices = tbank.demo_slices()
    N = len(train_ids)
    clusters = dataset.clusters()
    targets = [(c, "plain") for c in clusters]          # functional 'plain' matches the L2 outcome

    Gs, Ks, members = [], [], []
    t0 = time.time()
    for run in ens:
        m = os.path.basename(run)
        model = AT.load_ckpt_model(os.path.join(run, "final.pt"))
        tg, order = AT.build_targets(model, hbank, "base", targets)
        TG = torch.stack([tg[k] for k in order])                       # (T, p)
        PHI = torch.empty((N, TG.shape[1]), dtype=torch.float32, device="cuda")
        for i, d in enumerate(train_ids):
            PHI[i] = AT.demo_gradient(model, tbank, slices[d])
        G = (PHI @ PHI.T).double().cpu().numpy()                       # (N,N)
        K = (PHI @ TG.T).double().cpu().numpy()                        # (N,T)
        Gs.append(G)
        Ks.append(K)
        members.append(m)
        del model, PHI, TG, tg
        torch.cuda.empty_cache()
        print(f"[6.5] Gram {m}: G{G.shape} K{K.shape} ({time.time()-t0:.0f}s)", flush=True)

    np.savez(GK_CACHE, G=np.stack(Gs), K=np.stack(Ks), members=np.array(members),
             train_ids=np.array(train_ids), targets=np.array([c for c, _ in order]))
    print(f"[6.5] cached -> {GK_CACHE} ({time.time()-t0:.0f}s)")
    return np.load(GK_CACHE, allow_pickle=True)


# ---------------------------------------------------------------- closed-form lambda sweep
def scores_at_ridge(Z, ridge_rel):
    """-> {attr: {target: {demo_id: score}}}, ensemble-mean over members, at this ridge."""
    G, K = Z["G"], Z["K"]                                 # (M,N,N), (M,N,T)
    train_ids = list(Z["train_ids"])
    tgts = list(Z["targets"])
    M, N, T = K.shape
    acc = {a: np.zeros((N, T)) for a in ATTRS_RIDGE}
    for m in range(M):
        Gm, Km = G[m], K[m]
        lam = ridge_rel * float(np.mean(np.diag(Gm)))     # same adaptive scaling as attribution.py
        I = np.eye(N)
        acc["TRAK"] += np.linalg.solve(Gm + lam * I, Km)
        inner = np.linalg.solve(lam * N * I + Gm, Km)     # Woodbury empirical-Fisher IF
        acc["IF"] += (Km - Gm @ inner) / lam
    out = {}
    for a in ATTRS_RIDGE:
        S = acc[a] / M
        out[a] = {tgts[j]: {train_ids[i]: float(S[i, j]) for i in range(N)}
                  for j in range(T)}
    return out


# ---------------------------------------------------------------- LDS machinery
def demo_grain_lds(scores_by_demo, gmasks, obs):
    pred = np.array([sum(scores_by_demo.get(d, 0.0) for d in m["demos"]) for m in gmasks])
    out = np.array([obs.get(m["mask_id"], np.nan) for m in gmasks])
    ok = np.isfinite(out)
    rho = spearman(pred[ok], out[ok])
    return rho, spearman_p_onesided(rho, int(ok.sum())), int(ok.sum()), pred[ok], out[ok]


def cluster_grain_lds(scores_by_demo, fmasks, obs, target):
    """CONDITIONAL LDS: only masks that INCLUDE the target cluster (Phase-1 primary)."""
    pred, out = [], []
    for m in fmasks:
        if target not in m["clusters"]:
            continue
        v = obs.get(m["mask_id"])
        if v is None or not np.isfinite(v):
            continue
        pred.append(sum(scores_by_demo.get(d, 0.0) for d in m["demos"]))
        out.append(v)
    rho = spearman(pred, out)
    return rho, spearman_p_onesided(rho, len(pred)), len(pred)


def main():
    Z = build_gram_cache()

    # ---------------- ground truth (marker-gated artifacts, reused unchanged)
    gman = json.load(open(os.path.join(RESULTS, "demo_mask_manifest.json")))["masks"]
    fman = json.load(open(os.path.join(RESULTS, "mask_manifest.json")))
    fmasks = [{"mask_id": m["mask_id"], "demos": m["demos"], "clusters": m["clusters"]}
              for m in fman["masks"]]

    G6 = pd.read_parquet(os.path.join(P2_RESULTS, "stage_G6_outcomes.parquet"))
    dfF = pd.read_parquet(os.path.join(RESULTS, "stage_F_outcomes.parquet"))
    dfF["neg_plain_loss"] = -dfF.plain_loss

    p1 = json.load(open(os.path.join(P2_RESULTS, "p1_demo_grain.json")))
    ceil_demo = {t: p1["all_targets"][t]["neg_plain_loss"]["ceiling_6seed_SB"]
                 for t in p1["all_targets"]}
    nc = json.load(open(os.path.join(RESULTS, "noise_ceilings.json")))
    ceil_clu = {t: nc[t]["neg_plain_loss"]["ceiling"] for t in nc}

    clusters = dataset.clusters()
    # demo-grain observed: 6-seed mean of neg_plain_loss per (target, mask)
    obs_demo = {t: G6[G6.target == t].groupby("mask_id")["neg_plain_loss"].mean().to_dict()
                for t in clusters}
    # cluster-grain observed: 2-seed mean (Phase-1 F_SEEDS = 301,302)
    F2 = dfF[dfF.seed.isin([301, 302])]
    obs_clu = {t: F2[F2.target == t].groupby("mask_id")["neg_plain_loss"].mean().to_dict()
               for t in clusters}

    # ---------------- SELF-VALIDATION: default ridge must reproduce the archived table
    S_def = scores_at_ridge(Z, DEFAULT_RIDGE)
    arch = pd.read_parquet(os.path.join(RESULTS, "influence_table.parquet"))
    arch = arch[arch.functional == "plain"]
    repro = {}
    for a in ATTRS_RIDGE:
        rs, ds = [], []
        for t in clusters:
            sub = arch[(arch.attributor == a) & (arch.target == t)]
            old = dict(zip(sub.demo_id, sub.score))
            new = S_def[a][t]
            ids = sorted(old)
            o = np.array([old[d] for d in ids])
            n = np.array([new[d] for d in ids])
            rs.append(spearman(o, n))
            denom = max(np.abs(o).max(), 1e-30)
            ds.append(float(np.abs(o - n).max() / denom))
        repro[a] = {"min_spearman_vs_archived": float(np.min(rs)),
                    "max_rel_abs_diff": float(np.max(ds))}
        print(f"[6.5] reproduction @ default ridge {a}: "
              f"min rho vs archived = {np.min(rs):.6f}, max rel|diff| = {np.max(ds):.2e}")
    if min(repro[a]["min_spearman_vs_archived"] for a in ATTRS_RIDGE) < 0.999:
        raise RuntimeError("P6.5 self-validation FAILED: recomputation at the default ridge does "
                           "not reproduce results/influence_table.parquet. INSTRUMENT DEFECT.")

    # ---------------- the sweep
    rows = []
    for rr in RIDGE_GRID:
        S = scores_at_ridge(Z, rr)
        for a in ATTRS_RIDGE:
            for t in clusters:
                rho_d, p_d, n_d, _, _ = demo_grain_lds(S[a][t], gman, obs_demo[t])
                rho_c, p_c, n_c = cluster_grain_lds(S[a][t], fmasks, obs_clu[t], t)
                rows.append({
                    "ridge_rel": rr, "attributor": a, "target": t, "focal": t in FOCAL,
                    "demo_lds": rho_d, "demo_p1": p_d, "demo_n": n_d,
                    "demo_ceiling": ceil_demo[t],
                    "demo_ratio": rho_d / ceil_demo[t] if ceil_demo[t] > 0 else np.nan,
                    "demo_bar_half_ceiling": 0.5 * ceil_demo[t],
                    "demo_PASS": bool(np.isfinite(rho_d) and ceil_demo[t] > 0
                                      and rho_d >= 0.5 * ceil_demo[t]
                                      and np.isfinite(p_d) and p_d < ALPHA),
                    "cluster_lds": rho_c, "cluster_p1": p_c, "cluster_n": n_c,
                    "cluster_ceiling": ceil_clu[t],
                    "cluster_ratio": rho_c / ceil_clu[t] if ceil_clu[t] > 0 else np.nan,
                })
        print(f"[6.5] ridge_rel={rr:.0e} done", flush=True)

    # TracIn control: ridge-invariant by construction, read from the archived table
    for t in clusters:
        sub = arch[(arch.attributor == "TracIn") & (arch.target == t)]
        sc = dict(zip(sub.demo_id, sub.score))
        rho_d, p_d, n_d, _, _ = demo_grain_lds(sc, gman, obs_demo[t])
        rho_c, p_c, n_c = cluster_grain_lds(sc, fmasks, obs_clu[t], t)
        rows.append({"ridge_rel": np.nan, "attributor": "TracIn", "target": t,
                     "focal": t in FOCAL, "demo_lds": rho_d, "demo_p1": p_d, "demo_n": n_d,
                     "demo_ceiling": ceil_demo[t],
                     "demo_ratio": rho_d / ceil_demo[t], "demo_bar_half_ceiling": 0.5*ceil_demo[t],
                     "demo_PASS": bool(np.isfinite(rho_d) and rho_d >= 0.5*ceil_demo[t]
                                       and np.isfinite(p_d) and p_d < ALPHA),
                     "cluster_lds": rho_c, "cluster_p1": p_c, "cluster_n": n_c,
                     "cluster_ceiling": ceil_clu[t],
                     "cluster_ratio": rho_c / ceil_clu[t] if ceil_clu[t] > 0 else np.nan})

    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(P3_RESULTS, "p6_lambda_sweep.csv"), index=False)

    # ---------------- PREREGISTERED READ-OUT: max over lambda, focal targets
    R = T[T.attributor.isin(ATTRS_RIDGE)]
    tuned = {}
    for t in FOCAL:
        s = R[R.target == t]
        i = s.demo_lds.idxmax()
        b = s.loc[i]
        tuned[t] = {
            "oracle_tuned_best_ridge_rel": float(b.ridge_rel),
            "oracle_tuned_best_attributor": str(b.attributor),
            "oracle_tuned_best_demo_lds": float(b.demo_lds),
            "ceiling_6seed": float(b.demo_ceiling),
            "bar_half_ceiling": float(b.demo_bar_half_ceiling),
            "oracle_tuned_ratio": float(b.demo_ratio),
            "oracle_tuned_p_onesided": float(b.demo_p1),
            "oracle_tuned_CROSSES_HALF_CEILING": bool(b.demo_ratio >= 0.5),
            "oracle_tuned_PASS_ratio_AND_p": bool(b.demo_PASS),
            "LABEL": "ORACLE-TUNED UPPER BOUND -- lambda chosen on this same target. NOT held-out.",
        }

    # ---------------- MANDATORY CROSS-VALIDATION (tune on one focal, evaluate on the other)
    cv = {}
    for tune_on, eval_on in [("C1", "C5"), ("C5", "C1")]:
        s = R[R.target == tune_on]
        i = s.demo_lds.idxmax()
        rr, at = float(s.loc[i].ridge_rel), str(s.loc[i].attributor)
        e = R[(R.target == eval_on) & (R.ridge_rel == rr) & (R.attributor == at)].iloc[0]
        cv[f"tune_on_{tune_on}__evaluate_on_{eval_on}"] = {
            "frozen_ridge_rel": rr, "frozen_attributor": at,
            "heldout_demo_lds": float(e.demo_lds),
            "heldout_ceiling": float(e.demo_ceiling),
            "heldout_ratio": float(e.demo_ratio),
            "heldout_bar_half_ceiling": float(e.demo_bar_half_ceiling),
            "heldout_p_onesided": float(e.demo_p1),
            "heldout_CROSSES_HALF_CEILING": bool(e.demo_ratio >= 0.5),
            "heldout_PASS_ratio_AND_p": bool(e.demo_PASS),
            "LABEL": "CROSS-VALIDATED -- lambda and attributor frozen on the OTHER focal target.",
        }

    any_tuned = any(v["oracle_tuned_CROSSES_HALF_CEILING"] for v in tuned.values())
    any_cv = any(v["heldout_PASS_ratio_AND_p"] for v in cv.values())
    default_row = {t: float(R[(R.target == t) & (R.ridge_rel == DEFAULT_RIDGE)].demo_lds.max())
                   for t in FOCAL}

    out = {
        "stage": "P6.5 IF/TRAK ridge sweep",
        "closes": "audit MAJOR-2 (untuned lambda)",
        "n_retrains": 0,
        "method": ("G = PHI PHI^T and K = PHI TG^T do not depend on lambda, so ONE gradient pass "
                   "over the E=10 ensemble yields the EXACT IF and TRAK scores at every lambda "
                   "in the grid in closed form."),
        "ridge_rel_grid": RIDGE_GRID,
        "default_ridge_rel": DEFAULT_RIDGE,
        "grid_span_note": ("The grid spans 1e-6..1e+1, i.e. 1e-4x .. 1e+3x the default 1e-2 -- "
                           "strictly containing the 1e-4..1e+1 x default span the brief asks for."),
        "self_validation_reproduces_archived_table_at_default_ridge": repro,
        "ground_truth_demo": ("phase2/results/stage_G6_outcomes.parquet, 6-seed mean, "
                              "neg_plain_loss (the Phase-2 P1 primary), n=24 masks"),
        "ceilings_reused_unchanged": {"demo_6seed_SB": ceil_demo, "cluster": ceil_clu},
        "PREREGISTERED_READOUT_oracle_tuned_max_over_lambda": tuned,
        "MANDATORY_CROSS_VALIDATION": cv,
        "demo_lds_at_default_ridge": default_row,
        "TracIn_control": "ridge-invariant by construction; reported unchanged in the CSV.",
        "VERDICT": None,
        "VERDICT_TEXT": None,
    }
    if not any_tuned:
        out["VERDICT"] = "PHASE_2_CONCLUSION_STANDS"
        out["VERDICT_TEXT"] = (
            "No ridge in an 8-point log grid spanning 1e-6..1e+1 lifts demo-grain LDS to half of "
            "the seed-ensembled ceiling on either focal target. Even the ORACLE-TUNED maximum "
            "(lambda chosen on the target it is evaluated on -- an upper bound no honest method "
            "could achieve) stays below the bar. The 'attribution was given its best shot' claim "
            "is now a MEASURED statement, not an assumption. Phase 2's headline is unchanged and "
            "strengthened.")
    elif not any_cv:
        out["VERDICT"] = "ORACLE_TUNED_ONLY__NOT_HELD_OUT"
        out["VERDICT_TEXT"] = (
            "An oracle-tuned lambda crosses half-ceiling on at least one focal target, but the "
            "CROSS-VALIDATED (held-out) lambda does NOT. Tuning lambda on the target it is scored "
            "against is not a method; it is a multiple comparison over 8 grid points. Phase 2's "
            "conclusion stands, and the oracle-tuned value is reported as an upper bound.")
    else:
        out["VERDICT"] = "MATERIAL_UPDATE_TO_PHASE_2"
        out["VERDICT_TEXT"] = (
            "A CROSS-VALIDATED ridge (tuned on one focal target, frozen, evaluated on the other) "
            "reaches half of the seed-ensembled ceiling. This is a MATERIAL UPDATE to Phase 2's "
            "headline conclusion and MUST be reported prominently and plainly.")

    L.atomic_write_json(os.path.join(P3_RESULTS, "p6_lambda_sweep.json"), out)

    print("\n" + "=" * 92)
    print("P6.5 -- RIDGE SWEEP (demo grain, held-out L2, 6-seed ground truth)")
    print("=" * 92)
    print(f"{'target':7s} {'ridge':>8s} {'attr':>6s} {'LDS':>8s} {'ceiling':>8s} {'bar':>7s} "
          f"{'ratio':>6s} {'p1':>7s}")
    for t in FOCAL:
        for rr in RIDGE_GRID:
            s = R[(R.target == t) & (R.ridge_rel == rr)]
            b = s.loc[s.demo_lds.idxmax()]
            star = " <-- DEFAULT" if rr == DEFAULT_RIDGE else ""
            print(f"{t:7s} {rr:8.0e} {b.attributor:>6s} {b.demo_lds:+8.3f} {b.demo_ceiling:8.3f} "
                  f"{b.demo_bar_half_ceiling:7.3f} {b.demo_ratio:6.2f} {b.demo_p1:7.4f}{star}")
        print("-" * 92)
    print("\nORACLE-TUNED MAXIMUM (upper bound, not held-out):")
    for t, v in tuned.items():
        print(f"  {t}: ridge={v['oracle_tuned_best_ridge_rel']:.0e} "
              f"{v['oracle_tuned_best_attributor']:6s} LDS={v['oracle_tuned_best_demo_lds']:+.3f} "
              f"ratio={v['oracle_tuned_ratio']:.2f} p={v['oracle_tuned_p_onesided']:.4f} "
              f"crosses_half={v['oracle_tuned_CROSSES_HALF_CEILING']}")
    print("\nCROSS-VALIDATED (the number that licenses a claim):")
    for k, v in cv.items():
        print(f"  {k}: ridge={v['frozen_ridge_rel']:.0e} {v['frozen_attributor']:6s} "
              f"LDS={v['heldout_demo_lds']:+.3f} ratio={v['heldout_ratio']:.2f} "
              f"p={v['heldout_p_onesided']:.4f} PASS={v['heldout_PASS_ratio_AND_p']}")
    print("=" * 92)
    print(f"VERDICT: {out['VERDICT']}")
    print(out["VERDICT_TEXT"])
    print("=" * 92)


if __name__ == "__main__":
    main()
