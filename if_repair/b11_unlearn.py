"""W1 -- unlearning-LOO: a DYNAMIC counterfactual for demo removal (the headline attempt).

Every gradient estimator so far is linearized: it reads a dot product at the converged weights and
inherits Phi's run-to-run noise (G_rel_fro 0.879). The outcomes, by contrast, are stable
(Spearman 0.61-0.93 across regenerations). The hypothesis: a DYNAMIC counterfactual -- actually
perturbing the trained model to approximate "this demo was removed" and measuring the held-out
effect -- escapes both the rank-135 kernel ceiling and the Phi noise, because it never forms Phi.

Three variants, each scoring demo d by its effect on target cluster t's held-out L2:

  A ascent      from final.pt, k gradient-ASCENT steps on d's frames (raise NLL on d = "forget"),
                lr eta. score = L2_after - L2_before  (unlearning d hurts t => d was helpful).
  B finetune    k descent steps on the training set MINUS d (a truncated warm-start LOO).
                score = -(L2_after - L2_before) = L2_before - L2_after (removal hurting t => +).
  C scrub       ascent on d interleaved with a replay descent batch of the rest (SCRUB-lite).

score_d,t averaged over the regenerated E=5 members. Scores never touch mask outcomes => direct
LDS. Baseline is GradDot on the SAME regen E=5 ensemble (BLOCKERS #6). Grid k in {10,50,200},
eta in {1x, 0.1x} training lr; all k recorded from one run.
"""
from __future__ import annotations

import argparse
import copy
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402
from if_repair import functionals as F  # noqa: E402

D.add_repo_paths()
import dataset  # noqa: E402
import evaluate as EV  # noqa: E402
import train as TR  # noqa: E402
from if_repair import gradients as GR  # noqa: E402
from if_repair import retrain as RT  # noqa: E402
from lds import spearman  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
DEV = "cuda"


def to_gpu(bank):
    return (torch.from_numpy(bank.S).to(DEV), torch.from_numpy(bank.L).to(DEV),
            torch.from_numpy(bank.A).to(DEV))


def heldout_rows(targets):
    fidx = RT.frame_index()
    cor = fidx["cluster_of_row"]
    return {t: torch.from_numpy(np.nonzero(cor == t)[0]).to(DEV) for t in targets}


@torch.no_grad()
def l2_targets(model, HS, HL, HA, rows_t):
    out = {}
    for t, r in rows_t.items():
        v = model.l2(HS[r], HL[r], HA[r])
        out[t] = float(v.mean())
    return out


def member_scores(member, targets, ksteps, etas, variants, cfg, train_gpu, tslices,
                  held, rows_t, seed=0):
    S, L, A = train_gpu
    HS, HL, HA = held
    base_lr = cfg["lr"]
    model = EV.load_model(os.path.join(GR.REGEN, member, "final.pt"), device=DEV)
    orig = copy.deepcopy(model.state_dict())
    base = l2_targets(model, HS, HL, HA, rows_t)
    N = S.shape[0]
    all_rows = np.arange(N)
    kmax = max(ksteps)
    rng = np.random.default_rng(seed)
    # scores[(variant,eta,k)][t][demo]
    scores = {(v, e, k): {t: {} for t in targets}
              for v in variants for e in etas for k in ksteps}

    demos = list(tslices.keys())
    for di, d in enumerate(demos):
        drows = torch.from_numpy(tslices[d]).to(DEV)
        non_d = np.setdiff1d(all_rows, tslices[d])
        for e in etas:
            lr = e * base_lr
            for v in variants:
                model.load_state_dict(orig)
                opt = torch.optim.SGD(model.parameters(), lr=lr)
                model.train()
                for step in range(1, kmax + 1):
                    opt.zero_grad(set_to_none=True)
                    if v == "A":
                        loss = -model.nll(S[drows], L[drows], A[drows]).mean()   # ascent on d
                    elif v == "B":
                        b = torch.from_numpy(rng.choice(non_d, min(256, len(non_d)),
                                                        replace=False)).to(DEV)
                        loss = model.nll(S[b], L[b], A[b]).mean()                # descent, no d
                    else:  # C scrub-lite: ascent on d + replay descent on rest
                        b = torch.from_numpy(rng.choice(non_d, min(256, len(non_d)),
                                                        replace=False)).to(DEV)
                        loss = (-model.nll(S[drows], L[drows], A[drows]).mean()
                                + model.nll(S[b], L[b], A[b]).mean())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                    opt.step()
                    if step in ksteps:
                        model.eval()
                        aft = l2_targets(model, HS, HL, HA, rows_t)
                        for t in targets:
                            delta = aft[t] - base[t]
                            scores[(v, e, step)][t][d] = delta if v == "A" else -delta
                        model.train()
        if (di + 1) % 30 == 0:
            print(f"    [{member}] {di+1}/{len(demos)} demos", flush=True)
    model.load_state_dict(orig)
    del model
    torch.cuda.empty_cache()
    return scores, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--variants", default="A,B,C")
    ap.add_argument("--etas", default="1.0,0.1")
    ap.add_argument("--ksteps", default="10,50,200")
    ap.add_argument("--targets", default="C1,C5,C2")
    ap.add_argument("--tag", default="full")
    a = ap.parse_args()
    variants = a.variants.split(",")
    etas = [float(x) for x in a.etas.split(",")]
    ksteps = [int(x) for x in a.ksteps.split(",")]
    targets = a.targets.split(",")
    t0 = time.time()

    cfg = TR.load_cfg()
    members = sorted(os.path.basename(x) for x in glob.glob(os.path.join(GR.REGEN, "ens_s*"))
                     if os.path.exists(os.path.join(x, "final.pt")))[:a.members]
    print(f"[W1] {len(members)} members, variants={variants}, etas={etas}, k={ksteps}, "
          f"targets={targets}", flush=True)

    train_ids, _ = dataset.train_pool()
    tbank = dataset.Bank(train_ids)
    tslices = tbank.demo_slices()
    train_gpu = to_gpu(tbank)
    hbank = EV.heldout_bank("base")
    held = to_gpu(hbank)
    rows_t = heldout_rows(targets)

    agg = None
    for m in members:
        sc, base = member_scores(m, targets, ksteps, etas, variants, cfg, train_gpu, tslices,
                                 held, rows_t)
        if agg is None:
            agg = {key: {t: {d: 0.0 for d in sc[key][t]} for t in targets} for key in sc}
        for key in sc:
            for t in targets:
                for d, val in sc[key][t].items():
                    agg[key][t][d] += val / len(members)
        print(f"[W1] member {m} done ({(time.time()-t0)/60:.1f} min)", flush=True)

    # save raw scores
    os.makedirs(RESULTS, exist_ok=True)
    np.savez_compressed(os.path.join(HERE, "runs", f"b11_unlearn_scores_{a.tag}.npz"),
                        scores=np.array([{
                            "key": key, "target": t,
                            "scores": agg[key][t]} for key in agg for t in targets], dtype=object))

    # ---- LDS on G-series + paired vs GradDot(regenE5) pooled G/H/I
    from p6_lambda_sweep import demo_grain_lds
    from if_repair import b1_layerwise as B1
    from p6_lambda_extend import scores_graddot
    ens = B1.build_ensemble(members)
    graddot = scores_graddot(ens["ALL"], normalize_per_member=True)

    gm = D.demo_masks()
    obs, ceil = D.outcomes("bc_s10"), D.ceilings("bc_s10")
    rows = []
    for key in agg:
        v, e, k = key
        for t in targets:
            rho, p, n, _, _ = demo_grain_lds(agg[key][t], gm, obs[t])
            rows.append({"variant": v, "eta": e, "k": k, "target": t, "estimator": "unlearn",
                         "lds": rho, "ceiling": ceil[t], "ratio": rho / ceil[t], "p": p})
    for t in targets:
        rho, p, n, _, _ = demo_grain_lds(graddot[t], gm, obs[t])
        rows.append({"variant": "-", "eta": np.nan, "k": np.nan, "target": t,
                     "estimator": "GradDot_ALL_regenE5", "lds": rho, "ceiling": ceil[t],
                     "ratio": rho / ceil[t], "p": p})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, f"b11_unlearn_gseries_{a.tag}.csv"), index=False)
    print("\n" + "=" * 96)
    print(f"W1 -- unlearning-LOO, G-series LDS (regenE5 ensemble, archived p12 outcomes) [{a.tag}]")
    print("=" * 96)
    print(df.round(4).to_string(index=False))

    # kill-rule check: best cell paired delta vs GradDot on G-series
    print("\nKILL-RULE (paired Delta vs GradDot on this draw; W1 survives if >= +0.10 on C1 or C5):")
    for t in targets:
        g = df[(df.estimator == "GradDot_ALL_regenE5") & (df.target == t)].lds.iloc[0]
        best = df[(df.estimator == "unlearn") & (df.target == t)].sort_values("lds").iloc[-1]
        print(f"  {t}: best unlearn cell = {best.variant}/eta{best.eta}/k{int(best.k)} "
              f"lds={best.lds:.4f}  vs GradDot {g:.4f}  (delta {best.lds-g:+.4f})")
    print(f"\n[W1] total {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
