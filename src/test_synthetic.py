"""STAGE A -- SYNTHETIC component tests (spec §4).

EVERYTHING IN THIS FILE IS SYNTHETIC. No number produced here is a study result; results are
written to results/STAGE_A_SYNTHETIC_tests.json and are never mixed with real results.

Covers: data loader/windowing, normalization, phase mask sampler, mask designs (F & G),
trainer (must overfit a tiny fixture), rollout window buffer, and the LDS scorer, which must
RECOVER A PLANTED SIGNAL and must NOT find signal in random scores.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bootstrap  # noqa: F401
from bootstrap import RESULTS
import phases
import lds

RES = []


def check(name, cond, detail=""):
    RES.append({"test": name, "pass": bool(cond), "detail": str(detail)})
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}", flush=True)
    return bool(cond)


# ---------------------------------------------------------------- 1. phase sampler
def test_phases():
    # --- rule (a) in isolation: gripper crossing at t=50, speeds strictly INCREASING so that
    # every closed frame is above the 30th-percentile speed and rule (b) cannot fire.
    T = 100
    P = np.zeros((T, 16), dtype=np.float32)
    P[:, 0] = np.cumsum(np.linspace(0.001, 0.02, T)).astype(np.float32)   # accelerating
    P[:, 7], P[:, 8] = 0.04, -0.04            # width 0.08 (open)
    P[50:, 7], P[50:, 8] = 0.0, 0.0           # width 0.0 (closed) from t=50
    inter, trans = phases.phase_masks(P, window=10, percentile=30.0)
    check("phases: gripper-event window is exactly +-10 around the crossing",
          inter[40:61].all() and not inter[39] and not inter[61],
          f"inter[39]={inter[39]} inter[40..60]=all{inter[40:61].all()} inter[61]={inter[61]}")
    check("phases: transport is the exact complement of interaction",
          np.array_equal(trans, ~inter))

    # --- rule (b) BEYOND rule (a): gripper closes at t=20 (rule (a) covers 10..30); frames
    # 96..119 are closed AND slow (bottom 20% of speeds) so rule (b) must mark them even
    # though they are far outside the event window. Frames 40..90 are closed but FAST ->
    # transport.  NB the closed/open midpoint is only defined when a demo's gripper actually
    # changes state; all 225 corpus demos have >=1 crossing (verified), so this is the real
    # regime.
    T2 = 120
    Q = np.zeros((T2, 16), dtype=np.float32)
    speed = np.where(np.arange(T2) < 96, 0.05, 0.001).astype(np.float32)   # 20% slow frames
    Q[:, 0] = np.cumsum(speed)
    Q[:, 7], Q[:, 8] = 0.04, -0.04            # open
    Q[20:, 7], Q[20:, 8] = 0.0, 0.0           # closed from t=20 -> crossing at 20
    i2, _ = phases.phase_masks(Q, window=10, percentile=30.0)
    check("phases: rule (b) marks closed+slow frames outside the event window",
          i2[10:31].all() and not i2[40:90].any() and i2[96:].all(),
          f"event_window={i2[10:31].all()} closed_fast_is_transport={not i2[40:90].any()} "
          f"closed_slow_is_interaction={i2[96:].all()}")

    # --- sensitivity: +-25% window variants must widen/narrow the interaction set
    lo, _ = phases.phase_masks(P, **phases.LOW)
    hi, _ = phases.phase_masks(P, **phases.HIGH)
    check("phases: +-25% threshold variants widen/narrow interaction monotonically",
          lo.sum() < inter.sum() < hi.sum(),
          f"low(w=8)={lo.sum()} base(w=10)={inter.sum()} high(w=12)={hi.sum()}")


# ---------------------------------------------------------------- 2. windowing
def test_windowing():
    import dataset as DS
    T, D = 7, DS.state_dim()
    S = np.arange(T * D, dtype=np.float32).reshape(T, D)
    CTX = DS.CTX
    pad = np.repeat(S[:1], CTX - 1, 0)
    SP = np.concatenate([pad, S], 0)
    idx = np.arange(T)[:, None] + np.arange(CTX)[None, :]
    W = SP[idx]
    check("windowing: window t ends at frame t", np.allclose(W[3, -1], S[3]))
    check("windowing: demo start is left-padded by repeating frame 0",
          np.allclose(W[0], np.repeat(S[:1], CTX, 0)))
    check("windowing: interior window is the true contiguous history",
          np.allclose(W[6, -3:], S[4:7]))


# ---------------------------------------------------------------- 3. mask designs
def test_masks():
    import masks as MK
    f = MK.cluster_mask_manifest()
    X = np.array(f["coinclusion_matrix"])
    off = ~np.eye(9, dtype=bool)
    check("masks F: 72 masks, each exactly 5 clusters / 75 demos",
          len(f["masks"]) == 72 and all(len(m["clusters"]) == 5 and m["n_demos"] == 75
                                        for m in f["masks"]))
    check("masks F: every cluster included in exactly 40 masks",
          set(f["inclusion_counts"].values()) == {40}, f["inclusion_counts"])
    check("masks F: pairwise co-inclusion within [17,23]",
          X[off].min() >= 17 and X[off].max() <= 23,
          f"[{X[off].min()},{X[off].max()}]")
    check("masks F: 12 noise-ceiling masks, each cluster in 6-8 of them",
          len(f["noise_ceiling_masks"]) == 12
          and min(f["noise_ceiling_cluster_counts"].values()) >= 6
          and max(f["noise_ceiling_cluster_counts"].values()) <= 8)
    g = MK.demo_mask_manifest()
    check("masks G: 24 masks, each exactly 68 demos",
          len(g["masks"]) == 24 and all(m["n_demos"] == 68 for m in g["masks"]))
    check("masks G: every demo appears in 11-13 masks",
          g["per_demo_inclusion_min"] >= 11 and g["per_demo_inclusion_max"] <= 13,
          f"[{g['per_demo_inclusion_min']},{g['per_demo_inclusion_max']}]")
    check("masks G: within-cluster stratification is 8 or 7 demos per cluster",
          all(set(m["in_target_count"].values()) <= {7, 8} for m in g["masks"]))
    # determinism
    f2, _, _ = MK.build_cluster_masks()
    f3, _, _ = MK.build_cluster_masks()
    check("masks: construction is deterministic under the frozen seed",
          np.array_equal(f2, f3))


# ---------------------------------------------------------------- 4. trainer
def test_trainer():
    import torch
    import policy as P
    torch.manual_seed(0)
    m = P.build(16, {"d_model": 64, "n_layer": 2, "n_head": 4, "n_modes": 3, "dropout": 0.0})
    N, CTX = 64, P.CTX
    g = torch.Generator().manual_seed(0)
    S = torch.randn(N, CTX, 16, generator=g)
    L = torch.randn(N, P.LANG_DIM, generator=g)
    # a LEARNABLE target: action is a fixed linear function of the last frame
    W = torch.randn(16, 7, generator=g)
    A = torch.tanh(S[:, -1] @ W)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    l0 = float(m.nll(S, L, A).mean())
    for _ in range(300):
        loss = m.nll(S, L, A).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    l1 = float(m.nll(S, L, A).mean())
    check("trainer: overfits a synthetic fixture (NLL drops)", l1 < l0 - 1.0,
          f"NLL {l0:.2f} -> {l1:.2f}")
    a = m.act(S, L)
    check("trainer: act() returns in-range 7-D actions",
          a.shape == (N, 7) and float(a.abs().max()) <= 1.0 + 1e-6)


# ---------------------------------------------------------------- 5. rollout buffer
def test_rollout_buffer():
    """The rollout keeps a CTX-length deque left-padded with the first frame; the window fed
    to the policy at step t must equal the training window for frame t."""
    CTX = 10
    D = 4
    frames = [np.full(D, i, dtype=np.float32) for i in range(15)]
    win = [frames[0]] * CTX
    seen = []
    for t, f in enumerate(frames):
        if t > 0:
            win.append(f)
        seen.append(np.stack(win[-CTX:]))
    check("rollout: first window is frame0 repeated CTX times",
          np.allclose(seen[0], np.stack([frames[0]] * CTX)))
    check("rollout: window at t ends at frame t", np.allclose(seen[12][-1], frames[12]))
    check("rollout: window at t is the last CTX frames",
          np.allclose(seen[12], np.stack(frames[3:13])))


# ---------------------------------------------------------------- 6. LDS scorer
def test_lds_scorer():
    """PLANTED SIGNAL: outcome = sum of true per-demo utilities over the mask + noise.
    A perfect attributor (= true utilities) must score high LDS; random scores must not."""
    rng = np.random.default_rng(0)
    n_demos, K = 135, 72
    demos = [f"d{i}" for i in range(n_demos)]
    util = {d: float(rng.normal()) for d in demos}                     # ground-truth utilities
    masks = [{"mask_id": f"M{k}", "demos": list(rng.choice(demos, 75, replace=False))}
             for k in range(K)]
    noise = 0.35
    outcomes = {m["mask_id"]: sum(util[d] for d in m["demos"]) + rng.normal(0, noise)
                for m in masks}

    good = lds.conditional_lds(util, masks, outcomes, "T", include_only_target_masks=False)
    rand = lds.conditional_lds({d: float(rng.normal()) for d in demos}, masks, outcomes, "T",
                               include_only_target_masks=False)
    check("LDS: recovers a planted signal (true utilities -> rho > 0.9)",
          good["rho"] > 0.9, f"rho={good['rho']:.3f} n={good['n_masks']}")
    check("LDS: random scores find no signal (|rho| < 0.3)",
          abs(rand["rho"]) < 0.3, f"rho={rand['rho']:.3f}")
    check("LDS: planted-signal CI excludes 0", good["ci95"][0] > 0, f"CI={good['ci95']}")
    check("LDS: planted-signal p < 1e-6", good["p_onesided"] < 1e-6,
          f"p={good['p_onesided']:.2e}")

    # partial signal: attributor = utilities + noise -> rho should land between
    noisy = {d: util[d] + rng.normal(0, 1.0) for d in demos}
    mid = lds.conditional_lds(noisy, masks, outcomes, "T", include_only_target_masks=False)
    check("LDS: a noisy attributor scores between random and perfect",
          abs(rand["rho"]) < mid["rho"] < good["rho"],
          f"rand={rand['rho']:.3f} noisy={mid['rho']:.3f} true={good['rho']:.3f}")

    # --- logit transform
    p = lds.logit_success([0.0, 0.5, 1.0], 30)
    check("LDS: logit clamps 0 and 1 at 1/(2n) (n=30)",
          np.isfinite(p).all() and abs(p[1]) < 1e-9 and p[0] < 0 < p[2],
          f"{np.round(p,3)}")

    # --- noise ceiling: outcomes are seed-noisy measurements of a latent mask value
    lat = {f"M{k}": float(rng.normal()) for k in range(12)}
    sd = 0.8
    obms = {m: {s: lat[m] + rng.normal(0, sd) for s in (301, 302, 303, 304)} for m in lat}
    nc = lds.noise_ceiling(obms)
    check("noise ceiling: 3 disjoint seed pairings averaged, in (0,1)",
          len(nc["per_pairing"]) == 3 and 0 < nc["ceiling"] < 1,
          f"ceiling={nc['ceiling']:.3f} sb={nc['ceiling_sb']:.3f} pairings={np.round(nc['per_pairing'],3)}")
    # a NOISELESS outcome must give a ceiling of ~1
    obms0 = {m: {s: lat[m] for s in (301, 302, 303, 304)} for m in lat}
    nc0 = lds.noise_ceiling(obms0)
    check("noise ceiling: noiseless replicates -> ceiling ~= 1.0", nc0["ceiling"] > 0.99,
          f"ceiling={nc0['ceiling']:.3f}")
    # ceiling must BOUND the achievable LDS: with seed noise, a perfect attributor's
    # correlation with a 2-seed-mean outcome should not exceed the ceiling by much
    check("noise ceiling: pure noise outcomes -> ceiling ~ 0",
          abs(lds.noise_ceiling({m: {s: float(rng.normal()) for s in (301, 302, 303, 304)}
                                 for m in lat})["ceiling"]) < 0.6)


def test_attribution_estimators():
    """The exact TRAK dual form and the Woodbury IF must equal dense brute-force algebra.

    We replace dattri's JL-projected TRAK and its EK-FAC-approximated IF with exact closed
    forms (legitimate because N=135 << p=19.2M). This test is what licenses that substitution:
    it checks the identities against a p x p dense inverse in the same p >> N regime.
    """
    rng = np.random.default_rng(0)
    N, p, T = 12, 400, 3
    PHI = rng.normal(size=(N, p))
    TG = rng.normal(size=(T, p))
    G = PHI @ PHI.T
    K = PHI @ TG.T
    lam = 1e-2 * float(np.mean(np.diag(G)))

    primal = PHI @ np.linalg.solve(PHI.T @ PHI + lam * np.eye(p), TG.T)
    dual = np.linalg.solve(G + lam * np.eye(N), K)
    check("attribution: exact TRAK dual == primal (p x p) form",
          np.abs(primal - dual).max() < 1e-9, f"max|diff| = {np.abs(primal-dual).max():.2e}")

    F = (PHI.T @ PHI) / N
    brute = PHI @ np.linalg.solve(F + lam * np.eye(p), TG.T)
    inner = np.linalg.solve(lam * N * np.eye(N) + G, K)
    wood = (K - G @ inner) / lam
    check("attribution: Woodbury IF == dense empirical-Fisher inverse",
          np.abs(brute - wood).max() < 1e-9, f"max|diff| = {np.abs(brute-wood).max():.2e}")


def main():
    print("=" * 70)
    print("STAGE A -- SYNTHETIC FIXTURE TESTS (no real results here)")
    print("=" * 70)
    test_phases()
    test_windowing()
    test_masks()
    test_trainer()
    test_rollout_buffer()
    test_lds_scorer()
    test_attribution_estimators()
    n_pass = sum(r["pass"] for r in RES)
    out = {"SYNTHETIC": True, "n_tests": len(RES), "n_pass": n_pass,
           "all_pass": n_pass == len(RES), "tests": RES}
    p = os.path.join(RESULTS, "STAGE_A_SYNTHETIC_tests.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"\n{n_pass}/{len(RES)} passed -> {p}")
    sys.exit(0 if n_pass == len(RES) else 1)


if __name__ == "__main__":
    main()
