"""P17 -- EXPLORATORY cross-check: GradDot for the DIFFUSION policy. NOT a verdict.

Phase 4 left OPEN the hypothesis that the winning estimator differs by policy class: for the
BC-Transformer the raw gradient dot product (GradDot) beat TracIn (P11), while for the diffusion
policy TracIn passed (P13/P15). P17 computes ONE data point on that hypothesis: the diffusion
analog of GradDot, on the SAME frozen (t,eps)-bank denoise gradients the diffusion champion uses.

GradDot(diffusion) = the raw gradient dot product K = PHI @ TG.T at each member's FINAL checkpoint,
denoise test side (the same functional the P13/P15 champion used), E=5 members (dpens_s621-625),
per-member unit-L2 normalization, MEAN over members. One gradient pass (~0.2 GPU-h). Compared with
P15's S=10 diffusion MEDIAN ground truth, C1 and C5, alongside the archived TracIn numbers.

EXPLORATORY. No criterion, no gate, no pass/fail. The EXPLORATORY label appears in the artifact and
must appear wherever this table is mentioned.
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
import p5lib as L
from p5lib import P5_RESULTS, P3_RESULTS, P3_RUNS, RESULTS

sys.path.insert(0, os.path.join(L.ROOT, "src"))
import dataset  # noqa: E402

sys.path.insert(0, L.P3_SRC)
import evaluate_diffusion as EVD  # noqa: E402
import p10_attr as PA  # noqa: E402  (frozen diffusion gradient machinery)
from diffusion_data import ChunkBank, heldout_chunk_bank  # noqa: E402
from p10_bank import load_bank  # noqa: E402
from lds import spearman, spearman_p_onesided, mask_pred_score  # noqa: E402

MEMBERS = [f"dpens_s{s}" for s in range(621, 626)]        # E = 5
CACHE = os.path.join(P5_RESULTS, "p17_diffusion_gram_cache.npz")
FOCAL = ["C1", "C5"]
OUTCOME = "neg_plain_loss"


def build_cache():
    if os.path.exists(CACHE) and "--force" not in sys.argv:
        print(f"[P17] gram cache exists -> {CACHE}")
        return
    runs = [os.path.join(P3_RUNS, "P10ens", m) for m in MEMBERS]
    for r in runs:
        assert os.path.exists(os.path.join(r, "final.pt")), f"missing final.pt in {r}"

    # ---- probe-leak guard (all 5 members) BEFORE any gradient
    heldout = set(dataset.heldout_pool()[0])
    guard = L.assert_no_probe_leak(runs, heldout, context="P17 diffusion GradDot, E=5")
    print(f"[P17] probe-leak guard PASSED: {guard}")

    # ---- frozen (t,eps) bank + verify sha
    bank_path = os.path.join(P3_RESULTS, "p10_noise_bank.json")
    bank_sha = L.sha256_file(bank_path)
    assert bank_sha == "61aadccfef2fb45300d611f262bdc285c6a8f9888ed907a1c48b112d8405bc17", \
        f"noise bank sha mismatch: {bank_sha}"
    tb, eb = load_bank("cuda")
    print(f"[P17] frozen (t,eps) bank verified (sha {bank_sha[:12]}...), K={tb.shape[0]}")

    train_ids, _ = dataset.train_pool()
    cfg = json.load(open(os.path.join(runs[0], "train_meta.json")))["cfg"]
    tbank = ChunkBank(train_ids, H=cfg["h_chunk"])
    hbank = heldout_chunk_bank()
    slices = tbank.demo_slices()
    N = len(train_ids)
    clusters = dataset.clusters()

    Gs, Ks, members = [], [], []
    t0 = time.time()
    for run in runs:
        m = os.path.basename(run)
        model = EVD.load_model(os.path.join(run, "final.pt"), device="cuda")
        # denoise target gradients at the final checkpoint (the champion's test side)
        tg, order = PA.build_targets(model, hbank, tb, eb, kind="denoise")
        assert order == clusters, f"target order differs: {order} vs {clusters}"
        TG = torch.stack([tg[c] for c in order])                       # (T, p)
        PHI = torch.empty((N, TG.shape[1]), dtype=torch.float32, device="cuda")
        for i, d in enumerate(train_ids):
            PHI[i] = PA.demo_gradient(model, tbank, slices[d], tb, eb)
        G = (PHI @ PHI.T).double().cpu().numpy()
        K = (PHI @ TG.T).double().cpu().numpy()
        Gs.append(G)
        Ks.append(K)
        members.append(m)
        del model, PHI, TG, tg
        torch.cuda.empty_cache()
        print(f"[P17] {m} done ({time.time()-t0:.0f}s)", flush=True)

    np.savez(CACHE, G=np.stack(Gs), K=np.stack(Ks), members=np.array(members),
             train_ids=np.array(train_ids), targets=np.array(clusters),
             probe_leak_guard=json.dumps(guard))
    print(f"[P17] wrote {CACHE}")


def graddot_frame(train_ids, clusters):
    Z = np.load(CACHE, allow_pickle=True)
    K, G = Z["K"], Z["G"]
    mem, tids, tgts = list(Z["members"]), list(Z["train_ids"]), list(Z["targets"])
    assert tids == list(train_ids) and tgts == list(clusters)
    d = np.array([np.mean(np.diag(G[m])) for m in range(K.shape[0])])
    rows = []
    for mi, m in enumerate(mem):
        for j, c in enumerate(tgts):
            for i, t in enumerate(tids):
                rows.append(("GradDot", c, t, str(m), float(K[mi, i, j])))
                rows.append(("GradDot_dmean", c, t, str(m), float(K[mi, i, j] / d[mi])))
    return pd.DataFrame(rows, columns=["attributor", "target", "demo_id", "member", "score"])


def main():
    L.assert_prereg_locked()
    build_cache()

    clusters = dataset.clusters()
    train_ids, _ = dataset.train_pool()
    masks = json.load(open(os.path.join(RESULTS, "demo_mask_manifest.json")))["masks"]

    # P15 S=10 diffusion MEDIAN ground truth
    df = pd.read_parquet(os.path.join(P5_RESULTS, "p15_outcomes_S10.parquet"))
    p15v = json.load(open(os.path.join(P5_RESULTS, "p15_verdict.json")))
    SEEDS = p15v["seeds"]
    obs = {}
    for t in clusters:
        piv = df[df.target == t].pivot_table(index="mask_id", columns="seed", values=OUTCOME)[SEEDS]
        obs[t] = piv.median(axis=1).to_dict()

    gd = graddot_frame(train_ids, clusters)
    ceil = {t: p15v["all_targets_DESCRIPTIVE"][t]["ceiling_median_10seed_SB"] for t in clusters}

    # archived TracIn (P13 S=8 champion, P15 S=10 champion)
    p13 = json.load(open(os.path.join(L.P4_RESULTS, "p13_verdict.json")))
    tracin_s8 = {t: p13["all_targets_DESCRIPTIVE"][t]["estimators"]["TracIn_diffE5_normalized"]
                 for t in FOCAL}
    tracin_s10 = {t: p15v["all_targets_DESCRIPTIVE"][t]["estimators"]["TracIn_diffE5_normalized"]
                  for t in FOCAL}

    table = {"LABEL": "EXPLORATORY -- cross-check only, NOT a verdict, no criterion, no pass/fail",
             "ground_truth": f"P15 S={len(SEEDS)} diffusion MEDIAN, held-out L2",
             "targets": FOCAL, "n_masks": 24, "rows": {}}
    for t in FOCAL:
        out_v = np.array([obs[t].get(m["mask_id"], np.nan) for m in masks])
        row = {"ceiling_median_10seed_SB": ceil[t]}
        for attr, eid in (("GradDot", "GradDot_diffE5_normalized"),
                          ("GradDot_dmean", "GradDot_diffE5_dmean")):
            sc = L.normalized_ensemble_scores(gd, attr, t, train_ids, MEMBERS,
                                              normalize=(attr == "GradDot"))
            pred = np.array([mask_pred_score(sc, m["demos"]) for m in masks])
            ok = np.isfinite(out_v) & np.isfinite(pred)
            rho = spearman(pred[ok], out_v[ok])
            row[eid] = {"rho": rho, "ratio_to_ceiling": rho / ceil[t] if ceil[t] else np.nan,
                        "p_onesided": spearman_p_onesided(rho, int(ok.sum()))}
        row["TracIn_diffE5_normalized_S8_ARCHIVED"] = {
            "rho": tracin_s8[t]["rho"], "ratio": tracin_s8[t]["ratio_to_ceiling"],
            "p_onesided": tracin_s8[t]["p_onesided"]}
        row["TracIn_diffE5_normalized_S10"] = {
            "rho": tracin_s10[t]["rho"], "ratio": tracin_s10[t]["ratio_to_ceiling"],
            "p_onesided": tracin_s10[t]["p_onesided"]}
        table["rows"][t] = row

    L.atomic_write_json(os.path.join(P5_RESULTS, "p17_exploratory.json"), table)

    print("\n" + "=" * 104)
    print("P17 -- EXPLORATORY diffusion GradDot vs TracIn (S=10 median ground truth, n=24)")
    print("   *** EXPLORATORY -- NOT a verdict, no criterion ***")
    print("=" * 104)
    print(f"{'target':7s} {'ceiling':>8s} | {'GradDot rho':>12s} {'ratio':>6s} {'p1':>7s} | "
          f"{'dmean rho':>10s} {'ratio':>6s} | {'TracIn S8':>10s} {'S10':>8s}")
    for t in FOCAL:
        r = table["rows"][t]
        gdn = r["GradDot_diffE5_normalized"]
        dm = r["GradDot_diffE5_dmean"]
        print(f"{t:7s} {r['ceiling_median_10seed_SB']:8.3f} | "
              f"{gdn['rho']:+12.3f} {gdn['ratio_to_ceiling']:6.2f} {gdn['p_onesided']:7.4f} | "
              f"{dm['rho']:+10.3f} {dm['ratio_to_ceiling']:6.2f} | "
              f"{r['TracIn_diffE5_normalized_S8_ARCHIVED']['rho']:+10.3f} "
              f"{r['TracIn_diffE5_normalized_S10']['rho']:+8.3f}")
    print("=" * 104)


if __name__ == "__main__":
    main()
