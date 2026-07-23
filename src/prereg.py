"""Emit preregistration.json — FROZEN before any study-stage training (spec §2).

Everything a later stage could be tempted to tune post-hoc is fixed here: seeds, probe tasks,
phase thresholds, gate criteria, focal targets, statistical thresholds, and the budget cut
order. Written BEFORE Stage B (Gate 0). The only runs that precede it are the Stage-A
synthetic tests and the C1-only policy calibration that the spec explicitly permits
("tune briefly on C1 only, then freeze"); neither produces a study result.
"""
import os
import sys
import json
import hashlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bootstrap  # noqa: F401
from bootstrap import ROOT, RESULTS
import dataset
import phases
import masks as MK
import train as T

OUT = os.path.join(ROOT, "preregistration.json")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def build():
    man = dataset.manifest()
    cfg = T.load_cfg()
    f = MK.cluster_mask_manifest()
    g = MK.demo_mask_manifest()

    pre = {
        "frozen_at": "before Stage B (Gate 0); after Stage-A synthetic tests and C1-only "
                     "policy calibration, neither of which is a study result",
        "seeds": {
            "corpus_seed": 0,
            "heldout_seed": 1,
            "cluster_mask_seed": MK.CLUSTER_MASK_SEED,
            "demo_mask_seed": MK.DEMO_MASK_SEED,
            "noise_ceiling_subset_seed": MK.NOISE_CEIL_SEED,
            "stage_B_gate0": [101, 102, 103, 104, 105],
            "stage_C_quantity": [101, 102, 103],
            "stage_D_gate1": [101, 102],
            "stage_E_ensemble": [201, 202, 203, 204, 205, 206, 207, 208, 209, 210],
            "stage_F_cluster_masks": [301, 302],
            "stage_F_noise_ceiling_extra": [303, 304],
            "stage_G_demo_masks": [401, 402],
            "stage_I_rq4": [501, 502, 503],
        },
        "corpus": {
            "M_clusters": 9,
            "clusters": {c["cluster"]: {"suite": c["suite"], "scene": c["scene"],
                                        "n_tasks": c["n_tasks"]} for c in man["clusters"]},
            "n_train_per_cluster": 15, "N_train_total": 135,
            "n_heldout_per_cluster": 10, "n_heldout_total": 90,
            "state_dim": man["state_dim"], "obj_pad_dim": man["obj_pad_dim"],
            "action_dim": 7,
            "probe_tasks": dataset.probe_tasks(),
            "corpus_manifest_sha256_16": sha(os.path.join(RESULTS, "corpus_manifest.json")),
        },
        "phase_segmentation": {
            "rule": "interaction = within +-WINDOW steps of a gripper open/close midpoint "
                    "crossing, OR (gripper closed AND ee speed < PERCENTILE-th pct of that "
                    "demo's speeds); transport = complement",
            "frozen": phases.BASE,
            "sensitivity_variants": {"low": phases.LOW, "high": phases.HIGH},
            "under_identification_flag_if_transport_frac_std_below": 0.05,
        },
        "policy_and_training": {
            "arch": "state-based BC-Transformer, causal, CTX=10, GMM(5) action head",
            "params_M": 19.22,
            "loss": "GMM negative log-likelihood (frozen choice; the alternative L2 was not used)",
            "config": cfg,
            "budget_convention": "FIXED TOTAL GRADIENT STEPS for every run in the study "
                                 "(not fixed epochs): Gate 0 and Stage C compare datasets of "
                                 "different size, and fixed epochs would give the larger "
                                 "dataset proportionally more optimization, confounding "
                                 "'more data helps' with 'more steps'.",
            "n_checkpoints": cfg["n_ckpt"],
        },
        "evaluation": {
            "probe_battery": "3 probe tasks/cluster x 10 rollouts = 30 episodes/cluster, "
                             "plus held-out plain/transport/interaction losses on the "
                             "cluster's 10 held-out demos",
            "rollout_horizon": 600,
            "init_states": "LIBERO's own fixed per-task init states, indices 0..R-1 -- "
                           "identical initial conditions for every model in the study",
            "logit_transform": "logit(clamp(p, 1/(2n), 1-1/(2n))); n=30 -> clamp [1/60,59/60]",
        },
        "gates": {
            "gate0_stage_B": {
                "design": "target-only (C1's 15 demos) vs co-train (all 135), 5 seeds, "
                          "paired by seed; evaluate on the FULL libero_goal suite "
                          "(10 tasks x 20 rollouts = 200 episodes/model)",
                "criterion": "mean paired margin (co-train - target-only, averaged over the "
                             "10 tasks) >= +5 success points AND one-sided paired t-test "
                             "p < 0.05 (df=4, t > 2.132)",
                "on_fail": "run the same paired design for TWO more targets (C2, C5). If no "
                           "target passes -> write GATE0_FAIL.md and HALT the pipeline. If "
                           "some pass -> proceed, noting heterogeneity is itself a finding.",
            },
            "gate1_stage_D": {
                "design": "C1 only; K=12 fixed-size masks (each exactly 8 of C1's 15 demos; "
                          "each demo in 6 or 7 masks) x 2 seeds = 24 retrains",
                "criterion": "PASS iff ANY attributor (TracIn, TRAK, EK-FAC IF) reaches "
                             "Spearman > 0.50 vs retrained outcome (held-out plain loss "
                             "primary, success secondary) over the 12 masks",
                "on_fail": "complete Stages E-G anyway (retrains are attribution-agnostic "
                           "ground truth) but STOP before Stage H conclusions and write "
                           "GATE1_FAIL.md; report 'attribution too unfaithful to decompose'",
            },
        },
        "attribution": {
            "attributors": ["TracIn", "TRAK(E=10)", "EK-FAC IF"],
            "functionals": ["plain", "transport_masked", "interaction_masked"],
            "targets": dataset.clusters(),
            "grain": "per-demo scores over all 135 training demos; mask score = sum over the "
                     "mask's demos",
        },
        "lds": {
            "primary": "CONDITIONAL LDS -- Spearman(predicted, outcome) over ONLY the 40 "
                       "target-included masks; outcome = seed-mean logit-success on the target",
            "secondary": "full-72-mask LDS, labelled 'inflated by target inclusion'",
            "noise_ceiling": "12 replicate masks x 4 seeds; all 3 disjoint seed pairings; "
                             "Spearman between pair-mean outcome vectors; averaged; "
                             "bootstrap CI over masks. LDS is judged against the ceiling, "
                             "never against 1.0.",
            "headline_significance": "9 per-target conditional LDS on success; one-sided "
                                     "Bonferroni alpha = 0.05/9 = 0.00556 (critical rho ~ 0.41 "
                                     "at n=40)",
            "demo_grain": {
                "focal_targets": ["C1", "C5"],
                "focal_rule": "PRE-REGISTERED as confirmatory at Bonferroni-2 "
                              "(alpha = 0.025 one-sided, critical rho ~ 0.41 at n=24); the "
                              "other 7 targets are exploratory",
                "downgrade_rule": "if demo-grain LDS fails on BOTH focal targets, ALL per-demo "
                                  "claims downgrade to cluster grain and this is stated "
                                  "plainly in the report",
                "partial_spearman_control": "in-target demo count (7 vs 8)",
            },
        },
        "mask_designs": {
            "stage_F": {"K": f["K"], "clusters_per_mask": 5, "demos_per_mask": 75,
                        "inclusions_per_cluster": 40,
                        "coinclusion_range": [f["coinclusion_offdiag_min"],
                                              f["coinclusion_offdiag_max"]],
                        "noise_ceiling_masks": f["noise_ceiling_masks"],
                        "sha256_16": sha(os.path.join(RESULTS, "mask_manifest.json"))},
            "stage_G": {"K": g["K"], "demos_per_mask": 68,
                        "per_demo_inclusion_range": [g["per_demo_inclusion_min"],
                                                     g["per_demo_inclusion_max"]],
                        "sha256_16": sha(os.path.join(RESULTS, "demo_mask_manifest.json"))},
        },
        "budget_cut_order": [
            "1. rollouts 10 -> 7 per probe task",
            "2. demo-corpus probes restricted to C1+C5 only",
            "3. noise-ceiling replicates 12 -> 8 masks",
            "4. K 72 -> 63 (keeps balance at 35 inclusions/cluster)",
        ],
        "budget_ledger": {
            "B_gate0": {"retrains": "10-20", "gpu_h": "4-8", "episodes": "2000-4000"},
            "C_quantity": {"retrains": 18, "gpu_h": 18, "episodes": 3600},
            "D_gate1": {"retrains": 24, "gpu_h": 4, "episodes": 2400},
            "E_ensemble": {"retrains": 10, "gpu_h": 7, "episodes": 2700},
            "F_cluster": {"retrains": 168, "gpu_h": 84, "episodes": 45360},
            "G_demo": {"retrains": 48, "gpu_h": 23, "episodes": 12960},
            "I_rq4": {"retrains": 24, "gpu_h": 9, "episodes": 1440},
            "H_attribution": {"retrains": 0, "gpu_h": "5-10", "episodes": 0},
            "total": {"retrains": 302, "gpu_h": 154, "episodes": 70500},
            "pause_rule": "if a 75-demo retrain exceeds 0.8 GPU-h, or a stage's projected "
                          "total exceeds 1.5x this ledger, PAUSE and write BUDGET_ALERT.md",
        },
        "rq4_stage_I": {
            "budget_B": 15,
            "targets": ["C1", "C5"],
            "conditions": ["target_only", "influence_top15", "random15", "similarity_top15"],
            "seeds": [501, 502, 503],
            "eval": "target's 3 probe tasks x 20 rollouts",
            "note": "a negative result (influence <= similarity) is reported plainly",
        },
        "moderators_rq2": {
            "similarity": ["DTW over ee_states (position+gripper)",
                           "object-state MMD (RBF kernel)", "bddl object/primitive overlap"],
            "regression": "target's insider-advantage AUC ~ similarity + within-target "
                          "redundancy (mean pairwise in-target DTW)",
            "forbidden": "NO image/DINO similarity anywhere",
        },
        "deviations_from_spec": [
            "Python 3.12 (not 3.10): pre-existing env, all packages verified working.",
            "LIBERO installed as the hf_libero pip package rather than a repo clone; "
            "benchmark membership resolved from the installed benchmark at load time.",
            "MATERIAL: the policy is conditioned on a FROZEN task-language embedding "
            "(MiniLM, 384-D) of the LIBERO task string. The 10 libero_goal tasks share one "
            "scene and byte-identical reset states, so an unconditioned state policy provably "
            "cannot represent them and every arm of the study would measure label noise. "
            "The encoder is frozen; no image features are used anywhere.",
            "object-state is reconstructed by replaying each demo's stored sim states through "
            "the env (LIBERO hdf5 files do not store object-state), which also makes the "
            "offline and online featurizations byte-identical.",
            "Training budget is a fixed number of GRADIENT STEPS rather than fixed epochs "
            "(see policy_and_training.budget_convention).",
        ],
    }
    json.dump(pre, open(OUT, "w"), indent=1)
    return pre


if __name__ == "__main__":
    p = build()
    print(f"[prereg] wrote {OUT}")
    print(f"  focal targets: {p['lds']['demo_grain']['focal_targets']}")
    print(f"  seeds: {list(p['seeds'].keys())}")
    print(f"  policy: {p['policy_and_training']['params_M']}M params, "
          f"total_steps={p['policy_and_training']['config']['total_steps']}")
