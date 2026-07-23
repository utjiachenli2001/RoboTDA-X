"""Task 5 -- CLI.

    python -m if_repair.run anchors
    python -m if_repair.run dev     --config if_repair/configs/dev.yaml
    python -m if_repair.run holdout --config if_repair/configs/frozen.yaml
    python -m if_repair.run diffusion --config if_repair/configs/frozen.yaml

`dev` refuses hold-out targets. `holdout` refuses unless the config says frozen: true,
and prints the locked config before computing anything.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from if_repair import data as D  # noqa: E402
from if_repair import anchors as A  # noqa: E402
from if_repair.eval import evaluate_spec  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def _table(rows, bonferroni_n=1):
    df = pd.DataFrame(rows)
    df["alpha_bonf"] = df["alpha"] / bonferroni_n
    df["PASS"] = df.passed & (df.p < df.alpha_bonf)
    cols = ["estimator", "target", "lds", "ceiling", "ratio", "bar", "p",
            "alpha_bonf", "n", "PASS"]
    return df[cols]


def cmd_anchors(_a):
    A.main()


def cmd_dev(a):
    cfg = yaml.safe_load(open(a.config))
    targets = list(cfg.get("targets", D.DEV_TARGETS))
    bad = [t for t in targets if t in D.HOLDOUT_TARGETS]
    if bad:
        sys.exit(f"REFUSED: `dev` may not touch hold-out targets {bad}. "
                 f"Hold-out is {list(D.HOLDOUT_TARGETS)} and is computed once, via "
                 f"`run holdout` with a frozen config.")
    tier = cfg.get("tier", "bc_s10")
    rows = []
    for spec in cfg["estimators"]:
        rows += evaluate_spec(spec, targets, tier)
    t = _table(rows)
    print(t.to_string(index=False))
    t.to_csv(os.path.join(RESULTS, "dev_table.csv"), index=False)


def _run_frozen(cfg, targets, tier, outname, label):
    print("=" * 96)
    print(f"{label} -- CONFIG LOCKED (frozen: true). Computed ONCE.")
    print("=" * 96)
    print(json.dumps(cfg, indent=1))
    n = len(cfg["estimators"]) * len(targets)
    print(f"Bonferroni family size = {n} "
          f"({len(cfg['estimators'])} estimators x {len(targets)} targets)")
    rows = []
    for spec in cfg["estimators"]:
        rows += evaluate_spec(spec, targets, tier)
    t = _table(rows, bonferroni_n=n)
    print(t.to_string(index=False))
    t.to_csv(os.path.join(RESULTS, outname), index=False)
    return t


def cmd_holdout(a):
    cfg = yaml.safe_load(open(a.config))
    if not cfg.get("frozen", False):
        sys.exit("REFUSED: hold-out requires `frozen: true` in the config. "
                 "Hold-out is computed exactly once, with a config frozen on dev.")
    targets = list(cfg.get("holdout_targets", D.HOLDOUT_TARGETS))
    _run_frozen(cfg, targets, cfg.get("tier", "bc_s10"), "holdout_table.csv",
                "HOLD-OUT (C2, C4, C7, C9)")


def cmd_diffusion(a):
    cfg = yaml.safe_load(open(a.config))
    if not cfg.get("frozen", False):
        sys.exit("REFUSED: diffusion transfer requires `frozen: true`.")
    targets = list(cfg.get("diffusion_targets", ["C1", "C5"]))
    _run_frozen(cfg, targets, "diff_s10", "diffusion_table.csv",
                "DIFFUSION cross-policy-class transfer")


def main():
    p = argparse.ArgumentParser(prog="if_repair.run")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("anchors").set_defaults(f=cmd_anchors)
    for name, fn in (("dev", cmd_dev), ("holdout", cmd_holdout),
                     ("diffusion", cmd_diffusion)):
        s = sub.add_parser(name)
        s.add_argument("--config", required=True)
        s.set_defaults(f=fn)
    a = p.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    a.f(a)


if __name__ == "__main__":
    main()
