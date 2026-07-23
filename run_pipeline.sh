#!/usr/bin/env bash
# RoboTDA-X pipeline driver (post-Gate-0). Every stage is resumable: a run whose eval marker
# exists is skipped, so re-running this script after a kill picks up where it left off.
set -u
cd "$(dirname "$0")/src"
source /home/ljc/miniconda/etc/profile.d/conda.sh
conda activate robotda_x
L=../logs

run () {  # run <name> <cmd...>
  local name="$1"; shift
  echo "=== [$(date +%H:%M)] START $name ==="
  "$@" > "$L/$name.log" 2>&1
  local rc=$?
  echo "=== [$(date +%H:%M)] END $name (rc=$rc) ==="
  return $rc
}

# --- Stage E: the 10-model ensemble on all 135 demos (needed by TRAK and by attribution)
run stage_E python stage_efg.py --stage E

# --- Attribution over the full corpus (9 targets x 3 functionals x 135 demos)
run attribution python attribution.py

# --- Stage D: GATE 1 (attributor sanity). Uses Stage-B's full-C1 models.
run stage_D python stage_d.py

# --- Stage C: quantity sweep (+ TracIn intrusion vs Q)
run stage_C python stage_c.py

# --- Stage F: the main cluster-grain corpus (168 retrains, ~45k episodes)
run stage_F python stage_efg.py --stage F

# --- Stage G: demo-grain corpus (48 retrains)
run stage_G python stage_efg.py --stage G

# --- Moderators (DTW / MMD / bddl) then Stage H analysis
run moderators python moderators.py
run stage_H python analysis.py

# --- Stage I: RQ4 (needs the best LDS-validated attributor from Stage H)
run stage_I python stage_i.py

# --- Figures
run figures python figures.py

echo "=== pipeline complete ==="
