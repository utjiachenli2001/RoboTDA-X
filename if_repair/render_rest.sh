#!/usr/bin/env bash
# Render the remaining gallery demos, one process each, under an OS-level timeout.
#
# `timeout` sends SIGTERM then SIGKILL and does not care what the process is doing -- unlike the
# in-process signal.alarm that failed to interrupt a MuJoCo C-call stall for 14.5 hours. A demo that
# stalls now costs TIMEOUT seconds and is recorded as a failure.
set -u
cd ~/code/RoboTDA-X
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa CUDA_VISIBLE_DEVICES=0
TIMEOUT="${TIMEOUT:-240}"
OUT=docs/demo_gallery

# the 24 selected demos, in rank order, as "band<TAB>demo_id"
.venv/bin/python - <<'PY' > /tmp/sel.tsv
import json, os
s = json.load(open("if_repair/results/p17_demo_scores.json"))
rows = sorted(s["demos"], key=lambda r: -r["z_mean"])
for i, r in enumerate(rows[:12] + rows[-12:]):
    print(("high" if i < 12 else "low") + "\t" + r["demo_id"])
PY

n=0
while IFS=$'\t' read -r band did; do
  n=$((n+1))
  printf '[%02d/24] %-4s %s\n' "$n" "$band" "$did"
  timeout -k 10 "$TIMEOUT" .venv/bin/python -m if_repair.render_one --demo "$did" --band "$band" \
      2>/dev/null | tail -1
  rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    echo "    TIMEOUT after ${TIMEOUT}s — recording as a failure and moving on"
    .venv/bin/python - "$did" "$band" <<'PY'
import json, sys, os
did, band = sys.argv[1], sys.argv[2]
p = "docs/demo_gallery/gallery.json"
m = json.load(open(p))
m.setdefault("failures", [])
if not any(f["demo_id"] == did for f in m["failures"]):
    m["failures"].append({"demo_id": did, "band": band,
                          "reason": "MuJoCo solver stall on replayed actions — killed by OS timeout; "
                                    "an in-process signal cannot interrupt a C call"})
json.dump(m, open(p, "w"), indent=1)
PY
  fi
done < /tmp/sel.tsv

echo
.venv/bin/python -c "
import json; d=json.load(open('docs/demo_gallery/gallery.json'))
print(f\"DONE: {len(d['demos'])} rendered, {len(d.get('failures',[]))} failed\")
for f in d.get('failures',[]): print('   FAIL', f['demo_id'])
"
