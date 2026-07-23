#!/bin/bash
# Wait for the P8a orchestrator to exit, then drive the remaining GPU stages in dependency order.
while pgrep -f "p89_jobs.py --run" > /dev/null; do sleep 60; done
echo "[chain] P8a orchestrator exited at $(date -Is)"
exec /home/ljc/miniconda/envs/robotda_x/bin/python /mnt/sdb/ljc/RoboTDA-X/phase3/src/drive_rest.py
