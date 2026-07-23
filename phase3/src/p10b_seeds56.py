"""P10b EXTENSION: seeds 605, 606 -> S = 6, matching the BC arm exactly.

DISCLOSED DEVIATION (PHASE3_DEFECT.md §5): the preregistered S=4 ceiling collapsed (4 of 9
negative; SB consistency check off by 0.37), so the ratio-to-ceiling criterion had no power.
The criterion is UNCHANGED; only S changes, 4 -> 6. This is the move Phase 2 made on Phase 1:
raise the ceiling, do not lower the bar.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3lib as L
from p10_jobs import p10b_jobs
import orch3

jobs = p10b_jobs(seeds=[605, 606])
L.atomic_write_json(os.path.join(L.P3_RESULTS, "p10b_jobs_seeds56.json"), jobs)
print(f"[P10b-ext] {len(jobs)} jobs (24 masks x seeds 605,606)")
if "--run" in sys.argv:
    orch3.run_jobs(jobs, "P10b_s56")
