# PASS 8 -- in-flight status (delete when HANDOFF.md is rewritten)

**If you are picking this up mid-flight, read `p8_prereg.md` first, then this file.**

## Where it is

Campaign N is running detached on h200-3 under `nohup`, 3 workers, launched 2026-07-27.
Commits `956c061` (prereg, at zero runs) and `327f7c5` (scorer + pins) are pushed.

```bash
# progress
ls if_repair/runs/campaigns/N/*.npz | wc -l          # target 1390
pgrep -fc if_repair.retrain                          # expect 3
tail -3 if_repair/runs/logs/p8_N_w0.log
# relaunch a dead worker (run_job skips completed jobs, so this is always safe)
cd ~/code/RoboTDA-X && nohup env CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  -m if_repair.retrain --campaign N --worker 0 --nworkers 3 \
  > if_repair/runs/logs/p8_N_w0.log 2>&1 &
```

**`CUDA_VISIBLE_DEVICES=0` is required on every command.** `src/bootstrap.py` has
`ALLOWED_GPUS = (4,5,6,7)` from the original 8-GPU machine; on this 1-GPU box its probe fails and
it pins to a nonexistent GPU 4, after which `torch.cuda.is_available()` is False. It respects an
already-set value, so exporting it is the fix -- do not edit the repo.

## What must NOT happen

- **Do not run `confirm_nseries.py` more than once.** It refuses to overwrite, by design.
- **Do not edit `p8_prereg.md`.** It was frozen at 956c061 while campaign N had zero runs.
- **Do not extend campaign N to chase a p-value.** Same rule that bound campaign M.

## When the campaign finishes (or is time-boxed out)

The stopping rule is prespecified: analyse at the largest depth where ALL conditional masks are
complete. `achieved_depth()` computes it from the run directory; it does not need the campaign to
be 100% done, and a partial run is still a complete balanced design because the job list is
seed-major.

```bash
cd ~/code/RoboTDA-X
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m if_repair.confirm_nseries \
    --i_understand_this_scores_once
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m if_repair.p8_figs
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest if_repair/tests -q
```

Then write up FINDINGS / RESULTS / BLOCKERS / HANDOFF, commit, push.

## The result the write-up has to report honestly

The Stage F scan is a SCAN. Whatever campaign N says about the absolute bar, the pass's durable
finding is already banked and does not depend on it: every self-influence and leverage correction
from passes 4-7 reverses sign at cluster grain, and the plain GradDot baseline they were built to
improve is strongly positive there. If campaign N contradicts the scan, report the contradiction
-- the two use different outcome pipelines (probe battery vs `heldout_frame_losses`) and that is
itself the finding.
