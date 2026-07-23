# ENV.md — RoboTDA-X environment record

Recorded: 2026-07-11. Host: multi-GPU Linux server (8× NVIDIA RTX A6000, 48 GB each).
**GPU policy for this study: ONLY indices 4,5,6,7 are ever used.** GPUs 0–3 are occupied by
another user's jobs (48,459 MiB used at project start) and are never touched.

## Platform
| item | value |
|---|---|
| OS | Linux 5.15.0-185-generic |
| CPU cores | 64 |
| NVIDIA driver | 580.159.03 |
| GPUs used | 4,5,6,7 (RTX A6000, 49,140 MiB each) |
| Disk (project volume) | /mnt/sdb, 726 GB free at start |

## Python environment
Conda env `robotda_x` at `/home/ljc/miniconda/envs/robotda_x`.

| package | version |
|---|---|
| Python | 3.12.13 |
| torch | 2.11.0+cu130 (CUDA 13.0, cuDNN 91900) |
| torchvision | 0.26.0 |
| robomimic | 0.2.0 |
| robosuite | 1.4.0 |
| mujoco | 3.8.1 |
| libero (`hf_libero`) | 0.1.4 |
| dattri | 0.3.0 |
| numpy | 2.2.6 |
| scipy | 1.18.0 |
| pandas | 2.3.3 |
| matplotlib | 3.11.0 |
| h5py | 3.16.0 |
| fastdtw | 0.3.4 |
| transformers | 5.5.4 |

Full pin: `requirements.txt` (173 packages, `pip freeze`).

### Deviation from spec: Python 3.12, not 3.10
The pre-existing env is Python 3.12.13. All required packages import and run correctly
(verified below), so the env was kept rather than rebuilt. Recorded as a deviation.

### Deviation from spec: LIBERO installed as a package, not a repo clone
LIBERO is installed as the `hf_libero` 0.1.4 pip distribution (import name `libero`),
which vendors the same `libero.libero.benchmark` API, bddl files and task suites as the
official repo. `third_party/` is therefore empty. Benchmark membership, task names, bddl
paths and language strings are all resolved **from the installed benchmark at load time**
(`src/clusters.py`), so no task list is hardcoded — this satisfies the spec's requirement.
Assets are cached at `/home/ljc/.cache/libero/assets`.

## Install verification (Stage 0)

### Datasets
All five suites downloaded to `data/libero/` (46 GB total), with
`data/libero/download.done.marker`: `libero_goal`, `libero_spatial`, `libero_object`,
`libero_10`, `libero_90`. Files are robomimic-convention hdf5.

### hdf5 contents (verified on `libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5`)
```
data/demo_i/{actions (T,7), dones, rewards, states (T,79), robot_states (T,9),
             obs/{ee_states (T,6), ee_pos (T,3), ee_ori (T,3), gripper_states (T,2),
                  joint_states (T,7), agentview_rgb, eye_in_hand_rgb}}
50 demos per task.
```
**Action dim = 7. ✓**

**Important finding: `object-state` is NOT stored in the hdf5 files.** The stored `obs`
group contains only proprioception and images. `object-state` exists only as an *env*
observation. Consequently the observation the spec asks for
(`ee_states, gripper_states, joint_states, object-state`) can only be obtained by replaying
each demo's stored full sim states (`data/demo_i/states`) through the LIBERO env and reading
the env's observation dict. This is what `src/extract.py` does
(`sim.set_state_from_flattened` → `_get_observations`). It has the added benefit that the
offline (training) and online (rollout) featurizations are produced by the identical code
path, so there is no train/rollout observation mismatch.

### Env smoke test (reset + 5 random actions, one task per suite)
Ran `ControlEnv(use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False)`.

| cluster | task | env build | step time | object-state dim | proprio dim |
|---|---|---|---|---|---|
| C1 libero_goal | open_the_middle_drawer_of_the_cabinet | 1.8 s | 14.7 ms | 56 | 16 |
| C2 libero_spatial | pick_up_the_black_bowl_between_… | 0.5 s | 13.8 ms | 70 | 16 |
| C3 libero_object | pick_up_the_alphabet_soup_… | 0.6 s | 13.3 ms | 98 | 16 |
| C4 libero_10 | KITCHEN_SCENE3_turn_on_the_stove_and_… | 0.2 s | 10.5 ms | 28 | 16 |

Env obs dict contains `object-state`, `robot0_eef_pos`, `robot0_eef_quat`,
`robot0_gripper_qpos`, `robot0_joint_pos` — all keys required for the state featurization. ✓

`object-state` **dimension varies by task** (28…98 observed) because it concatenates
per-object pose features and tasks have different object counts. The corpus therefore fixes a
global `obj_pad_dim` = max over all 70 corpus tasks and zero-pads.

### Observation vector (frozen)
```
state = [ robot0_eef_pos(3), robot0_eef_quat(4), robot0_gripper_qpos(2),
          robot0_joint_pos(7),                                    # proprio, 16-D
          object-state(zero-padded to obj_pad_dim) ]
```
The 16-D proprio block is the env-native equivalent of the hdf5's
`ee_states`/`gripper_states`/`joint_states` (eef pose + gripper width + joint angles);
it is read from the env so that offline and online featurization are byte-identical.
Z-normalized with statistics computed from the training pool only. **No images anywhere.**

### Deviation from spec #1 (material): task-language conditioning
The spec specifies a state-only policy with no task conditioning. This is **provably
degenerate on LIBERO**: the 10 `libero_goal` tasks share one scene and one object set, and
their reset states are byte-identical. Verified directly —

```
open_the_middle_drawer_of_the_cabinet   objdim=56 first8=[-0.091 -0.002 0.97 0. 0. 0.707 0.707 0.129]
put_the_bowl_on_the_stove               objdim=56 first8=[-0.091 -0.002 0.97 0. 0. 0.707 0.707 0.129]
put_the_wine_bottle_on_top_of_the_cabinet objdim=56 first8=[-0.091 -0.002 0.97 0. 0. 0.707 0.707 0.129]
```

An unconditioned state policy cannot represent a mapping from this state to three different
goal behaviours, so *both* the target-only and co-train arms of Gate 0 would be measuring
label noise, and every downstream attribution number would be attribution of noise.

The policy is therefore conditioned on a **frozen task-language embedding** of the LIBERO
task's `language` string (available as benchmark metadata, e.g. `"put the bowl on the
stove"`), concatenated to the state at every timestep. The encoder is frozen and never
trained, so it adds no parameters to the attributed model's data-dependent path; it is
metadata, not perception. **No image features are used anywhere in the study**, including in
the RQ2 moderator analysis, which remains trajectory-space (DTW / MMD / bddl overlap) as
specified.
