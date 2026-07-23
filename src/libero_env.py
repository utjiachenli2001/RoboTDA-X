"""LIBERO env wrapper for RoboTDA-X: state-based (no rendering), fast.

State featurization (consistent for offline demo extraction AND online rollout):
    concat[ robot0_eef_pos(3), robot0_eef_quat(4), robot0_gripper_qpos(2),
            robot0_joint_pos(7), object-state(var, zero-padded to global max) ]

We never use stored obs images; proprio + object-state are recomputed from the
env so that offline (replay of stored `states`) and online (live rollout) are identical.
"""
import os
import numpy as np
import bootstrap  # noqa: F401  (pins LIBERO_CONFIG_PATH / MUJOCO_GL before libero import)

# proprio keys and fixed dims
PROPRIO_KEYS = [
    ("robot0_eef_pos", 3),
    ("robot0_eef_quat", 4),
    ("robot0_gripper_qpos", 2),
    ("robot0_joint_pos", 7),
]
PROPRIO_DIM = sum(d for _, d in PROPRIO_KEYS)  # 16
ACTION_DIM = 7


def get_bddl_path(suite, task_name):
    from libero.libero import benchmark, get_libero_path
    bm = benchmark.get_benchmark_dict()[suite]()
    bddl_root = get_libero_path("bddl_files")
    for t in bm.tasks:
        if t.name == task_name:
            return os.path.join(bddl_root, t.problem_folder, t.bddl_file)
    raise KeyError(f"{task_name} not in {suite}")


def make_env(bddl_path, horizon=600, seed=0):
    """Fast low-dim LIBERO env: no cameras, no offscreen renderer."""
    from libero.libero.envs.env_wrapper import ControlEnv
    env = ControlEnv(
        bddl_file_name=bddl_path,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        horizon=horizon,
    )
    env.seed(seed)
    return env


def raw_object_state(obs):
    """Extract object-state vector from an env obs dict."""
    if "object-state" in obs:
        return np.asarray(obs["object-state"], dtype=np.float32)
    # fallback: some envs name it 'object_state'
    if "object_state" in obs:
        return np.asarray(obs["object_state"], dtype=np.float32)
    raise KeyError("no object-state in obs; keys=" + str(list(obs.keys())))


def proprio_vec(obs):
    parts = []
    for k, d in PROPRIO_KEYS:
        v = np.asarray(obs[k], dtype=np.float32).reshape(-1)
        assert v.shape[0] == d, f"{k} dim {v.shape[0]} != {d}"
        parts.append(v)
    return np.concatenate(parts)  # (16,)


def featurize(obs, obj_pad_dim):
    """proprio(16) + object-state zero-padded to obj_pad_dim -> (16+obj_pad_dim,)."""
    p = proprio_vec(obs)
    o = raw_object_state(obs)
    if o.shape[0] < obj_pad_dim:
        o = np.concatenate([o, np.zeros(obj_pad_dim - o.shape[0], np.float32)])
    elif o.shape[0] > obj_pad_dim:
        o = o[:obj_pad_dim]
    return np.concatenate([p, o])


def check_success(env):
    try:
        if env.check_success():
            return True
    except Exception:
        pass
    try:
        if env.env._check_success():
            return True
    except Exception:
        pass
    return False


def replay_states_extract(env, states):
    """Replay stored full sim states -> list of (proprio, raw_object_state) per frame.

    Uses robosuite reset_to via sim.set_state_from_flattened for exact reconstruction.
    """
    sim = env.env.sim
    out_p, out_o = [], []
    for t in range(len(states)):
        sim.set_state_from_flattened(states[t])
        sim.forward()
        obs = env.env._get_observations(force_update=True)
        out_p.append(proprio_vec(obs))
        out_o.append(raw_object_state(obs))
    return np.stack(out_p), np.stack(out_o)
