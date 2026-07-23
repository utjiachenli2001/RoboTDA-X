"""Phase segmentation (FROZEN rule, per spec §2).

Per timestep, from the extracted proprio stream:
  v_t          = || pos_{t+1} - pos_t ||           (ee speed, eef_pos = proprio[:, 0:3])
  width_t      = gripper_qpos[t,0] - gripper_qpos[t,1]   (gripper opening)
  midpoint     = (min(width) + max(width)) / 2     (per demo)
  gripper event at t  iff (width_t - mid) and (width_{t-1} - mid) have opposite sign
  interaction  = |t - e| <= WINDOW for some event e
                 OR (gripper closed at t  AND  v_t < PERCENTILE-th pct of that demo's speeds)
  transport    = NOT interaction

Frozen thresholds: WINDOW=10 steps, PERCENTILE=30.
Sensitivity variants at +-25%: low=(8, 22.5), high=(12, 37.5).
"""
import numpy as np

BASE = {"window": 10, "percentile": 30.0}
LOW = {"window": 8, "percentile": 22.5}
HIGH = {"window": 12, "percentile": 37.5}
VARIANTS = {"base": BASE, "low": LOW, "high": HIGH}

EEF_POS = slice(0, 3)
GRIP_QPOS = slice(7, 9)   # proprio layout: eef_pos(3) eef_quat(4) gripper_qpos(2) joint_pos(7)


def ee_speed(proprio):
    """v_t = ||pos_{t+1} - pos_t||; last step repeats the previous value. Shape (T,)."""
    pos = np.asarray(proprio[:, EEF_POS], dtype=np.float64)
    T = pos.shape[0]
    if T < 2:
        return np.zeros(T)
    d = np.linalg.norm(np.diff(pos, axis=0), axis=1)      # (T-1,)
    return np.concatenate([d, d[-1:]])                     # (T,)


def gripper_width(proprio):
    q = np.asarray(proprio[:, GRIP_QPOS], dtype=np.float64)
    return q[:, 0] - q[:, 1]


def phase_masks(proprio, window=10, percentile=30.0):
    """Return (interaction, transport) boolean arrays of shape (T,)."""
    T = proprio.shape[0]
    v = ee_speed(proprio)
    w = gripper_width(proprio)
    mid = 0.5 * (w.min() + w.max())
    rel = w - mid

    inter = np.zeros(T, dtype=bool)

    # (a) within +-window of a gripper open/close crossing
    if T >= 2:
        cross = np.nonzero(np.sign(rel[1:]) * np.sign(rel[:-1]) < 0)[0] + 1  # index of the step after the crossing
        for e in cross:
            lo, hi = max(0, e - window), min(T, e + window + 1)
            inter[lo:hi] = True

    # (b) gripper closed AND slow
    closed = rel < 0                       # width below midpoint => closed
    thr = np.percentile(v, percentile)
    inter |= (closed & (v < thr))

    return inter, ~inter


def transport_fraction(proprio, **kw):
    _, tr = phase_masks(proprio, **kw)
    return float(tr.mean())


def all_variant_masks(proprio):
    """{'base'|'low'|'high' -> (interaction, transport)}"""
    return {k: phase_masks(proprio, **v) for k, v in VARIANTS.items()}
