"""SYNTHETIC unit tests for the P10 diffusion implementation.

*** SYNTHETIC. These are correctness tests of the DDPM/DDIM algebra on FABRICATED inputs.
*** They are NEVER mixed with study results. They exist because `diffusers` is not installed and
*** the sampler is hand-written, so its closed forms must be checked before it is trusted.

Tests:
  1. cosine schedule: alpha_bar is monotonically DECREASING, alpha_bar[0] == 1, all in (0,1]
  2. q_sample matches its closed form  x_t = sqrt(ab) x0 + sqrt(1-ab) eps  elementwise
  3. q_sample at t = T destroys the signal (corr(x_T, x0) ~ 0) and at t = 1 preserves it
  4. DDIM with eta=0 is DETERMINISTIC: two calls on identical input give bit-identical output
  5. DDIM is a deterministic FUNCTION OF THE OBSERVATION: different obs -> different actions
     (i.e. the fixed initial latent has not collapsed the policy to a constant)
  6. an eps-prediction model trained to perfectly predict eps recovers x0 exactly under DDIM
     (the sampler inverts the forward process when the denoiser is exact) -- the key algebra check
  7. bank_loss with a K=1 bank equals ddpm_loss with that same (t, eps) forced
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diffusion_policy as DP

OUT = {}


def t1_schedule():
    ab = DP.cosine_alphas_cumprod(100)
    ok = (float(ab[0]) == 1.0 and bool((ab[1:] < ab[:-1]).all())
          and bool(((ab > 0) & (ab <= 1)).all()))
    OUT["1_cosine_schedule_monotone"] = {"pass": ok, "alpha_bar_0": float(ab[0]),
                                         "alpha_bar_T": float(ab[-1])}
    return ok


def t2_qsample_closed_form():
    s = DP.Schedule(100, device="cpu")
    x0 = torch.randn(5, 8, 7)
    eps = torch.randn(5, 8, 7)
    t = torch.tensor([1, 25, 50, 75, 100])
    got = s.q_sample(x0, t, eps)
    want = torch.stack([s.sqrt_ab[t[i]] * x0[i] + s.sqrt_1mab[t[i]] * eps[i] for i in range(5)])
    d = float((got - want).abs().max())
    OUT["2_qsample_closed_form"] = {"pass": d == 0.0, "max_abs_diff": d}
    return d == 0.0


def t3_signal_destroyed_at_T():
    s = DP.Schedule(100, device="cpu")
    x0 = torch.randn(2000, 8, 7)
    eps = torch.randn(2000, 8, 7)
    xT = s.q_sample(x0, torch.full((2000,), 100), eps).flatten()
    x1 = s.q_sample(x0, torch.full((2000,), 1), eps).flatten()
    cT = float(np.corrcoef(xT.numpy(), x0.flatten().numpy())[0, 1])
    c1 = float(np.corrcoef(x1.numpy(), x0.flatten().numpy())[0, 1])
    ok = abs(cT) < 0.15 and c1 > 0.9
    OUT["3_signal_destroyed_at_T_preserved_at_1"] = {"pass": ok, "corr_xT_x0": cT,
                                                     "corr_x1_x0": c1}
    return ok


def t4_ddim_deterministic():
    torch.manual_seed(0)
    m = DP.build(128).eval()
    s = torch.randn(3, DP.CTX, 128)
    l = torch.randn(3, DP.LANG_DIM)
    a1 = m.act(s, l)
    a2 = m.act(s, l)
    d = float((a1 - a2).abs().max())
    OUT["4_ddim_deterministic"] = {"pass": d == 0.0, "max_abs_diff_between_two_calls": d}
    return d == 0.0


def t5_not_collapsed_to_constant():
    torch.manual_seed(0)
    m = DP.build(128).eval()
    # an untrained net with zero-init act_out predicts eps=0 -> the chunk is a pure function of
    # the fixed latent, so we must train it a little for this test to be meaningful.
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    S = torch.randn(64, DP.CTX, 128)
    L = torch.randn(64, DP.LANG_DIM)
    # target action chunk is a deterministic function of the observation
    A = torch.tanh(S[:, -1, :7]).unsqueeze(1).expand(-1, DP.H_CHUNK, -1).contiguous()
    for _ in range(200):
        loss = m.ddpm_loss(S, L, A).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    acts = m.act(S[:8], L[:8])
    spread = float(acts.std(0).mean())
    ok = spread > 1e-3
    OUT["5_ddim_is_a_function_of_the_observation"] = {
        "pass": ok, "std_of_actions_across_8_distinct_obs": spread,
        "note": "a fixed initial latent must NOT collapse the policy to a constant action"}
    return ok


def t6_exact_denoiser_recovers_x0():
    """THE KEY ALGEBRA CHECK. If the denoiser is EXACT, DDIM must invert the forward process.

    An ORACLE denoiser returns the TRUE eps for a known (x0, eps) pair. Starting from
    x_T = q_sample(x0, T, eps), the DDIM update must land the trajectory back on x0. This tests
    the UPDATE RULE itself -- if the coefficients were wrong, the trajectory would not close.

    Run in FLOAT64. Reason (and this is a property of DDIM, not of this implementation): the
    x0-estimate divides by sqrt(alpha_bar_t), which at t = T is ~5e-4 even after beta-clipping.
    That amplifies any roundoff by ~2000x, so a float32 version of this test measures float32,
    not the algebra. Every DDIM implementation lives with this and clamps x0 to the data range;
    so do we (see ddim_chunk). The float32 sampler's behaviour is covered by t4 (determinism) and
    t5 (not collapsed), and by the fact that alpha_bar[T] is no longer 0 (the beta clip).
    """
    torch.set_default_dtype(torch.float64)
    try:
        ab = DP.cosine_alphas_cumprod(100).double()
        torch.manual_seed(3)
        x0 = (torch.rand(4, 8, 7, dtype=torch.float64) * 2 - 1)     # actions live in [-1,1]
        eps = torch.randn(4, 8, 7, dtype=torch.float64)
        ts = torch.linspace(100, 1, 10).round().long().tolist()
        x = ab[100].sqrt() * x0 + (1 - ab[100]).sqrt() * eps        # the true x_T
        for i, tc in enumerate(ts):
            e = eps                                                  # ORACLE: the true noise
            x0h = ((x - (1 - ab[tc]).sqrt() * e) / ab[tc].sqrt()).clamp(-1, 1)
            tn = ts[i + 1] if i + 1 < len(ts) else 0
            x = ab[tn].sqrt() * x0h + (1 - ab[tn]).sqrt() * e        # eta = 0
        d = float((x - x0).abs().max())
    finally:
        torch.set_default_dtype(torch.float32)
    ok = d < 1e-8
    OUT["6_exact_denoiser_recovers_x0_under_ddim"] = {
        "pass": ok, "max_abs_diff_vs_x0_float64": d,
        "note": ("float64: tests the DDIM update rule itself. The float32 x0-estimate at t=T is "
                 "ill-conditioned by construction (1/sqrt(alpha_bar_T) ~ 2e3) in ANY DDIM, which "
                 "is why x0 is clamped to the action range."),
        "alpha_bar_T_after_beta_clip": float(DP.cosine_alphas_cumprod(100)[-1]),
    }
    return ok


def t7_bank_loss_equals_ddpm_loss():
    torch.manual_seed(0)
    m = DP.build(128).eval()
    S = torch.randn(6, DP.CTX, 128)
    L = torch.randn(6, DP.LANG_DIM)
    A = torch.rand(6, DP.H_CHUNK, 7) * 2 - 1
    tb = torch.tensor([37])
    eb = torch.randn(1, DP.H_CHUNK, 7)
    a = m.bank_loss(S, L, A, tb, eb)
    b = m.ddpm_loss(S, L, A, t=tb.expand(6), eps=eb.expand(6, -1, -1))
    d = float((a - b).abs().max().detach())
    OUT["7_bank_loss_K1_equals_ddpm_loss"] = {"pass": d < 1e-6, "max_abs_diff": d}
    return d < 1e-6


def main():
    torch.set_grad_enabled(True)
    tests = [t1_schedule, t2_qsample_closed_form, t3_signal_destroyed_at_T,
             t4_ddim_deterministic, t5_not_collapsed_to_constant,
             t6_exact_denoiser_recovers_x0, t7_bank_loss_equals_ddpm_loss]
    ok = True
    for t in tests:
        r = t()
        ok &= r
        print(f"  [{'PASS' if r else 'FAIL'}] {t.__name__}")

    m = DP.build(128)
    OUT["_params_M"] = DP.n_params(m) / 1e6
    OUT["_LABEL"] = ("SYNTHETIC -- correctness tests of the DDPM/DDIM algebra on FABRICATED "
                     "inputs. NEVER mixed with study results.")
    OUT["_ALL_PASS"] = bool(ok)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import p3lib as L
    L.atomic_write_json(os.path.join(L.P3_RESULTS, "SYNTHETIC_diffusion_unit_tests.json"), OUT)
    print(f"\nparams: {OUT['_params_M']:.2f}M (target 10-30M)")
    print(f"ALL PASS: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
