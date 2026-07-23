"""P10 -- state-based DIFFUSION POLICY on the SAME interface as the BC-Transformer.

WHY IMPLEMENTED HERE RATHER THAN VIA `diffusers`: the frozen robotda_x env does not have
`diffusers` installed, and Phase 3 does not install into it (the env stays frozen and
reproducible). DDPM training and DDIM sampling are ~50 lines of closed-form algebra; they are
implemented directly and UNIT-TESTED against their closed forms in test_diffusion_SYNTHETIC.py
(labelled SYNTHETIC, never mixed with results).

INTERFACE -- deliberately IDENTICAL to policy.BCTransformer, so that the only thing that changes
between the two arms of P10 is the POLICY CLASS:
    * same 128-D observation, z-normalized with the FROZEN Phase-1 norm_stats (never refit)
    * same CTX = 10 observation history
    * same frozen MiniLM 384-D language conditioning
    * same 7-D action space

WHAT DIFFERS (the policy class itself):
    * head: an action-CHUNK denoiser -- predicts H = 8 future actions jointly
    * objective: DDPM epsilon-prediction MSE (not GMM NLL)
    * inference: DDIM, eta = 0, K = 10 steps

DETERMINISM (the hard requirement -- Phase 1's whole instrument depends on it):
BC-Transformer is deterministic because act() is an argmax. A diffusion policy is deterministic
iff (a) the sampler is DDIM with eta = 0 (no injected noise between steps), AND (b) the initial
latent x_T is FIXED rather than freshly sampled. We therefore draw x_T ONCE from a fixed seed
(DDIM_INIT_SEED) and reuse that same tensor at every call, for every model, forever. The policy
is then a deterministic function of (observation, language) -- exactly like argmax -- and a
repeated episode replays bit-for-bit. This is asserted end-to-end before any P10 experiment runs
(p10_determinism.py).

RECEDING HORIZON: the policy predicts a chunk of H = 8 actions but EXECUTES ONLY THE FIRST, then
replans. This keeps the closed-loop protocol IDENTICAL to the BC-Transformer's (one policy call
per env step), so the two policy classes differ in their class, not in their control protocol.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

CTX = 10
LANG_DIM = 384
ACTION_DIM = 7
H_CHUNK = 8               # action-chunk length
N_TRAIN_STEPS = 100       # DDPM diffusion steps
N_DDIM_STEPS = 10         # DDIM eval steps (eta = 0)
DDIM_INIT_SEED = 12345    # the FIXED initial latent -- what makes the policy deterministic


# ---------------------------------------------------------------- noise schedule
def cosine_alphas_cumprod(T, s=0.008, max_beta=0.999):
    """Nichol & Dhariwal cosine schedule WITH BETA CLIPPING. -> alpha_bar (T+1,), alpha_bar[0]=1.

    THE BETA CLIP IS NOT COSMETIC -- the SYNTHETIC unit test t6 failed without it.

    The raw cosine schedule gives f(T) = cos(pi/2)^2 = 0 exactly, so alpha_bar[T] = 0. The DDIM
    x0-estimate divides by sqrt(alpha_bar[t]):

        x0_hat = (x_t - sqrt(1 - ab_t) * eps) / sqrt(ab_t)

    so at t = T that is a division by ~0, which amplifies float32 roundoff by ~1e4 and makes the
    first denoising step numerically meaningless. Clipping each beta_t = 1 - ab_t/ab_{t-1} to
    <= 0.999 before re-accumulating (the standard `squaredcos_cap_v2` construction) keeps
    alpha_bar[T] safely positive and the x0-estimate well-conditioned. With the clip, an EXACT
    denoiser recovers x0 to < 1e-4 in float32 (test t6); without it, it does not.
    """
    t = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab_raw = f / f[0]
    betas = (1.0 - ab_raw[1:] / ab_raw[:-1]).clamp(max=max_beta)     # (T,)
    alphas = 1.0 - betas
    ab = torch.cat([torch.ones(1, dtype=torch.float64), torch.cumprod(alphas, 0)])
    return ab.float()


class Schedule:
    """Holds alpha_bar and the DDPM/DDIM coefficients derived from it."""

    def __init__(self, T=N_TRAIN_STEPS, device="cuda"):
        self.T = T
        ab = cosine_alphas_cumprod(T).to(device)      # (T+1,)
        self.alpha_bar = ab
        self.sqrt_ab = ab.sqrt()
        self.sqrt_1mab = (1 - ab).sqrt()

    def q_sample(self, x0, t, eps):
        """Forward diffusion: x_t = sqrt(ab_t) x0 + sqrt(1 - ab_t) eps. t is (B,) in [1..T]."""
        a = self.sqrt_ab[t].view(-1, 1, 1)
        b = self.sqrt_1mab[t].view(-1, 1, 1)
        return a * x0 + b * eps


# ---------------------------------------------------------------- modules
class Block(nn.Module):
    def __init__(self, d, h, p):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d),
                                 nn.Dropout(p))

    def forward(self, x, attn_mask=None):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


def timestep_embedding(t, dim):
    """Sinusoidal timestep embedding. t: (B,) long."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(a), torch.sin(a)], dim=-1)


class DiffusionPolicy(nn.Module):
    def __init__(self, state_dim, d_model=384, n_obs_layer=4, n_den_layer=6, n_head=6,
                 dropout=0.1, h_chunk=H_CHUNK, T=N_TRAIN_STEPS):
        super().__init__()
        self.state_dim, self.H, self.T, self.d = state_dim, h_chunk, T, d_model

        # ---- observation encoder (causal transformer over the CTX-frame history + language)
        self.state_proj = nn.Linear(state_dim, d_model)
        self.lang_proj = nn.Linear(LANG_DIM, d_model)
        self.obs_pos = nn.Parameter(torch.zeros(1, CTX, d_model))
        self.obs_blocks = nn.ModuleList([Block(d_model, n_head, dropout)
                                         for _ in range(n_obs_layer)])
        self.obs_ln = nn.LayerNorm(d_model)
        causal = torch.triu(torch.ones(CTX, CTX, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal", causal, persistent=False)

        # ---- denoiser (transformer over the H action tokens, conditioned on obs + lang + t)
        self.act_in = nn.Linear(ACTION_DIM, d_model)
        self.act_pos = nn.Parameter(torch.zeros(1, h_chunk, d_model))
        self.t_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(),
                                   nn.Linear(d_model, d_model))
        self.den_blocks = nn.ModuleList([Block(d_model, n_head, dropout)
                                         for _ in range(n_den_layer)])
        self.den_ln = nn.LayerNorm(d_model)
        self.act_out = nn.Linear(d_model, ACTION_DIM)

        self.apply(self._init)
        nn.init.trunc_normal_(self.obs_pos, std=0.02)
        nn.init.trunc_normal_(self.act_pos, std=0.02)
        nn.init.zeros_(self.act_out.weight)      # predict eps = 0 at init (standard, stabilizes)
        nn.init.zeros_(self.act_out.bias)

        self.sched = None                        # built lazily on the right device

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def schedule(self, device):
        if self.sched is None or self.sched.alpha_bar.device != torch.device(device):
            self.sched = Schedule(self.T, device=device)
        return self.sched

    def encode_obs(self, states, lang):
        """states (B,CTX,state_dim), lang (B,LANG_DIM) -> cond (B,d)."""
        x = self.state_proj(states) + self.lang_proj(lang).unsqueeze(1) + self.obs_pos
        for b in self.obs_blocks:
            x = b(x, self.causal)
        return self.obs_ln(x)[:, -1]                       # last frame's token = the condition

    def eps_pred(self, cond, x_t, t):
        """cond (B,d), x_t (B,H,7), t (B,) -> predicted noise (B,H,7)."""
        c = cond + self.t_mlp(timestep_embedding(t, self.d))          # (B,d)
        h = self.act_in(x_t) + self.act_pos + c.unsqueeze(1)          # (B,H,d)
        for b in self.den_blocks:
            h = b(h)
        return self.act_out(self.den_ln(h))

    # ---------------------------------------------------------------- training objective
    def ddpm_loss(self, states, lang, actions, t=None, eps=None, generator=None):
        """DDPM epsilon-prediction MSE, PER SAMPLE. actions (B,H,7). Returns (B,)."""
        B = states.shape[0]
        dev = states.device
        s = self.schedule(dev)
        if t is None:
            t = torch.randint(1, self.T + 1, (B,), device=dev, generator=generator)
        if eps is None:
            eps = torch.randn(actions.shape, device=dev, generator=generator)
        x_t = s.q_sample(actions, t, eps)
        pred = self.eps_pred(self.encode_obs(states, lang), x_t, t)
        return ((pred - eps) ** 2).mean(dim=(1, 2))                   # (B,)

    def bank_loss(self, states, lang, actions, t_bank, eps_bank):
        """The ATTRIBUTION functional: denoising loss averaged over a FIXED (t, eps) bank.

        t_bank: (K,) long; eps_bank: (K,H,7). The SAME bank is used for the train-side and the
        test-side gradients (MOTIVE-style variance reduction; it also removes the timestep
        sampling noise that would otherwise dominate a per-demo gradient). Returns (B,).
        """
        B = states.shape[0]
        K = t_bank.shape[0]
        s = self.schedule(states.device)
        cond = self.encode_obs(states, lang)                          # (B,d) -- computed ONCE
        tot = 0.0
        for k in range(K):
            t = t_bank[k].expand(B)
            e = eps_bank[k].unsqueeze(0).expand(B, -1, -1)
            x_t = s.q_sample(actions, t, e)
            pred = self.eps_pred(cond, x_t, t)
            tot = tot + ((pred - e) ** 2).mean(dim=(1, 2))
        return tot / K

    # ---------------------------------------------------------------- deterministic inference
    @torch.no_grad()
    def ddim_chunk(self, states, lang, n_steps=N_DDIM_STEPS):
        """DETERMINISTIC DDIM (eta = 0) from a FIXED initial latent. -> (B,H,7) action chunk."""
        B, dev = states.shape[0], states.device
        s = self.schedule(dev)
        g = torch.Generator(device="cpu").manual_seed(DDIM_INIT_SEED)
        x = torch.randn(1, self.H, ACTION_DIM, generator=g).to(dev).expand(B, -1, -1).contiguous()
        cond = self.encode_obs(states, lang)
        ts = torch.linspace(self.T, 1, n_steps).round().long().tolist()
        for i, t_cur in enumerate(ts):
            t = torch.full((B,), t_cur, device=dev, dtype=torch.long)
            eps = self.eps_pred(cond, x, t)
            ab_t = s.alpha_bar[t_cur]
            x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
            x0 = x0.clamp(-1, 1)                       # actions live in [-1,1]
            t_next = ts[i + 1] if i + 1 < len(ts) else 0
            ab_n = s.alpha_bar[t_next]
            x = ab_n.sqrt() * x0 + (1 - ab_n).sqrt() * eps      # eta = 0: NO noise injected
        return x

    @torch.no_grad()
    def act(self, states, lang, n_steps=N_DDIM_STEPS):
        """The executed action: the FIRST action of the denoised chunk. Deterministic."""
        return self.ddim_chunk(states, lang, n_steps)[:, 0].clamp(-1, 1)

    def l2(self, states, lang, actions_first, n_steps=N_DDIM_STEPS):
        """L2 on the EXECUTED (DDIM) action -- the outcome matched to the BC-Transformer's.

        actions_first: (B,7) the ground-truth action at time t.
        """
        a = self.act(states, lang, n_steps)
        return ((a - actions_first) ** 2).sum(-1)


def build(state_dim, cfg=None):
    cfg = cfg or {}
    return DiffusionPolicy(
        state_dim,
        d_model=cfg.get("d_model", 384),
        n_obs_layer=cfg.get("n_obs_layer", 4),
        n_den_layer=cfg.get("n_den_layer", 6),
        n_head=cfg.get("n_head", 6),
        dropout=cfg.get("dropout", 0.1),
        h_chunk=cfg.get("h_chunk", H_CHUNK),
        T=cfg.get("n_train_steps", N_TRAIN_STEPS),
    )


def n_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    m = build(128)
    print(f"DiffusionPolicy params: {n_params(m)/1e6:.2f}M  (target 10-30M)")
    s = torch.randn(4, CTX, 128)
    l = torch.randn(4, LANG_DIM)
    a = torch.rand(4, H_CHUNK, 7) * 2 - 1
    print("ddpm_loss", m.ddpm_loss(s, l, a).shape)
    print("act", m.act(s, l).shape)
