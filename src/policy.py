"""State-based BC-Transformer policy with a GMM action head (robomimic BC-Transformer analogue).

Input at each frame: [ z-normed state (128) , frozen task-language embedding (384) ]
Backbone:  causal GPT, CTX=10 frames, d_model=512, 6 layers, 8 heads  (~19.3M params)
Head:      GMM over the 7-D action, 5 modes  (loss = -log p(a_t | s_{t-9..t}, task))

Only the LAST position is supervised, so the training loss is exactly the deployment
conditional: predict a_t given the same 10-frame window the rollout will see. This makes the
per-timestep loss unambiguous, which the phase-masked functionals and the attribution
methods both depend on.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

CTX = 10
LANG_DIM = 384
ACTION_DIM = 7


class Block(nn.Module):
    def __init__(self, d, h, p):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(p))

    def forward(self, x, causal):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=causal, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class BCTransformer(nn.Module):
    def __init__(self, state_dim, d_model=512, n_layer=6, n_head=8, n_modes=5,
                 dropout=0.1, min_std=1e-4):
        super().__init__()
        self.state_dim, self.n_modes, self.min_std = state_dim, n_modes, min_std
        self.state_proj = nn.Linear(state_dim, d_model)
        self.lang_proj = nn.Linear(LANG_DIM, d_model)
        self.pos = nn.Parameter(torch.zeros(1, CTX, d_model))
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d_model, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_modes * (2 * ACTION_DIM + 1))
        causal = torch.triu(torch.ones(CTX, CTX, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal", causal, persistent=False)
        self.apply(self._init)
        nn.init.trunc_normal_(self.pos, std=0.02)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, states, lang):
        """states (B,CTX,state_dim), lang (B,LANG_DIM) -> (logits(B,K), mu(B,K,7), std(B,K,7))"""
        x = self.state_proj(states) + self.lang_proj(lang).unsqueeze(1) + self.pos
        x = self.drop(x)
        for b in self.blocks:
            x = b(x, self.causal)
        h = self.ln_f(x)[:, -1]                                  # LAST position only
        o = self.head(h).view(-1, self.n_modes, 2 * ACTION_DIM + 1)
        logits = o[..., 0]                                       # (B,K)
        mu = o[..., 1:1 + ACTION_DIM]                            # (B,K,7)
        std = F.softplus(o[..., 1 + ACTION_DIM:]) + self.min_std
        return logits, mu, std

    def nll(self, states, lang, actions):
        """Per-sample GMM negative log-likelihood. Returns (B,)."""
        logits, mu, std = self(states, lang)
        a = actions.unsqueeze(1)                                 # (B,1,7)
        # log N(a; mu, std) summed over action dims
        lp = (-0.5 * ((a - mu) / std) ** 2 - torch.log(std) - 0.5 * math.log(2 * math.pi)).sum(-1)
        return -torch.logsumexp(torch.log_softmax(logits, -1) + lp, dim=-1)

    def mean_action(self, states, lang):
        """Mixture-mean action sum_k pi_k mu_k. Differentiable (no argmax)."""
        logits, mu, _ = self(states, lang)
        w = torch.softmax(logits, -1).unsqueeze(-1)              # (B,K,1)
        return (w * mu).sum(1)                                   # (B,7)

    def l2(self, states, lang, actions):
        """Per-sample squared action error ||a_hat - a||^2. Returns (B,).

        This is the study's plain action-loss FUNCTIONAL (the spec allows "L2 or GMM NLL").
        We evaluate/attribute with L2 and train with GMM NLL, because the mean GMM NLL proved
        to be a broken measurement: it is unbounded and heavy-tailed, so when the GMM's sigma
        collapses on some seeds the held-out mean NLL swings 8-10x (e.g. mask D00: 23.6 vs
        187.0 on identical data) while the MEDIAN per-frame NLL barely moves (27.6 vs 32.4).
        An outcome that unreliable cannot be predicted by any attributor, so it would have
        indicted the attributors for a defect of the metric. L2 on the executed action is
        bounded (actions are in [-1,1]), behaviourally meaningful, and stable.
        """
        return ((self.mean_action(states, lang) - actions) ** 2).sum(-1)

    @torch.no_grad()
    def act(self, states, lang):
        """Deterministic action for rollout: mean of the highest-weight mode."""
        logits, mu, _ = self(states, lang)
        k = logits.argmax(-1)                                    # (B,)
        a = mu[torch.arange(mu.shape[0], device=mu.device), k]   # (B,7)
        return a.clamp(-1, 1)


def build(state_dim, cfg=None):
    cfg = cfg or {}
    return BCTransformer(
        state_dim,
        d_model=cfg.get("d_model", 512),
        n_layer=cfg.get("n_layer", 6),
        n_head=cfg.get("n_head", 8),
        n_modes=cfg.get("n_modes", 5),
        dropout=cfg.get("dropout", 0.1),
    )


def n_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    m = build(128)
    print(f"params: {n_params(m)/1e6:.2f}M  (spec requires 10-50M)")
    s = torch.randn(4, CTX, 128)
    l = torch.randn(4, LANG_DIM)
    a = torch.rand(4, 7) * 2 - 1
    print("nll", m.nll(s, l, a).shape, "act", m.act(s, l).shape)
