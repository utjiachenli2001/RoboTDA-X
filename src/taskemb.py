"""Frozen task-language embeddings (metadata conditioning; see ENV.md Deviation #1).

The 10 libero_goal tasks share one scene and byte-identical reset states, so an
unconditioned state policy cannot represent them. We condition on a FROZEN sentence
embedding of the task's LIBERO `language` string. The encoder is never trained and
contributes no data-dependent parameters to the attributed model.

Encoder: sentence-transformers/all-MiniLM-L6-v2 (mean-pooled, L2-normalized, 384-D).
Cached once to results/task_emb.npz — deterministic, computed on CPU.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bootstrap  # noqa: F401
from bootstrap import RESULTS

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMB_DIM = 384
CACHE = os.path.join(RESULTS, "task_emb.npz")

_cache = None


def task_language():
    """{task_name -> language string} over all installed suites we use."""
    from libero.libero import benchmark
    bm = benchmark.get_benchmark_dict()
    out = {}
    for suite in ("libero_goal", "libero_spatial", "libero_object", "libero_10", "libero_90"):
        for t in bm[suite]().tasks:
            out[t.name] = t.language
    return out


def build_cache():
    import torch
    from transformers import AutoTokenizer, AutoModel
    lang = task_language()
    names = sorted(lang.keys())
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModel.from_pretrained(MODEL).eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(names), 32):
            batch = [lang[n] for n in names[i:i + 32]]
            enc = tok(batch, padding=True, truncation=True, return_tensors="pt")
            out = mdl(**enc).last_hidden_state              # (B,T,384)
            m = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out * m).sum(1) / m.sum(1).clamp(min=1e-9)   # mean pool
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            embs.append(pooled.cpu().numpy().astype(np.float32))
    E = np.concatenate(embs, 0)
    assert E.shape == (len(names), EMB_DIM), E.shape
    np.savez(CACHE, names=np.array(names), emb=E,
             lang=np.array([lang[n] for n in names]))
    return names, E


def load():
    """{task_name -> (384,) float32}"""
    global _cache
    if _cache is None:
        if not os.path.exists(CACHE):
            build_cache()
        z = np.load(CACHE, allow_pickle=False)
        _cache = {str(n): z["emb"][i] for i, n in enumerate(z["names"])}
    return _cache


if __name__ == "__main__":
    names, E = build_cache()
    print(f"[taskemb] {len(names)} tasks, dim={E.shape[1]}, cache={CACHE}")
    # sanity: goal tasks must have DISTINCT embeddings (that's the whole point)
    import itertools
    g = [n for n in names if "drawer" in n or "stove" in n][:5]
    d = load()
    for a, b in itertools.islice(itertools.combinations(g, 2), 5):
        print(f"  cos({a[:28]}, {b[:28]}) = {float(d[a] @ d[b]):.3f}")
