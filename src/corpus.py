"""Corpus construction for RoboTDA-X (deterministic).

Writes:
  results/task_meta.json     - per task: objects(from bddl), object_state_dim, num_demos, demo_lengths
  results/corpus_manifest.json - clusters, train/heldout demo IDs, probe tasks, obj_pad_dim
Seeds (named, frozen): corpus_seed=0 (train pool), heldout_seed=1 (held-out).
Selection rule: sort tasks alphabetically; round-robin over tasks taking lowest demo indices.
"""
import os
import re
import json
import glob
import numpy as np
import h5py

import sys
sys.path.insert(0, os.path.dirname(__file__))
from clusters import resolve_clusters, hdf5_filename
import libero_env as LE

ROOT = "/mnt/sdb/ljc/RoboTDA-X"
DATA = os.path.join(ROOT, "data/libero")
RESULTS = os.path.join(ROOT, "results")

N_TRAIN_PER_CLUSTER = 15
N_HELDOUT_PER_CLUSTER = 10
N_PROBE_TASKS = 3


def parse_bddl_objects(suite, task_name):
    """Return sorted list of object *categories* from the task's bddl (:objects section)."""
    bddl = LE.get_bddl_path(suite, task_name)
    with open(bddl) as f:
        txt = f.read()
    m = re.search(r"\(:objects(.*?)\)", txt, re.S)
    cats = set()
    if m:
        for line in m.group(1).strip().splitlines():
            line = line.strip()
            mm = re.match(r"\S+\s*-\s*(\S+)", line)
            if mm:
                cats.add(mm.group(1))
    return sorted(cats)


def task_hdf5(cluster_suite, task_name):
    return os.path.join(DATA, cluster_suite, hdf5_filename(task_name))


def demo_lengths(h5path):
    """Return {demo_id: T} for demos, ordered demo_0..demo_{n-1}."""
    out = {}
    with h5py.File(h5path, "r") as f:
        data = f["data"]
        ids = sorted(data.keys(), key=lambda s: int(s.split("_")[1]))
        for k in ids:
            out[k] = int(data[k]["actions"].shape[0])
    return out


def object_state_dim(suite, task_name):
    """Build env, reset, read object-state dim."""
    bddl = LE.get_bddl_path(suite, task_name)
    env = LE.make_env(bddl, horizon=10, seed=0)
    obs = env.reset()
    dim = int(LE.raw_object_state(obs).shape[0])
    try:
        env.close()
    except Exception:
        pass
    return dim


def build_task_meta(clusters, need_data=True):
    meta = {}
    for c in clusters:
        for t in c["tasks"]:
            if t in meta:
                continue
            entry = {
                "cluster": c["cluster"],
                "suite": c["suite"],
                "objects": parse_bddl_objects(c["suite"], t),
                "object_state_dim": object_state_dim(c["suite"], t),
            }
            if need_data:
                h5 = task_hdf5(c["suite"], t)
                if os.path.exists(h5):
                    dl = demo_lengths(h5)
                    entry["num_demos"] = len(dl)
                    entry["demo_lengths"] = dl
            meta[t] = entry
    return meta


def round_robin_select(tasks, per_task_ndemos, n_take, skip=None):
    """Round-robin over alphabetically-sorted tasks, taking lowest demo indices.
    Returns list of (task, demo_id). `skip` = set of (task,demo_id) already used."""
    skip = skip or set()
    tasks = sorted(tasks)
    picked = []
    rank = 0
    maxn = max(per_task_ndemos.values()) if per_task_ndemos else 0
    while len(picked) < n_take and rank < maxn:
        for t in tasks:
            if rank < per_task_ndemos.get(t, 0):
                did = f"demo_{rank}"
                if (t, did) not in skip:
                    picked.append((t, did))
                    if len(picked) >= n_take:
                        break
        rank += 1
    return picked


def greedy_probe_tasks(tasks, task_objects, k=N_PROBE_TASKS):
    """Greedy max-coverage over object categories; ties -> alphabetical."""
    tasks = sorted(tasks)
    covered = set()
    chosen = []
    remaining = list(tasks)
    while remaining and len(chosen) < k:
        best = None
        best_gain = -1
        for t in remaining:
            gain = len(set(task_objects.get(t, [])) - covered)
            if gain > best_gain or (gain == best_gain and (best is None or t < best)):
                best_gain = gain
                best = t
        chosen.append(best)
        covered |= set(task_objects.get(best, []))
        remaining.remove(best)
    return sorted(chosen)


def build_corpus():
    os.makedirs(RESULTS, exist_ok=True)
    clusters = resolve_clusters()
    print("[corpus] building task meta (env resets)...", flush=True)
    meta = build_task_meta(clusters, need_data=True)
    obj_pad_dim = max(m["object_state_dim"] for m in meta.values())
    state_dim = LE.PROPRIO_DIM + obj_pad_dim
    with open(os.path.join(RESULTS, "task_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)

    manifest = {
        "clusters": [], "obj_pad_dim": obj_pad_dim, "proprio_dim": LE.PROPRIO_DIM,
        "state_dim": state_dim, "action_dim": LE.ACTION_DIM,
        "n_train_per_cluster": N_TRAIN_PER_CLUSTER,
        "n_heldout_per_cluster": N_HELDOUT_PER_CLUSTER,
        "corpus_seed": 0, "heldout_seed": 1,
    }
    for c in clusters:
        tasks = c["tasks"]
        ndemos = {t: meta[t].get("num_demos", 0) for t in tasks}
        train = round_robin_select(tasks, ndemos, N_TRAIN_PER_CLUSTER)
        heldout = round_robin_select(tasks, ndemos, N_HELDOUT_PER_CLUSTER, skip=set(train))
        task_objs = {t: meta[t]["objects"] for t in tasks}
        probes = greedy_probe_tasks(tasks, task_objs)
        manifest["clusters"].append({
            "cluster": c["cluster"], "suite": c["suite"], "scene": c["scene"],
            "tasks": tasks,
            "train_demos": [{"task": t, "demo": d} for t, d in train],
            "heldout_demos": [{"task": t, "demo": d} for t, d in heldout],
            "probe_tasks": probes,
            "n_tasks": len(tasks),
        })
    with open(os.path.join(RESULTS, "corpus_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"[corpus] obj_pad_dim={obj_pad_dim} state_dim={state_dim}")
    for cm in manifest["clusters"]:
        print(f"  {cm['cluster']} {cm['suite']} scene={cm['scene']} "
              f"train={len(cm['train_demos'])} held={len(cm['heldout_demos'])} probes={cm['probe_tasks']}")
    return manifest


if __name__ == "__main__":
    build_corpus()
