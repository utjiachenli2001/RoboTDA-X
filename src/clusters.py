"""Deterministic cluster resolution for RoboTDA-X (metadata-only, from installed LIBERO benchmark).

M = 9 clusters:
  C1 = libero_goal, C2 = libero_spatial, C3 = libero_object, C4 = libero_10
  C5-C9 = 5 largest libero_90 scene groups by task count (ties -> alphabetical).

No task lists are hardcoded; everything is resolved at load time from the benchmark.
"""
import re
import collections

# Fixed suite clusters C1-C4
SUITE_CLUSTERS = [
    ("C1", "libero_goal"),
    ("C2", "libero_spatial"),
    ("C3", "libero_object"),
    ("C4", "libero_10"),
]


def _benchmark_dict():
    from libero.libero import benchmark
    return benchmark.get_benchmark_dict()


def suite_task_names(suite):
    """Ordered list of task names for a suite, as the installed benchmark reports them."""
    bm = _benchmark_dict()
    B = bm[suite]()
    return [t.name for t in B.tasks]


def scene_of(task_name):
    """Parse scene group name (e.g. KITCHEN_SCENE2) from a libero_90 task name."""
    m = re.match(r"([A-Z_]+SCENE\d+)", task_name)
    return m.group(1) if m else task_name.split("_demo")[0]


def libero90_scene_groups():
    """Return list of (scene, [tasks...]) sorted by (-count, scene) -- deterministic rank."""
    names = suite_task_names("libero_90")
    groups = collections.defaultdict(list)
    for n in names:
        groups[scene_of(n)].append(n)
    ranked = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    return ranked


def resolve_clusters():
    """Return ordered list of dicts: {cluster, suite, scene, tasks:[...]} for C1..C9.

    For C1-C4 the 'tasks' are the full suite tasks; scene is None.
    For C5-C9 the tasks are the scene group's tasks within libero_90.
    """
    out = []
    for cid, suite in SUITE_CLUSTERS:
        out.append({
            "cluster": cid,
            "suite": suite,
            "scene": None,
            "tasks": suite_task_names(suite),
        })
    ranked = libero90_scene_groups()
    top5 = ranked[:5]
    for i, (scene, tasks) in enumerate(top5):
        out.append({
            "cluster": f"C{5 + i}",
            "suite": "libero_90",
            "scene": scene,
            "tasks": sorted(tasks),  # deterministic within-scene order
        })
    return out


def hdf5_filename(task_name):
    """HF/robomimic convention: <task_name>_demo.hdf5"""
    return f"{task_name}_demo.hdf5"


if __name__ == "__main__":
    import json
    cl = resolve_clusters()
    for c in cl:
        print(c["cluster"], c["suite"], c["scene"], f"{len(c['tasks'])} tasks")
    print(json.dumps({c["cluster"]: c["tasks"] for c in cl}, indent=0)[:200])
