"""Download exactly the LIBERO hdf5 files needed for RoboTDA-X clusters C1-C9.

- C1 libero_goal: full suite (need full 500-demo suite for Stage B/C quantity sweep).
- C2 libero_spatial, C3 libero_object, C4 libero_10: full suite (10 tasks each).
- C5-C9 libero_90: only the tasks in the 5 largest scene groups.

libero_object files already exist locally (CA-ICIL) -> symlink instead of re-download.
Writes done.marker when complete. Resumable (hf_hub_download resumes partial files).
"""
import os
import sys
import shutil
import time

sys.path.insert(0, os.path.dirname(__file__))
from clusters import resolve_clusters, hdf5_filename

DATA_ROOT = "/mnt/sdb/ljc/RoboTDA-X/data/libero"
REPO = "yifengzhu-hf/LIBERO-datasets"
EXISTING_OBJECT = "/mnt/sdb/ljc/CA-ICIL_URICL/data/libero/libero_object"


def needed_files():
    """Return dict suite -> set(filenames) required."""
    clusters = resolve_clusters()
    req = {}
    for c in clusters:
        suite = c["suite"]
        req.setdefault(suite, set())
        for t in c["tasks"]:
            req[suite].add(hdf5_filename(t))
    return req


def main():
    from huggingface_hub import hf_hub_download
    req = needed_files()
    total = sum(len(v) for v in req.values())
    print(f"[download] {total} files across {list(req.keys())}", flush=True)
    os.makedirs(DATA_ROOT, exist_ok=True)
    done = 0
    for suite, files in req.items():
        dest_dir = os.path.join(DATA_ROOT, suite)
        os.makedirs(dest_dir, exist_ok=True)
        for fn in sorted(files):
            dest = os.path.join(dest_dir, fn)
            if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
                done += 1
                continue
            # reuse pre-existing libero_object local copies
            if suite == "libero_object":
                src = os.path.join(EXISTING_OBJECT, fn)
                if os.path.exists(src) and os.path.getsize(src) > 1_000_000:
                    if not os.path.exists(dest):
                        os.symlink(src, dest)
                    done += 1
                    print(f"[symlink] {suite}/{fn}", flush=True)
                    continue
            t0 = time.time()
            path = hf_hub_download(
                repo_id=REPO, repo_type="dataset",
                filename=f"{suite}/{fn}",
                local_dir=DATA_ROOT,
            )
            done += 1
            sz = os.path.getsize(path) / 1e6
            print(f"[dl {done}/{total}] {suite}/{fn} {sz:.0f}MB {time.time()-t0:.0f}s", flush=True)
    # write manifest of what we have
    with open(os.path.join(DATA_ROOT, "download.done.marker"), "w") as f:
        f.write(f"downloaded {done}/{total} files\n")
    print(f"[download] DONE {done}/{total}", flush=True)


if __name__ == "__main__":
    main()
