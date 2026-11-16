"""Run a harness tree over captured segments -> evaluator prediction npz per segment.

  python -m harness.run --tree TREE --cache CAPTURE_DIR --view VIEW_DIR --out PRED_DIR [--jobs 16] [--seqs a,b]

VIEW_DIR is a dataset view in the ACE-protocol layout (segments/<seq>.json with K, W, H, video and the segment list).
CAPTURE_DIR holds <segment>.npz with the raw predict_video() arrays (capture/run_ace.py). Modules run segment by
segment; the process pool only parallelises segments. Run agent-written trees inside the sandbox.
"""
from __future__ import annotations

import os

# One BLAS / OpenMP / torch-CPU thread per process, set before numpy or torch is imported.
# The parallelism is the worker pool: with the libraries' default (a thread per core, ~255 here) 16-24 workers per replay
# and several concurrent replays meant ~24k threads on 384 cores and 10-20x slower replays.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
# WiLoR for image-evidence modules (brief: read-only under $WILOR_MINI_ROOT). Every replay (agent sessions, gate, Loop B
# targets, round stages) needs the variable: without it a module using WiLoR crashes (ultralytics then tries to
# download the missing weights, without network). Default: rsi/third_party/wilor_root.
if not os.path.isdir(os.environ.get("WILOR_MINI_ROOT", "") or "/nonexistent"):
    _w = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "wilor_root")
    if os.path.isdir(os.path.join(_w, "wilor_mini")):
        os.environ["WILOR_MINI_ROOT"] = _w

import argparse
import glob
import json
import multiprocessing as mp
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.core import Segment, load_chain, run_segment  # noqa: E402

_CHAIN = None


def _cards(min_free_mib=8000):
    """Cards a worker may put an image model on: those visible (CUDA_VISIBLE_DEVICES, else every /dev/nvidiaN) with at
    least min_free_mib free (the host may be shared). Queried in the parent, so no worker initialises CUDA to find out."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    cards = [c for c in vis.split(",") if c.strip()] if vis is not None else \
        sorted((p[len("/dev/nvidia"):] for p in glob.glob("/dev/nvidia[0-9]*")), key=int)
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30).stdout
        free = {i.strip(): int(f) for i, f in (l.split(",") for l in q.strip().splitlines())}
        roomy = [c for c in cards if free.get(c, 0) >= min_free_mib]
        return roomy or cards
    except Exception:
        return cards


def _init(tree, cards=None, counter=None):
    """Worker start. Worker i sees only card cards[i % n]: a module that runs an image model on
    cuda (e.g. WiLoR, ~4.4 GB per process) in 16-32 workers would otherwise put them all on card 0 and run out of memory."""
    global _CHAIN
    if cards and counter is not None:                 # also with one card: it may not be card 0 (the only roomy one)
        with counter.get_lock():
            i = counter.value; counter.value += 1
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = cards[i % len(cards)]
    _CHAIN = load_chain(Path(tree))


def _one(job):
    seg_name, raw_path, K, W, H, video, start, end, out = job
    raw = dict(np.load(raw_path))
    seg = Segment(raw, K, W, H, video=video, start=start, n=end - start)
    pred = run_segment(_CHAIN, seg)
    np.savez_compressed(out, **pred)
    return seg_name


def jobs_for(view: Path, cache: Path, out: Path, seqs=None):
    jobs = []
    for sj in sorted((view / "segments").glob("*.json")):
        if seqs and sj.stem not in seqs:
            continue
        m = json.loads(sj.read_text())
        video = str(view / "videos" / Path(m["video"]).name)
        for sg in m["segments"]:
            raw = cache / f"{sg['name']}.npz"
            if not raw.is_file():
                raise FileNotFoundError(f"capture missing: {raw}")
            jobs.append((sg["name"], str(raw), m["K"], m["W"], m["H"], video, sg["start"], sg["end"], str(out / f"{sg['name']}.npz")))
    return jobs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", required=True, type=Path)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--view", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--seqs", default="")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    js = jobs_for(a.view, a.cache, a.out, set(a.seqs.split(",")) if a.seqs else None)
    t0 = time.time()
    if a.jobs <= 1:
        _init(a.tree)
        for j in js:
            _one(j)
    else:
        ctx = mp.get_context("fork")                       # the counter is inherited, not pickled
        with ProcessPoolExecutor(a.jobs, mp_context=ctx, initializer=_init,
                                 initargs=(str(a.tree), _cards(), ctx.Value("i", 0))) as ex:
            list(ex.map(_one, js, chunksize=4))
    print(f"{len(js)} segments -> {a.out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
