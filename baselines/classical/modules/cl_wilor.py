"""Naive two-model ensemble: the network's finger articulation is slerped towards WiLoR's with
one global weight alpha. WiLoR runs on its demo crop (box 2.5x the hand, no perspective rectification), with no outlier
gate and no temporal smoothing of its output; global orientation, translation and shape stay the network's.
WiLoR inference reuses the release wrapper (H5 wilor.py); its per-segment output is cached under $CL_WILOR_CACHE so an
alpha sweep runs WiLoR once per (network, view)."""
import importlib.util
import os

import numpy as np

# trees are built outside the repository (make_trees.py), so the release root comes from the environment (run.sh sets it)
_SRC = os.path.join(os.environ["HANDRSI_ROOT"], "rsi/harness/trees/H5/modules/wilor.py")
_sp = importlib.util.spec_from_file_location("cl_wilor_base", _SRC)
W = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(W)
_run0 = W._run


def _run_cached(state, seg, params):
    d = os.environ.get("CL_WILOR_CACHE")
    if not d:
        return _run0(state, seg, params)
    os.makedirs(d, exist_ok=True)
    f = os.path.join(d, f"{os.path.splitext(os.path.basename(seg._video or 'x'))[0]}_{seg._start}_{seg.n}.npz")
    if os.path.isfile(f):
        z = np.load(f)
        state["wilor_k3"] = z["k3"]
        return z["hp"], z["kp2d"], z["ok"], z["go"]
    hp, kp2d, ok, go = _run0(state, seg, params)
    tmp = f"{f}.{os.getpid()}.tmp.npz"
    np.savez(tmp, hp=hp, kp2d=kp2d, ok=ok, go=go, k3=state["wilor_k3"])
    os.replace(tmp, f)
    return hp, kp2d, ok, go


W._run = _run_cached


def apply(state, seg, params):
    p = {"box_scale": 2.5, "perspective_crop": False, "gate": 1e9, "sig": 0.0}
    p.update(params)
    return W.apply(state, seg, p)
