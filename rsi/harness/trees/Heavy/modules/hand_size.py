"""hand_size: one guarded per-track hand-size estimate (replaces hand_scale; runs last).

Size (not proportions) of each hand track, from two independent estimates of the same physical quantity:
  * the network's own raw shape (seg.raw["betas"], before any shrinkage prior in the chain): MANO bone length of the
    track's mean raw betas;
  * WiLoR's metric 3D keypoints (state["wilor_k3"], tight crop): median bone length over frames.
The target log size is a fixed blend of the two; the posed hand is scaled about its wrist (joints and mesh together),
so rot / transl / 2D anchors do not move. Guard (from R2 a3): a frame keeps its size if scaling would change whether
any joint projects inside the image.
"""
import sys

import numpy as np

_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
          (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
_PA = [b[0] for b in _BONES]
_CH = [b[1] for b in _BONES]
_REF = {}


def _blen(X):
    return np.linalg.norm(X[..., _CH, :] - X[..., _PA, :], axis=-1).sum(-1)


def _mano_blen(side, betas):
    import rsi_paths  # noqa: F401  (puts ace_eval on sys.path)
    from ace_eval.mano import forward
    from harness.core import mano_models
    J, _ = forward(mano_models(), side, np.zeros((len(betas), 3)), np.zeros((len(betas), 45)), betas)
    return _blen(J.astype(np.float64))


def _inside(J, seg):
    z = J[..., 2]
    uv = seg.project(np.where(np.isfinite(J), J, 1.0))
    return ((z > 0.01) & (uv[..., 0] >= 0) & (uv[..., 0] < seg.W) & (uv[..., 1] >= 0) & (uv[..., 1] < seg.H)).any(-1)


def apply(state, seg, params):
    w_wilor = float(params.get("w_wilor", 0.3))
    w_raw = float(params.get("w_raw", 1.0))
    bias = float(params.get("bias", 0.0))
    a = float(params.get("a", 1.0))
    clip = float(params.get("clip", 0.15))
    min_frames = int(params.get("min_frames", 5))
    all_frames = bool(params.get("all_frames", True))
    guard = bool(params.get("guard", True))
    J = state["joints_cam"]
    V = state["verts_cam"]
    K3 = state.get("wilor_k3")
    if K3 is None and w_wilor > 0:
        sys.stderr.write("[hand_size] no WiLoR keypoints in the state; size from the network's raw shape only\n")
    rb = np.asarray(seg.raw.get("betas"), np.float64)[:len(J)] if "betas" in seg.raw else None
    fin = np.isfinite(J).all((2, 3)) & np.isfinite(V).all((2, 3))
    live = (state["exists"] > 0.5) & fin
    for s in (0, 1):
        m = live[:, s]
        if m.sum() < min_frames:
            continue
        la = _blen(J[m, s].astype(np.float64))
        la = la[la > 1e-4]
        if len(la) < min_frames:
            continue
        lla = float(np.log(la.mean()))
        terms, wts = [], []
        if rb is not None and w_raw > 0:
            b = rb[m, s].reshape(-1, rb.shape[-1])
            b = b[np.isfinite(b).all(1)]
            if len(b) >= min_frames:
                terms.append(float(np.log(_mano_blen(s, b.mean(0, keepdims=True))[0])) - lla)
                wts.append(w_raw)
        if K3 is not None and w_wilor > 0:
            both = m & np.isfinite(K3[:, s]).all((1, 2))
            if both.sum() >= min_frames:
                lw = _blen(K3[both, s].astype(np.float64))
                lw = lw[lw > 1e-4]
                if len(lw) >= min_frames:
                    terms.append(float(np.log(np.median(lw))) - lla)
                    wts.append(w_wilor)
        if not terms:
            continue
        r = a * float(np.dot(wts, terms) / np.sum(wts)) + bias
        k = float(np.exp(np.clip(r, -clip, clip)))
        if abs(k - 1.0) < 1e-6:
            continue
        sel = fin[:, s] if all_frames else m
        idx = np.nonzero(sel)[0]
        Jo = J[idx, s].astype(np.float64)
        c = Jo[:, :1]
        Jn = c + k * (Jo - c)
        if guard:
            keep = _inside(Jn, seg) == _inside(Jo, seg)
            idx = idx[keep]
            Jn = Jn[keep]
            c = c[keep]
        J[idx, s] = Jn.astype(J.dtype)
        V[idx, s] = (c + k * (V[idx, s].astype(np.float64) - c)).astype(V.dtype)
    state["joints_cam"] = J
    state["verts_cam"] = V
    return state
