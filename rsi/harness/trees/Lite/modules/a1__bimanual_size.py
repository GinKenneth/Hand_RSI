"""a1__bimanual_size: one wearer, two hands -> pull both hands' sizes toward their reliability-weighted consensus.

Re-implemented at integration (R4) from a1's notes (the module itself was never submitted): each hand's chain size is
the hand_size blend of its raw-betas size and WiLoR's metric size, so |log chain size - log WiLoR size| is that hand's
own disagreement (its reliability cue). The two hands of one segment belong to the same person; their log sizes are
fused by inverse variance (var = disagreement^2 + eps) and each hand is scaled about its wrist toward the consensus by
lam * v / (v + tau^2) (a hand whose two estimates already agree stays), clipped to +-clip. Same in-image guard as
hand_size: a frame keeps its size if scaling would change whether any joint projects inside the image.
Joints and mesh scale together about the wrist; rot / transl / 2D anchors are unchanged. NumPy only, no inference.
"""
import numpy as np

_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
          (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
_PA = [b[0] for b in _BONES]
_CH = [b[1] for b in _BONES]


def _blen(X):
    return np.linalg.norm(X[..., _CH, :] - X[..., _PA, :], axis=-1).sum(-1)


def _inside(J, seg):
    uv = seg.project(J)
    return ((J[..., 2] > 0.01) & (uv[..., 0] >= 0) & (uv[..., 0] < seg.W) & (uv[..., 1] >= 0)
            & (uv[..., 1] < seg.H)).any(-1)


def apply(state, seg, params):
    lam = float(params.get("lam", 0.75))
    eps = float(params.get("eps", 1e-4))
    clip = float(params.get("clip", 0.08))
    tau = float(params.get("tau", 0.01))
    min_frames = int(params.get("min_frames", 5))
    guard = bool(params.get("guard", True))
    K3 = state.get("wilor_k3")
    if K3 is None:
        return state
    J = state["joints_cam"]
    V = state["verts_cam"]
    fin = np.isfinite(J).all((2, 3)) & np.isfinite(V).all((2, 3))
    live = (state["exists"] > 0.5) & fin
    lc, var = {}, {}
    for s in (0, 1):
        m = live[:, s]
        both = m & np.isfinite(K3[:, s]).all((1, 2))
        if m.sum() < min_frames or both.sum() < min_frames:
            return state
        la = _blen(J[m, s].astype(np.float64))
        lw = _blen(K3[both, s].astype(np.float64))
        la, lw = la[la > 1e-4], lw[lw > 1e-4]
        if len(la) < min_frames or len(lw) < min_frames:
            return state
        lc[s] = float(np.log(la.mean()))
        var[s] = (lc[s] - float(np.log(np.median(lw)))) ** 2 + eps
    w = {s: 1.0 / var[s] for s in (0, 1)}
    cons = (w[0] * lc[0] + w[1] * lc[1]) / (w[0] + w[1])
    for s in (0, 1):
        g = lam * var[s] / (var[s] + tau ** 2) if tau > 0 else lam
        r = float(np.clip(g * (cons - lc[s]), -clip, clip))
        k = float(np.exp(r))
        if abs(k - 1.0) < 1e-6:
            continue
        idx = np.nonzero(fin[:, s])[0]
        Jo = J[idx, s].astype(np.float64)
        c = Jo[:, :1]
        Jn = c + k * (Jo - c)
        if guard:
            keep = _inside(Jn, seg) == _inside(Jo, seg)
            idx, Jn, c = idx[keep], Jn[keep], c[keep]
        J[idx, s] = Jn.astype(J.dtype)
        V[idx, s] = (c + k * (V[idx, s].astype(np.float64) - c)).astype(V.dtype)
    state["joints_cam"] = J
    state["verts_cam"] = V
    return state
