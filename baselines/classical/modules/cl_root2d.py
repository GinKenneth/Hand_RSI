"""Classical baseline: camera translation by linear least squares on the network's own 2D keypoints.

With the MANO joints J (root-relative to the translation) fixed, the translation t minimises the algebraic
reprojection error of J + t against the direct 2D head (two linear equations per joint, as in DLT / PnP for
translation only). The result is blended with the network's translation (alpha) and rejected when implausible.
"""
import numpy as np


def apply(state, seg, params):
    alpha = float(params.get("alpha", 1.0))
    fix_z = bool(params.get("fix_z", False))      # weak-perspective variant: keep the network's depth, fit x and y only
    K = np.asarray(seg.K, np.float64); fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    J = np.asarray(state["joints_cam"], np.float64) - np.asarray(state["transl"], np.float64)[:, :, None]
    uv = np.asarray(state["joints_2d"], np.float64)
    T = np.asarray(state["transl"], np.float64).copy()
    F = J.shape[0]
    for f in range(F):
        for s in (0, 1):
            X, u = J[f, s], uv[f, s]
            if not (np.isfinite(X).all() and np.isfinite(u).all()):
                continue
            du, dv = u[:, 0] - cx, u[:, 1] - cy
            A = np.zeros((42, 3)); b = np.zeros(42)
            A[:21, 0] = fx; A[:21, 2] = -du; b[:21] = du * X[:, 2] - fx * X[:, 0]
            A[21:, 1] = fy; A[21:, 2] = -dv; b[21:] = dv * X[:, 2] - fy * X[:, 1]
            if fix_z:
                tz = T[f, s, 2]
                t = np.array([np.mean((b[:21] + du * tz) / fx), np.mean((b[21:] + dv * tz) / fy), tz])
            else:
                t, *_ = np.linalg.lstsq(A, b, rcond=None)
            if not (0.05 < t[2] < 3.0) or np.linalg.norm(t - T[f, s]) > 0.3:
                continue
            T[f, s] = (1 - alpha) * T[f, s] + alpha * t
    state["transl"] = T.astype(np.float32)
    return seg.mano(state)
