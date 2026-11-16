"""Temporal refinement of the MANO parameters inside one segment.

Local polynomial (Savitzky-Golay-like, Gaussian-weighted) regression over time of
  - the global orientation (in the tangent space of each frame's own rotation),
  - the 15 hand-pose joint rotations (same, per joint),
  - the root translation (in camera coordinates),
using only the frames where the hand is predicted (exists > 0.5), weighted by the existence score. A local quadratic
fit removes frame-to-frame noise without the lag / flattening bias of a plain moving average (and is unbiased at the
segment ends, where a padded filter is not). joints_cam / verts_cam are recomputed with MANO afterwards."""
import numpy as np
from scipy.spatial.transform import Rotation as Rot


def _weights(idx, f, sg, conf):
    w = np.exp(-0.5 * ((idx - f) / sg) ** 2) * conf[idx]
    keep = w > 1e-3
    return idx[keep], w[keep]


def _fit(ii, ww, f, v, deg):
    X = np.stack([(ii - f) ** k for k in range(deg + 1)], 1).astype(float)
    XtW = X.T * ww
    A = XtW @ X + 1e-9 * np.eye(deg + 1)
    return np.linalg.solve(A, XtW @ v.reshape(len(ii), -1))[0]


def smooth_rot(R, valid, conf, sg, deg):
    """R (F,J,3,3) -> smoothed, only frames in valid are changed / used."""
    F, J = R.shape[:2]
    out = R.copy()
    idx = np.where(valid)[0]
    if sg <= 0 or len(idx) < deg + 2:
        return out
    for f in idx:
        ii, ww = _weights(idx, f, sg, conf)
        if len(ii) < deg + 2:
            continue
        D = np.einsum('jki,njkl->njil', R[f], R[ii])           # R_f^T R_t, body-frame residuals
        v = Rot.from_matrix(D.reshape(-1, 3, 3)).as_rotvec().reshape(len(ii), J * 3)
        a = _fit(ii, ww, f, v, deg).reshape(J, 3)
        out[f] = np.einsum('jik,jkl->jil', R[f], Rot.from_rotvec(a).as_matrix())
    return out


def smooth_vec(x, valid, conf, sg, deg):
    out = x.copy()
    idx = np.where(valid)[0]
    if sg <= 0 or len(idx) < deg + 2:
        return out
    for f in idx:
        ii, ww = _weights(idx, f, sg, conf)
        if len(ii) < deg + 2:
            continue
        out[f] = _fit(ii, ww, f, x[ii], deg).reshape(x.shape[1:])
    return out


def apply(state, seg, params):
    gs, gd = float(params.get("go_sigma", 2.0)), int(params.get("go_deg", 2))
    hs, hd = float(params.get("hp_sigma", 0.0)), int(params.get("hp_deg", 2))
    ts, td = float(params.get("tr_sigma", 0.0)), int(params.get("tr_deg", 2))
    ks, kd = float(params.get("kp2d_sigma", 0.0)), int(params.get("kp2d_deg", 2))
    thr = float(params.get("exist_thr", 0.5))
    for s in (0, 1):
        ex = state["exists"][:, s].astype(float)
        fin = np.isfinite(state["go"][:, s]).all((1, 2)) & np.isfinite(state["hp"][:, s]).all((1, 2, 3)) & \
            np.isfinite(state["transl"][:, s]).all(1)
        valid = (ex > thr) & fin
        conf = np.clip(ex, 1e-3, 1.0)
        if gs > 0:
            state["go"][:, s] = smooth_rot(state["go"][:, s][:, None], valid, conf, gs, gd)[:, 0]
        if hs > 0:
            state["hp"][:, s] = smooth_rot(state["hp"][:, s], valid, conf, hs, hd)
        if ts > 0:
            state["transl"][:, s] = smooth_vec(state["transl"][:, s].astype(float), valid, conf, ts, td).astype(state["transl"].dtype)
        if ks > 0:
            k2 = state["joints_2d"][:, s].astype(float)
            v2 = valid & np.isfinite(k2).all((1, 2))
            state["joints_2d"][:, s] = smooth_vec(k2, v2, conf, ks, kd).astype(state["joints_2d"].dtype)
    return seg.mano(state)
