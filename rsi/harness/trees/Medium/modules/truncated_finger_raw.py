"""Out-of-view fingers of partly visible hands: blend their articulation back toward the network's own (a4, R5).

Why: the image tail (WiLoR articulation, 2D anchor bands, finger-only fits) changes a finger's articulation from
image evidence. A finger whose joints project OUTSIDE the image has no such evidence. Its image-driven change is
then noise, so the network's own temporal articulation prior (raw `hand_pose`) is the better estimate. This is the
per-finger version of a8's `a8__oos_raw_articulation` (whole hand fully out of view), for hands that are only partly
in view.
What: for every predicted hand with at least one joint in the image, a finger counts as out of view when >= `min_out`
of its 4 joints project outside the image. Its 3 MANO joint rotations are moved toward the raw network rotations by
`weight` (geodesic, per joint: R <- exp(w log(R_raw R^T)) R). The weight is smoothed in time (Gaussian `w_sigma`
inside each run of predicted, in-view frames). The correction rotation vector is smoothed too (`d_sigma`), so a
finger leaving or entering the image fades in and out without a jump (Jitter).
Only the articulation changes: wrist, orientation, shape and translation stay fixed. `joints_cam`/`verts_cam` get the
MANO difference, so the 2D anchors are untouched. A hand-frame whose in-image status would flip is left unchanged
(detection is not this module's business).
Dev (replay of the H4* tail, 624 HOT3D / 262 ARCTIC segments): HOT3D S +0.076 %, PA-p +0.42 %, 15/15 recordings
better; ARCTIC S -0.009 %. Original NumPy/SciPy code (BSD-3-Clause numpy/scipy), no external code.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

# finger -> (MANO hand_pose indices, OpenPose joint indices)
FINGERS = [((0, 1, 2), (5, 6, 7, 8)), ((3, 4, 5), (9, 10, 11, 12)), ((6, 7, 8), (17, 18, 19, 20)),
           ((9, 10, 11), (13, 14, 15, 16)), ((12, 13, 14), (1, 2, 3, 4))]


def _inside(seg, X, m=0.):
    uv = seg.project(X)
    return (X[..., 2] > .01) & (uv[..., 0] >= m) & (uv[..., 0] < seg.W - m) & (uv[..., 1] >= m) & (uv[..., 1] < seg.H - m)


def _runs(mask):
    out, i, n = [], 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        out.append((i, j + 1))
        i = j + 1
    return out


def apply(state, seg, params):
    w = float(params.get("weight", 0.5))
    min_out = int(params.get("min_out", 1))
    margin = float(params.get("margin_px", 0.0))
    ws = float(params.get("w_sigma", 3.0))
    dsig = float(params.get("d_sigma", 2.0))
    raw = np.asarray(seg.raw.get('hand_pose', []), np.float64)
    if raw.size != state['hp'].size or w <= 0:
        return state
    raw = raw.reshape(state['hp'].shape)
    J = np.asarray(state['joints_cam'])
    F = len(J)
    ins = _inside(seg, J, margin)
    hp = np.asarray(state['hp'], np.float64)
    hand_ok = ((np.asarray(state['exists']) > .5) & ins.any(-1) & np.isfinite(J).all((-1, -2))
               & np.isfinite(raw).all((-1, -2, -3)) & np.isfinite(hp).all((-1, -2, -3)))
    wt = np.zeros((F, 2, 15))
    for hpi, opi in FINGERS:
        out = (~ins[:, :, list(opi)]).sum(-1) >= min_out
        for k in hpi:
            wt[:, :, k] = np.where(hand_ok & out, w, 0.)
    runs = {s: _runs(hand_ok[:, s]) for s in (0, 1)}
    if ws > 0:
        for s in (0, 1):
            for a, b in runs[s]:
                wt[a:b, s] = gaussian_filter1d(wt[a:b, s], ws, axis=0, mode='nearest')
    wt[~hand_ok] = 0.
    if not np.any(wt > 1e-3):
        return state
    delta = raw @ hp.swapaxes(-1, -2)
    delta[~hand_ok] = np.eye(3)
    rv = Rotation.from_matrix(delta.reshape(-1, 3, 3)).as_rotvec().reshape(F, 2, 15, 3)
    if dsig > 0:
        for s in (0, 1):
            for a, b in runs[s]:
                rv[a:b, s] = gaussian_filter1d(rv[a:b, s], dsig, axis=0, mode='nearest')
    corr = Rotation.from_rotvec((wt[..., None] * rv).reshape(-1, 3)).as_matrix().reshape(F, 2, 15, 3, 3)
    old = {k: np.asarray(state[k]).copy() for k in ("go", "hp", "betas", "transl")}
    new = dict(old, hp=corr @ hp)
    A = seg.mano(dict(old, joints_cam=None, verts_cam=None))
    B = seg.mano(dict(new, joints_cam=None, verts_cam=None))
    dJ = np.nan_to_num(np.asarray(B['joints_cam'], np.float64) - A['joints_cam'])
    dV = np.nan_to_num(np.asarray(B['verts_cam'], np.float64) - A['verts_cam'])
    touched = (wt > 1e-3).any(-1)
    touched &= _inside(seg, J + dJ).any(-1) == ins.any(-1) if margin == 0 else _inside(seg, J + dJ).any(-1) == _inside(seg, J).any(-1)
    if not touched.any():
        return state
    state['hp'] = np.where(touched[..., None, None, None], new['hp'], state['hp']).astype(np.asarray(state['hp']).dtype)
    state['joints_cam'] = (J + np.where(touched[..., None, None], dJ, 0)).astype(J.dtype)
    state['verts_cam'] = (state['verts_cam'] + np.where(touched[..., None, None], dV, 0)).astype(np.asarray(state['verts_cam']).dtype)
    return state
