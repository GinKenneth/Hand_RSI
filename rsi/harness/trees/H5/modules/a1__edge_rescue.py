"""edge_rescue: keep alive hands that the network places just outside the image.

Observation (HOT3D dev, parent = empty tree): 94% of the missed on-screen hands are hands the network DOES keep
(exists > 0.5) but whose MANO mesh projects entirely outside the image, so the evaluator's on-screen gate drops them.
They are edge hands (71% have 1-3 GT joints in frame). Conditioned on the prediction, near-border hands are biased
outward (median +9 px radial error for hands with 1-3 projected joints in frame), the usual shrink-toward-the-prior
effect for truncated objects. Among kept hands whose nearest joint is outside the frame by less than ~7.5% of the
hand's own projected size, the GT hand is on-screen in ~60% of the cases; beyond that it drops quickly.

Gate (v2, after the first submission regressed FAcc/F1 on val): rescue only when the network's own in-frame
visibility head agrees (exists_2d > e2_min; 0.01 in v2, 0.001 in v3) and the hand is outside by <= rel_max = 5% of its projected size
(optionally any hand within rel_any). v3: hands with exists_2d in (0.001, 0.01] that are this close to the
frame are on-screen in 77% of dev cases (159/207) and FAcc improves or holds in every HOT3D dev sequence. On dev the e2-gated rule keeps a higher per-sequence precision (worst sequence
0.46 vs 0.24 for the geometric rule alone): 2D visibility and 3D placement disagree, and the visibility head wins.

Rule: for such a hand, translate it parallel to the image plane (transl, joints and verts together; pose, shape and
orientation untouched) so that its nearest joint lands `margin_px` inside the frame. The per-frame shift is smoothed
in time (Gaussian, sigma frames) so that rescued frames do not add acceleration; rescued frames are re-checked to be
in frame after smoothing. Nothing else changes.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d


def _outdist(uv, z, W, H):
    dx = np.maximum(np.maximum(-uv[..., 0], uv[..., 0] - W), 0)
    dy = np.maximum(np.maximum(-uv[..., 1], uv[..., 1] - H), 0)
    return np.where(z > 0.01, np.hypot(dx, dy), np.inf)


def _needed_shift(J, uv, od, K, W, H, m):
    j = int(np.argmin(od))
    u, v = uv[j]
    tu = np.clip(u, m, W - 1 - m)
    tv = np.clip(v, m, H - 1 - m)
    zj = J[j, 2]
    return np.array([(tu - u) * zj / K[0, 0], (tv - v) * zj / K[1, 1], 0.0])


def apply(state, seg, params):
    rel_max = float(params.get('rel_max', 0.05))
    e2_min = float(params.get('e2_min', 0.01))
    rel_any = float(params.get('rel_any', 0.0))
    e2 = np.asarray(seg.raw['exists_2d'], np.float64)[:state['exists'].shape[0]].reshape(-1, 2)
    m = float(params.get('margin_px', 1.0))
    sigma = float(params.get('sigma', 1.0))
    K = seg.K
    W, H = seg.W, seg.H
    J = state['joints_cam'].astype(np.float64)
    F = J.shape[0]
    uv = seg.project(J)
    od_all = _outdist(uv, J[..., 2], W, H)
    shift = np.zeros((F, 2, 3))
    need = np.zeros((F, 2), bool)
    for t in range(F):
        for s in (0, 1):
            if state['exists'][t, s] <= 0.5 or not np.all(np.isfinite(J[t, s])):
                continue
            d = od_all[t, s]
            if np.any(d == 0) or not np.isfinite(d.min()):
                continue
            size = np.linalg.norm(uv[t, s].max(0) - uv[t, s].min(0))
            rel = d.min() / max(size, 1.0)
            if not ((rel <= rel_max and e2[t, s] > e2_min) or rel <= rel_any):
                continue
            need[t, s] = True
            shift[t, s] = _needed_shift(J[t, s], uv[t, s], d, K, W, H, m)
    if not need.any():
        return state
    if sigma > 0:
        sm = gaussian_filter1d(shift, sigma, axis=0, mode='nearest')
        for s in (0, 1):
            if not need[:, s].any():
                sm[:, s] = 0
                continue
            for t in np.nonzero(need[:, s])[0]:
                Jt = J[t, s] + sm[t, s]
                uvt = seg.project(Jt)
                dt = _outdist(uvt, Jt[:, 2], W, H)
                if not np.any(dt == 0):
                    sm[t, s] += _needed_shift(Jt, uvt, dt, K, W, H, m)
        shift = sm
    state['transl'] = (state['transl'] + shift).astype(np.float32)
    state['joints_cam'] = state['joints_cam'] + shift[:, :, None]
    state['verts_cam'] = state['verts_cam'] + shift[:, :, None]
    return state
