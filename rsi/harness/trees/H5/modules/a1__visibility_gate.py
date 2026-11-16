"""a1__visibility_gate: a hand placed fully inside the image must also be seen by the 2D visibility head.

ACE outputs two existence cues per hand: exists_3d (used as the existence score) and exists_2d, the probability that
the hand is visible in the image. When the 3D placement puts every joint of a hand inside the image, the hand would
be fully visible, so a visibility score below 0.5 (temporally smoothed, sigma frames) contradicts the placement; such
a hand is marked not present (existence lowered to <= exist_floor). Hands that are only partly inside are left alone:
there the visibility head is naturally lower (truncation) and the placement decides. Pose and placement unchanged.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d


def apply(state, seg, params):
    thr = float(params.get('vis_thr', 0.5))
    sigma = float(params.get('sigma', 1.0))
    floor = float(params.get('exist_floor', 0.5))
    min_in = int(params.get('min_joints_in', 21))
    F = state['exists'].shape[0]
    e2 = np.asarray(seg.raw['exists_2d'], np.float64)[:F].reshape(F, 2)
    if sigma > 0:
        e2 = gaussian_filter1d(e2, sigma, axis=0, mode='nearest')
    J = np.asarray(state['joints_cam'], np.float64)
    fin = np.isfinite(J).all((-1, -2))
    Jf = np.where(fin[..., None, None], J, 1.0)
    uv = seg.project(Jf)
    inside = (Jf[..., 2] > 0.01) & (uv[..., 0] >= 0) & (uv[..., 0] < seg.W) & (uv[..., 1] >= 0) & (uv[..., 1] < seg.H)
    drop = fin & (inside.sum(-1) >= min_in) & (e2 < thr) & (state['exists'] > floor)
    if drop.any():
        ex = state['exists'].copy()
        ex[drop] = floor
        state['exists'] = ex
    return state
