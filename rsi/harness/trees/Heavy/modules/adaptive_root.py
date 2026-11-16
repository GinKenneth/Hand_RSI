"""Regularize root motion while retaining sharp, observed hand movements.

The mean joint location is less sensitive to articulation around the wrist.
A robust acceleration prior smooths this center, relaxing around rapid motion.
A small medium-frequency raw-root residual retains independent placement
evidence; the slow part is already handled by the rigid placement refinement.
"""
import numpy as np
from scipy.linalg import solve_banded
from scipy.ndimage import gaussian_filter1d, minimum_filter1d


def _limit(x, radius):
    return x * np.minimum(1.0, radius / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12))


def _smooth(x, strength, scale):
    n = len(x)
    d = np.diff(np.eye(n), n=2, axis=0)
    out = x.copy()
    for _ in range(3):
        acceleration = d @ out
        weight = 1.0 / np.sqrt(1.0 + (np.linalg.norm(acceleration, axis=-1) / scale) ** 2)
        a = np.eye(n) + strength * (d.T * weight) @ d
        band = np.zeros((5, n))
        for k in range(-2, 3):
            band[2-k, max(k, 0):min(n, n+k)] = np.diag(a, k)
        out = solve_banded((2, 2), band, x)
    return out


def apply(state, seg, params):
    j = np.asarray(state['joints_cam'], np.float64)
    t = np.asarray(state['transl'], np.float64)
    n = len(t)
    if n < 4:
        return state
    valid = np.isfinite(j).all((2, 3)) & np.isfinite(t).all(-1)
    uv = seg.project(np.where(valid[..., None, None], j, 0.0))
    margin = np.minimum.reduce([uv[..., 0], seg.W-uv[..., 0],
                                uv[..., 1], seg.H-uv[..., 1]]).min(-1)
    visibility = np.asarray(seg.raw.get('exists_2d', np.ones((n, 2))), np.float64)[:n].reshape(n, 2)
    confidence = np.clip(margin / 20.0, 0.0, 1.0) * (state['exists'] > 0.5) * valid
    confidence *= np.nan_to_num(np.clip(visibility, 0.0, 1.0))
    confidence = minimum_filter1d(confidence, size=9, axis=0, mode='nearest')
    gate = gaussian_filter1d(confidence, 3.0, axis=0, mode='nearest') * valid
    if not np.any(gate > 1e-6):
        return state

    medium = np.zeros_like(t)
    raw_weight = float(params.get('raw_weight', 0.25))
    if raw_weight > 0 and 'cam_trans' in seg.raw:
        residual = np.asarray(seg.raw['cam_trans'], np.float64)[:n].reshape(n, 2, 3) - t
        residual = np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)
        time = np.linspace(-1.0, 1.0, n)
        basis = np.stack([np.ones(n), time, time*time], -1)
        slow = np.zeros_like(residual)
        for side in (0, 1):
            weighted = basis.T * confidence[:, side]
            slow[:, side] = basis @ np.linalg.solve(weighted @ basis + 1e-6*np.eye(3), weighted @ residual[:, side])
        medium = gaussian_filter1d(residual, 2.0, axis=0, mode='nearest') - slow

    center = j.mean(-2)
    shift = np.zeros_like(t)
    strength = float(params.get('strength', 10.0))
    scale = max(float(params.get('acceleration_scale', 0.0005)), 1e-6)
    for side in (0, 1):
        good = valid[:, side]
        if good.sum() < 4:
            continue
        source = center[:, side] + raw_weight * medium[:, side]
        if not good.all():
            for axis in range(3):
                source[:, axis] = np.interp(np.arange(n), np.flatnonzero(good), source[good, axis])
        shift[:, side] = _smooth(source, strength, scale) - source + raw_weight * medium[:, side]
    shift = _limit(shift * gate[..., None], float(params.get('max_shift', 0.01)))
    placed = j + shift[:, :, None]
    after = seg.project(placed)

    def visible(points, projection):
        return ((points[..., 2] > 0.01) & (projection[..., 0] >= 0) &
                (projection[..., 0] < seg.W) & (projection[..., 1] >= 0) &
                (projection[..., 1] < seg.H)).any(-1)

    keep = valid & (visible(j, uv) == visible(placed, after))
    shift = np.where(keep[..., None], shift, 0.0)
    state['transl'] = t + shift
    state['joints_cam'] = j + shift[:, :, None]
    state['verts_cam'] = state['verts_cam'] + shift[:, :, None]
    return state
