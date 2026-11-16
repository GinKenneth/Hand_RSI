"""Estimate hand placement from the frozen network's two independent 3D heads.

The direct wrist is converted to the current MANO root convention. A robust
acceleration prior acts on the observed mean hand center. Translation updates
are bounded, reduced during overlapping-hand occlusion, and constrained to
preserve projected hand presence throughout the displacement.

An optional joint-wise prior holds articulation fixed and accounts for each
joint's acceleration. High-frequency disagreement between the independent
placement heads reduces trust in an observation without confusing a constant
head offset with temporal localization noise.
"""
import sys
import numpy as np
from scipy.linalg import solve_banded
from scipy.ndimage import gaussian_filter1d, minimum_filter1d


def _smooth(x, strength, scale):
    n = len(x)
    d = np.diff(np.eye(n), n=2, axis=0)
    out = x.copy()
    for _ in range(3):
        a2 = d @ out
        w = 1.0 / np.sqrt(1.0 + (np.linalg.norm(a2, axis=-1) / scale) ** 2)
        a = np.eye(n) + strength * (d.T * w) @ d
        band = np.zeros((5, n))
        for k in range(-2, 3):
            band[2-k, max(k, 0):min(n, n+k)] = np.diag(a, k)
        out = solve_banded((2, 2), band, x)
    return out


def _smooth_joints(observed, relative, strength, scale, confidence):
    n = len(observed)
    d = np.diff(np.eye(n), n=2, axis=0)
    local_acceleration = np.einsum('ij,jkc->ikc', d, relative)
    out = observed.copy()
    for _ in range(4):
        acceleration = (d @ out)[:, None] + local_acceleration
        weight = 1.0 / np.sqrt(1.0 + (np.linalg.norm(acceleration, axis=-1) / scale) ** 2)
        a = np.diag(confidence) + strength * (d.T * weight.mean(-1)) @ d
        rhs = observed * confidence[:, None]
        rhs -= strength * d.T @ (weight[..., None] * local_acceleration).mean(1)
        band = np.zeros((5, n))
        for k in range(-2, 3):
            band[2-k, max(k, 0):min(n, n+k)] = np.diag(a, k)
        out = solve_banded((2, 2), band, rhs)
    return out


def _visibility_limit(joints, shift, seg, sigma):
    # Each camera-frustum inequality is linear along a translation. Restrict
    # the complete prefix of the displacement, including image-corner crossings.
    fx, fy = seg.K[0, 0], seg.K[1, 1]
    cx, cy = seg.K[0, 2], seg.K[1, 2]
    def inequalities(x):
        px, py, pz = x[..., 0], x[..., 1], x[..., 2]
        return np.stack([pz, fx*px+cx*pz, (seg.W-cx)*pz-fx*px,
                         fy*py+cy*pz, (seg.H-cy)*pz-fy*py], -1)
    start = inequalities(joints)
    start[..., 0] -= 0.01
    direction = inequalities(shift)[:, :, None]
    zero = np.abs(direction) < 1e-14
    ratio = -start / np.where(zero, 1.0, direction)
    lower = np.maximum(0.0, np.max(np.where(direction > 1e-14, ratio, -np.inf), -1))
    upper = np.minimum(1.0, np.min(np.where(direction < -1e-14, ratio, np.inf), -1))
    possible = (lower < upper) & ~np.any(zero & (start <= 0), -1)
    inside = (start > 0).all(-1)
    before = inside.any(-1)
    leave = np.max(np.where(inside, upper, 0.0), -1)
    enter = np.min(np.where(possible, lower, 1.0), -1)
    cap = np.clip(np.where(before, leave, enter), 0.0, 1.0)
    alpha = np.where(cap < 1.0, 0.98 * cap, 1.0)
    if sigma > 0:
        radius = int(4.0 * sigma + 0.5)
        alpha = minimum_filter1d(alpha, 2*radius+1, axis=0, mode='nearest')
        alpha = gaussian_filter1d(alpha, sigma, axis=0, mode='nearest')
    return alpha


def apply(state, seg, params):
    t = np.asarray(state['transl'], np.float64)
    j = np.asarray(state['joints_cam'], np.float64)
    n = len(t)
    if n < 4:
        return state
    if 'cam_trans' not in seg.raw or 'direct_wrist_cam' not in seg.raw:
        print('root_source: missing raw placement head; passing through', file=sys.stderr)
        return state
    raw = np.asarray(seg.raw['cam_trans'], np.float64)[:n].reshape(n, 2, 3)
    wrist = np.asarray(seg.raw['direct_wrist_cam'], np.float64)[:n].reshape(n, 2, 3)
    weight = float(params.get('direct_weight', 0.15))
    direct = wrist - (j[:, :, 0] - t)
    observed = (1.0-weight) * raw + weight * direct
    offset = j.mean(2) - t
    valid = np.isfinite(observed).all(-1) & np.isfinite(j).all((2, 3)) & np.isfinite(t).all(-1)
    confidence = np.ones((n, 2))
    uncertainty_scale = float(params.get('uncertainty_scale', 0.0))
    if uncertainty_scale > 0:
        disagreement = np.where(valid[..., None], raw - direct, 0.0)
        disagreement -= gaussian_filter1d(disagreement, 3.0, axis=0)
        confidence = np.maximum(0.1, 1.0 / (1.0 +
            (np.linalg.norm(disagreement, axis=-1) / uncertainty_scale) ** 2))
    shift = np.zeros_like(t)
    for side in (0, 1):
        good = valid[:, side]
        if good.sum() < 4:
            continue
        relative = j[:, side] - t[:, side, None]
        source = observed[:, side].copy()
        if not good.all():
            for axis in range(3):
                source[:, axis] = np.interp(np.arange(n), np.flatnonzero(good), source[good, axis])
                for joint in range(relative.shape[1]):
                    relative[:, joint, axis] = np.interp(np.arange(n), np.flatnonzero(good), relative[good, joint, axis])
        strength = float(params.get('strength', 30.0))
        scale = max(float(params.get('acceleration_scale', 0.001)), 1e-6)
        if params.get('joint_acceleration', False):
            target = _smooth_joints(source, relative, strength, scale, confidence[:, side])
        else:
            center_offset = relative.mean(1)
            target = _smooth(source + center_offset, strength, scale) - center_offset
        shift[:, side] = target - t[:, side]
    shift *= np.minimum(1.0, float(params.get('max_shift', 0.02)) /
                        np.maximum(np.linalg.norm(shift, axis=-1, keepdims=True), 1e-12))
    shift *= float(params.get('blend', 0.8))
    shift = np.where(valid[..., None], shift, 0.0)
    # Placement evidence must not change hand entry/exit decisions.
    def visible(x):
        uv = seg.project(x)
        return ((x[..., 2] > 0.01) & (uv[..., 0] >= 0) & (uv[..., 0] < seg.W) &
                (uv[..., 1] >= 0) & (uv[..., 1] < seg.H)).any(-1)
    if params.get('preserve_visibility', True):
        before = visible(j)
        keep = before == visible(j + shift[:, :, None])
        if params.get('visibility_mode', 'hard') == 'soft':
            alpha = _visibility_limit(j, shift, seg, float(params.get('visibility_sigma', 1.0)))
            shift *= alpha[..., None]
        else:
            shift = np.where(keep[..., None], shift, 0.0)
    if params.get('preserve_visibility', True):
        shift = np.where((visible(j) == visible(j + shift[:, :, None]))[..., None], shift, 0.0)
    if params.get('overlap_gate', False):
        # Strong image overlap makes independent hand localization ambiguous.
        uv = seg.project(j)
        lo, hi = uv.min(2), uv.max(2)
        extent = np.maximum(0.0, np.minimum(hi[:, 0], hi[:, 1]) - np.maximum(lo[:, 0], lo[:, 1]))
        intersection = extent.prod(-1)
        area = np.maximum(hi-lo, 0.0).prod(-1)
        overlap = intersection / np.maximum(area.sum(-1)-intersection, 1e-6)
        overlap = np.nan_to_num(overlap, nan=0.0, posinf=1.0, neginf=0.0)
        both = (state['exists'] > 0.5).all(-1)
        confidence = np.where(both, np.clip((0.75-overlap)/0.5, 0.0, 1.0), 1.0)
        confidence = gaussian_filter1d(confidence, float(params.get('overlap_sigma', 2.0)), mode='nearest')
        shift *= confidence[:, None, None]
    state['transl'] = t + shift
    state['joints_cam'] = j + shift[:, :, None]
    if 'verts_cam' in state:
        state['verts_cam'] = state['verts_cam'] + shift[:, :, None]
    return state
