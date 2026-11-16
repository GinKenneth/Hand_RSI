"""Refine rigid hand placement with image anchors and a smooth root prior.

Joint rotation/translation fitting keeps orientation error from being absorbed
into monocular depth. Only a slow correction is retained. The raw root head
supplies a second, conservative low-frequency placement estimate, and an
acceleration prior removes residual root noise. All evidence is segment-local.
"""
import numpy as np
from scipy.linalg import solve
from scipy.ndimage import gaussian_filter1d, minimum_filter1d
from scipy.spatial.transform import Rotation


def _skew(v):
    out = np.zeros(v.shape + (3,))
    out[..., 0, 1], out[..., 0, 2] = -v[..., 2], v[..., 1]
    out[..., 1, 0], out[..., 1, 2] = v[..., 2], -v[..., 0]
    out[..., 2, 0], out[..., 2, 1] = -v[..., 1], v[..., 0]
    return out


def _limit(x, radius):
    return x * np.minimum(1.0, radius / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12))


def _fit(joints, anchors, K, valid, translation_prior, rotation_prior):
    center = joints[:, :, :1]
    relative = joints - center
    target = (anchors - K[:2, 2]) / np.array([K[0, 0], K[1, 1]])
    pars = np.zeros(joints.shape[:2] + (6,))
    prior = np.diag([translation_prior] * 3 + [rotation_prior] * 3)
    for _ in range(4):
        rot = Rotation.from_rotvec(pars[..., 3:].reshape(-1, 3)).as_matrix().reshape(joints.shape[:2] + (3, 3))
        rotated = np.einsum('fsik,fsjk->fsji', rot, relative)
        p = rotated + center + pars[..., :3][:, :, None]
        z = np.maximum(p[..., 2], 0.1)
        residual = target - p[..., :2] / z[..., None]
        weight = valid[..., None] / (1.0 + (np.linalg.norm(residual, axis=-1) / 0.02) ** 2)
        jac = np.zeros(p.shape[:-1] + (2, 3))
        jac[..., 0, 0], jac[..., 1, 1] = 1.0 / z, 1.0 / z
        jac[..., 0, 2], jac[..., 1, 2] = -p[..., 0] / z ** 2, -p[..., 1] / z ** 2
        jac = np.concatenate([jac, jac @ -_skew(rotated)], axis=-1)
        a = np.einsum('...joi,...jok,...j->...ik', jac, jac, weight) + prior
        b = np.einsum('...joi,...jo,...j->...i', jac, residual, weight) - pars @ prior
        pars += np.linalg.solve(a, b[..., None])[..., 0]
    return pars


def apply(state, seg, params):
    joints = np.asarray(state['joints_cam'], np.float64)
    anchors = np.asarray(state['joints_2d'], np.float64)
    n = len(joints)
    if n < 4:
        return state
    valid = (np.isfinite(joints).all((2, 3)) & np.isfinite(anchors).all((2, 3))
             & np.isfinite(state['transl']).all(-1) & np.isfinite(state['go']).all((2, 3)))
    safe_j = np.where(valid[..., None, None], joints, 0.0)
    safe_uv = np.where(valid[..., None, None], anchors, 0.0)
    projection = seg.project(safe_j)
    margin = np.minimum.reduce([projection[..., 0], seg.W - projection[..., 0],
                                projection[..., 1], seg.H - projection[..., 1]]).min(-1)
    visibility = np.asarray(seg.raw.get('exists_2d', np.ones((n, 2))), np.float64)[:n].reshape(n, 2)
    confidence = np.clip(margin / 20.0, 0.0, 1.0) * (state['exists'] > 0.5)
    confidence *= np.nan_to_num(np.clip(visibility, 0.0, 1.0)) * valid
    confidence = minimum_filter1d(confidence, size=9, axis=0, mode='nearest')
    gate = gaussian_filter1d(confidence, 3.0, axis=0, mode='nearest') * valid
    if not np.any(gate > 1e-6):
        return state

    correction = _fit(safe_j, safe_uv, np.asarray(seg.K), valid,
                      float(params.get('translation_prior', 100.0)),
                      float(params.get('rotation_prior', 1.0)))
    sigma = float(params.get('correction_sigma', 5.0))
    correction = (gaussian_filter1d(correction * confidence[..., None], sigma, axis=0, mode='nearest') /
                  np.maximum(gaussian_filter1d(confidence, sigma, axis=0, mode='nearest')[..., None], 1e-6))
    shift = _limit(correction[..., :3], 0.01) * gate[..., None]
    angle = _limit(correction[..., 3:], 0.08) * gate[..., None]

    raw_weight = float(params.get('raw_weight', 0.5))
    if raw_weight > 0 and 'cam_trans' in seg.raw:
        residual = np.asarray(seg.raw['cam_trans'], np.float64)[:n].reshape(n, 2, 3) - state['transl']
        residual = np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)
        t = np.linspace(-1.0, 1.0, n)
        basis = np.stack([np.ones(n), t, t * t], axis=-1)
        slow = np.zeros_like(residual)
        for side in (0, 1):
            weighted = basis.T * confidence[:, side]
            slow[:, side] = basis @ np.linalg.solve(weighted @ basis + 1e-6 * np.eye(3), weighted @ residual[:, side])
        shift += _limit(raw_weight * slow, 0.01) * gate[..., None]

    strength = float(params.get('root_strength', 3.0))
    root = np.nan_to_num(np.asarray(state['transl'], np.float64)) + shift
    d = np.diff(np.eye(n), n=2, axis=0)
    smoothed = solve(np.eye(n) + strength * d.T @ d, root.reshape(n, -1), assume_a='pos').reshape(n, 2, 3)
    shift += (smoothed - root) * gate[..., None]
    shift = _limit(shift, float(params.get('max_shift', 0.02)))
    rot = Rotation.from_rotvec(angle.reshape(-1, 3)).as_matrix().reshape(n, 2, 3, 3)
    center = joints[:, :, :1]
    placed = np.einsum('fsik,fsjk->fsji', rot, joints - center) + center + shift[:, :, None]
    after = seg.project(placed)

    # Preserve in-frame hand presence at image entries and exits.
    def visible(j, uv):
        return ((j[..., 2] > 0.01) & (uv[..., 0] >= 0) & (uv[..., 0] < seg.W) &
                (uv[..., 1] >= 0) & (uv[..., 1] < seg.H)).any(-1)
    keep = valid & (visible(joints, projection) == visible(placed, after))
    shift = np.where(keep[..., None], shift, 0.0)
    rot = np.where(keep[..., None, None], rot, np.eye(3))
    placed = np.where(keep[..., None, None], placed, joints)
    verts = np.asarray(state['verts_cam'])
    state['verts_cam'] = np.einsum('fsik,fsjk->fsji', rot, verts - center) + center + shift[:, :, None]
    state['joints_cam'] = placed
    state['transl'] = state['transl'] + shift
    state['go'] = rot @ state['go']
    state['joints_2d'] = anchors + float(params.get('projection_weight', 0.375)) * np.where(
        keep[..., None, None], after - projection, 0.0)
    return state
