"""Refine a smooth root trajectory using the hand's image-space anchors.

A robust perspective fit estimates a conservative translation residual while
holding articulation fixed. An angular residual weight limits inconsistent
joints; out-of-frame anchors receive less weight. A root prior and temporal
minimum-jerk regularizer retain the network's metric depth and trajectory.
"""
from functools import lru_cache
import numpy as np
from scipy.linalg import cho_factor, cho_solve


@lru_cache(maxsize=8)
def _kernel(n, strength):
    d = np.diff(np.eye(n), n=3, axis=0)
    return cho_solve(cho_factor(np.eye(n) + strength * d.T @ d), np.eye(n))


def apply(state, seg, params):
    j = np.asarray(state['joints_cam'], np.float64)
    uv = np.asarray(state['joints_2d'], np.float64)
    n = len(j)
    if n < 4:
        return state
    valid = np.isfinite(j).all((2, 3)) & np.isfinite(uv).all((2, 3))
    # Invalid observations contribute zero correction rather than contaminating
    # the trajectory fit.
    j = np.where(valid[:, :, None, None], j, 0.0)
    uv = np.where(valid[:, :, None, None], uv, 0.0)
    focal = np.array([seg.K[0, 0], seg.K[1, 1]])
    target = (uv - seg.K[:2, 2]) / focal
    inside = ((uv[..., 0] >= 0) & (uv[..., 0] < seg.W) &
              (uv[..., 1] >= 0) & (uv[..., 1] < seg.H))
    ridge = float(params.get('ridge', 10.0))
    shift = np.zeros(j.shape[:2] + (3,))
    for _ in range(3):
        p = j + shift[:, :, None]
        z = np.maximum(p[..., 2], 0.1)
        residual = target - p[..., :2] / z[..., None]
        weight = 1 / (1 + (np.linalg.norm(residual, axis=-1) / 0.05) ** 2)
        weight *= np.where(inside, 1.0, 0.1) * valid[:, :, None]
        jac = np.zeros(j.shape[:3] + (2, 3))
        jac[..., 0, 0] = 1 / z
        jac[..., 1, 1] = 1 / z
        jac[..., 0, 2] = -p[..., 0] / z ** 2
        jac[..., 1, 2] = -p[..., 1] / z ** 2
        a = np.einsum('...joi,...jok,...j->...ik', jac, jac, weight) + ridge * np.eye(3)
        b = np.einsum('...joi,...jo,...j->...i', jac, residual, weight) - ridge * shift
        shift += np.linalg.solve(a, b[..., None])[..., 0]
    weight = np.full((n, 2), float(params.get('weight', 0.1)))
    confidence_weight = float(params.get('confidence_weight', 0.0))
    if confidence_weight > 0 and 'exists_2d' in seg.raw:
        confidence = np.asarray(seg.raw['exists_2d'][:n], np.float64).reshape(n, 2)
        confidence = np.nan_to_num(np.clip(confidence, 0.0, 1.0))
        weight += confidence_weight * confidence ** 2
    shift *= weight[:, :, None]
    shift = (_kernel(n, float(params.get('strength', 10.0))) @
             shift.reshape(n, -1)).reshape(n, 2, 3)
    state['transl'] = state['transl'] + shift
    state['joints_cam'] = state['joints_cam'] + shift[:, :, None]
    state['verts_cam'] = state['verts_cam'] + shift[:, :, None]
    return state
