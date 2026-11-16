"""Regularize the root trajectory within each contiguous hand track.

A quadratic finite-difference prior uses both sides of a frame. Third
differences preserve quadratic motion, permitting stronger denoising than an
acceleration prior. This refinement follows the shared R1_a6_req_2 result.
Only rigid placement changes: the surface, joints and root move together.
"""
import numpy as np
from scipy.linalg import solve_banded


def _smooth(x, strength, order=2):
    n = len(x)
    d = np.diff(np.eye(n), order, axis=0)
    a = np.eye(n) + strength * (d.T @ d)
    width = min(order, n - 1)
    band = np.zeros((2 * width + 1, n))
    for k in range(-width, width + 1):
        band[width - k, max(k, 0):min(n, n + k)] = np.diag(a, k)
    return solve_banded((width, width), band, x)


def apply(state, seg, params):
    strength = float(params.get('strength', 1.0))
    order = int(params.get('order', 2))
    max_shift = float(params.get('max_shift', 0.01))
    t = np.asarray(state['transl'], dtype=np.float64)
    observed = t.copy()
    direct_weight = float(params.get('direct_weight', 0.25))
    if direct_weight > 0 and 'direct_wrist_cam' in seg.raw:
        direct = np.asarray(seg.raw['direct_wrist_cam'], dtype=np.float64)[:len(t)]
        delta = direct_weight * (direct - state['joints_cam'][:, :, 0])
        delta[~np.isfinite(delta).all(-1)] = 0
        delta *= np.minimum(1.0, max_shift / np.maximum(np.linalg.norm(delta, axis=-1, keepdims=True), 1e-12))
        observed += delta
    shift = np.zeros_like(t)
    for side in range(2):
        active = (state['exists'][:, side] > 0.5) & np.isfinite(t[:, side]).all(-1)
        edges = np.diff(np.r_[False, active, False].astype(np.int8))
        for lo, hi in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if hi - lo < 3:
                continue
            delta = _smooth(observed[lo:hi, side], strength, order) - t[lo:hi, side]
            delta *= np.minimum(1.0, max_shift / np.maximum(np.linalg.norm(delta, axis=-1, keepdims=True), 1e-12))
            shift[lo:hi, side] = delta
    state['transl'] = (t + shift).astype(np.float32)
    state['joints_cam'] = state['joints_cam'] + shift[:, :, None, :]
    state['verts_cam'] = state['verts_cam'] + shift[:, :, None, :]
    return state
