"""Resolve disagreement between in-image visibility and MANO placement.

Use 2D palm anchors for a bounded lateral translation when the MANO prediction
extends beyond the image. Weight image evidence by visibility confidence and
taper the correction as more geometry becomes visible. A short temporal
average can reduce visibility flicker. This is a fixed-depth perspective fit,
so the surface, root parameter and joints move together.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d


def apply(state, seg, params):
    if 'exists_2d' not in seg.raw:
        return state
    joints = np.asarray(state['joints_cam'], dtype=np.float64)
    uv = seg.project(joints)
    joint_visible = ((uv[..., 0] >= 0) & (uv[..., 0] < seg.W)
               & (uv[..., 1] >= 0) & (uv[..., 1] < seg.H)
               & (joints[..., 2] > 0.01))
    visible = joint_visible.any(-1)
    confidence = np.asarray(seg.raw['exists_2d'])[:len(joints)]
    sigma = float(params.get('temporal_sigma', 0))
    if sigma > 0:
        confidence = gaussian_filter1d(confidence, sigma, axis=0)
    temperature = float(params.get('visibility_temperature', 1.0))
    if temperature != 1.0:
        p = np.clip(confidence, 1e-8, 1 - 1e-8)
        a = p ** (1.0 / temperature)
        b = (1 - p) ** (1.0 / temperature)
        confidence = a / (a + b)
    soft = bool(params.get('confidence_weight', False))
    power = float(params.get('partial_power', 0))
    visibility_weight = ((1 - joint_visible.mean(-1)) ** power
                         if power > 0 else (~visible).astype(float))
    active = (visibility_weight > 0) & (confidence > (0 if soft else 0.5)) & (state['exists'] > 0.5)
    if not active.any():
        return state
    palm = [0, 5, 9, 13, 17]
    j = joints[:, :, palm]
    anchors = np.asarray(state['joints_2d'], dtype=np.float64)[:, :, palm]
    k = np.asarray(seg.K)
    target = (anchors - k[:2, 2]) / np.array([k[0, 0], k[1, 1]])
    residual = target * j[..., 2:] - j[..., :2]
    delta = np.zeros(joints.shape[:2] + (3,))
    delta[..., :2] = residual.mean(-2)
    good = np.isfinite(delta).all(-1) & (j[..., 2] > 0.01).all(-1)
    delta[~(active & good)] = 0
    limit = float(params.get('max_shift', 0.005))
    delta *= np.minimum(1, limit / np.maximum(np.linalg.norm(delta, axis=-1, keepdims=True), 1e-12))
    if soft:
        delta *= np.clip(confidence, 0, 1)[..., None]
    delta *= visibility_weight[..., None]
    state['transl'] = state['transl'] + delta
    state['joints_cam'] = state['joints_cam'] + delta[:, :, None]
    state['verts_cam'] = state['verts_cam'] + delta[:, :, None]
    return state
