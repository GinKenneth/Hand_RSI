"""Recover the raw network articulation for hands fully outside the image.

Blend rotation residuals across the view boundary and preserve the incoming
wrist and hand size. Original NumPy/SciPy implementation (BSD-3-Clause).
"""
import numpy as np
from scipy.spatial.transform import Rotation


def apply(state, seg, params):
    raw = np.asarray(seg.raw.get('hand_pose', []), np.float64)
    if raw.size != state['hp'].size:
        return state
    raw = raw.reshape(state['hp'].shape)
    projected = seg.project(state['joints_cam'])
    inside = ((state['joints_cam'][..., 2] > .01)
              & (projected[..., 0] >= 0) & (projected[..., 0] < seg.W)
              & (projected[..., 1] >= 0) & (projected[..., 1] < seg.H))
    outside = ~inside.any(-1)
    n = len(outside)
    width = max(float(params.get('fade_frames', 4.)), 1.)
    weight = np.zeros((n, 2), np.float64)
    for side in (0, 1):
        visible = np.flatnonzero(~outside[:, side])
        if not len(visible):
            weight[:, side] = 1.
            continue
        distance = np.abs(np.arange(n)[:, None] - visible).min(1)
        weight[:, side] = np.where(outside[:, side], np.clip(distance / width, 0., 1.), 0.)
    finite = np.isfinite(raw).all((2, 3, 4)) & np.isfinite(state['hp']).all((2, 3, 4))
    weight *= finite
    if not np.any(weight > 0):
        return state
    old_joints = state['joints_cam'].copy()
    old_verts = state['verts_cam'].copy()
    old_hp = state['hp'].copy()
    raw_delta = raw @ old_hp.swapaxes(-1, -2)
    rotvec = Rotation.from_matrix(raw_delta.reshape(-1, 3, 3)).as_rotvec().reshape(n, 2, 15, 3)
    correction = Rotation.from_rotvec((weight[..., None, None] * rotvec).reshape(-1, 3)).as_matrix().reshape(n, 2, 15, 3, 3)
    state['hp'] = correction @ old_hp
    recomputed = seg.mano(dict(state))
    new_joints = recomputed['joints_cam']
    new_verts = recomputed['verts_cam']
    old_relative = old_joints - old_joints[:, :, :1]
    new_relative = new_joints - new_joints[:, :, :1]
    scale = np.sum(old_relative * new_relative, axis=(2, 3)) / np.maximum(np.sum(new_relative ** 2, axis=(2, 3)), 1e-12)
    scale = np.clip(np.nan_to_num(scale, nan=1.), .8, 1.2)
    wrist = old_joints[:, :, :1]
    candidate_joints = wrist + scale[..., None, None] * new_relative
    candidate_verts = wrist + scale[..., None, None] * (new_verts - new_joints[:, :, :1])
    candidate_uv = seg.project(candidate_joints)
    candidate_inside = ((candidate_joints[..., 2] > .01)
                        & (candidate_uv[..., 0] >= 0) & (candidate_uv[..., 0] < seg.W)
                        & (candidate_uv[..., 1] >= 0) & (candidate_uv[..., 1] < seg.H))
    changed = (weight > 0) & ~candidate_inside.any(-1)
    state['joints_cam'][changed] = candidate_joints[changed]
    state['verts_cam'][changed] = candidate_verts[changed]
    state['hp'][~changed] = old_hp[~changed]
    return state
