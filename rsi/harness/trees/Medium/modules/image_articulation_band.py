"""Complementary image articulation on well-supported interior hands.

Reuse the existing WiLoR finger rotations. Retain only the middle temporal
band of the disagreement with ACE, rejecting large angular disagreements.
The full MANO mean is included before composing rotations. Root, shape and
global orientation stay fixed; normal MANO recomputation supplies geometry.

Original NumPy/SciPy implementation (BSD-3-Clause), complementary filtering
idea as in Mahony et al., IEEE TAC 2008. WiLoR: arXiv:2409.12259, supplied
model CC BY-NC-ND 4.0. No additional image inference or weights.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation


def apply(state, seg, params):
    image = state.get('wilor_hp')
    good = state.get('wilor_good')
    if image is None or good is None:
        return state
    from harness.core import mano_models
    old = {k: state[k].copy() for k in ('transl', 'joints_cam', 'verts_cam')}
    base_geometry = seg.mano(dict(state))
    relative = state['joints_cam'] - state['joints_cam'][:, :, :1]
    base_relative = base_geometry['joints_cam'] - base_geometry['joints_cam'][:, :, :1]
    mesh_scale = np.sum(relative * base_relative, axis=(2, 3)) / np.maximum(np.sum(base_relative ** 2, axis=(2, 3)), 1e-12)
    mesh_scale = np.clip(np.nan_to_num(mesh_scale, nan=1.0), .5, 2.0)
    hp = state['hp']
    finite = np.isfinite(hp).all((2, 3, 4))
    J = state['joints_cam']
    projected = seg.project(J)
    margin = np.minimum.reduce([projected[..., 0], seg.W - 1 - projected[..., 0],
                                projected[..., 1], seg.H - 1 - projected[..., 1]])
    live = ((state['exists'] > .5) & finite & np.isfinite(J).all((2, 3))
            & (J[..., 2] > .05).all(-1))
    if params.get('interior', True):
        live &= (margin[..., 0] > 5) & ((margin > 5).sum(-1) >= 10)
    available = good & live & np.isfinite(image).all((2, 3))
    if not available.any():
        return state
    means = np.stack([mano_models()[s].hand_mean.detach().cpu().numpy().reshape(15, 3)
                      for s in (0, 1)])
    safe_hp = np.where(finite[..., None, None, None], hp, np.eye(3))
    full_vectors = Rotation.from_matrix(safe_hp.reshape(-1, 3, 3)).as_rotvec().reshape(*hp.shape[:3], 3)
    full = Rotation.from_rotvec((full_vectors + means[None]).reshape(-1, 3)).as_matrix().reshape(hp.shape)
    image_vectors = np.where(available[..., None, None], image, 0.0)
    image_rot = Rotation.from_rotvec(image_vectors.reshape(-1, 3)).as_matrix().reshape(hp.shape)
    discrepancy = Rotation.from_matrix(np.einsum('...ji,...jk->...ik', full, image_rot).reshape(-1, 3, 3))
    discrepancy = discrepancy.as_rotvec().reshape(*hp.shape[:3], 3)
    tau = np.radians(float(params.get('agreement_deg', 25)))
    weight = available[..., None] * np.exp(-.5 * (np.linalg.norm(discrepancy, axis=-1) / tau) ** 2)

    def smooth(sigma):
        den = gaussian_filter1d(weight, sigma, axis=0, mode='constant')
        num = gaussian_filter1d(discrepancy * weight[..., None], sigma, axis=0, mode='constant')
        return num / np.maximum(den, 1e-6)[..., None], den

    near, confidence = smooth(float(params.get('sigma_near', 2)))
    far, _ = smooth(float(params.get('sigma_far', 16)))
    correction = float(params.get('weight', .25)) * (near - far) * np.minimum(confidence, 1)[..., None]
    cap = np.radians(float(params.get('cap_deg', 5)))
    correction *= np.minimum(1, cap / np.maximum(np.linalg.norm(correction, axis=-1, keepdims=True), 1e-9))
    updated = np.einsum('...ij,...jk->...ik', full,
                        Rotation.from_rotvec(correction.reshape(-1, 3)).as_matrix().reshape(hp.shape))
    vectors = Rotation.from_matrix(updated.reshape(-1, 3, 3)).as_rotvec().reshape(*hp.shape[:3], 3) - means[None]
    result = Rotation.from_rotvec(vectors.reshape(-1, 3)).as_matrix().reshape(hp.shape)
    state['hp'] = np.where(live[..., None, None, None], result, hp).astype(hp.dtype)
    updated = np.any(state['hp'] != hp, axis=(2, 3, 4))
    state = seg.mano(state)
    center = state['joints_cam'][:, :, :1].copy()
    for key in ('joints_cam', 'verts_cam'):
        state[key] = center + mesh_scale[..., None, None] * (state[key] - center)
    shift = old['joints_cam'][:, :, 0] - state['joints_cam'][:, :, 0]
    state['transl'] += shift
    state['joints_cam'] += shift[:, :, None]
    state['verts_cam'] += shift[:, :, None]
    for key, value in old.items():
        state[key][~updated] = value[~updated]
    return state
