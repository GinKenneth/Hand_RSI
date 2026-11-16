"""Within-segment denoising of camera translation and MANO rotations.

A symmetric Gaussian suppresses frame-to-frame prediction noise without causal
lag. Rotation matrices are averaged and projected to SO(3), so the smoothed
parameters remain valid MANO poses. No information crosses a segment boundary.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch

# Replays already parallelize over segments. Keep each MANO forward bounded.
torch.set_num_threads(1)


def _rotation_filter(R, sigma):
    if sigma <= 0:
        return R
    A = gaussian_filter1d(R, sigma, axis=0, mode="nearest", truncate=3.0)
    u, _, vt = np.linalg.svd(A)
    u[..., :, -1] *= np.linalg.det(u @ vt)[..., None]
    return u @ vt


def apply(state, seg, params):
    sigma_rot = float(params.get("sigma_rot", 0.7))
    sigma_pose = float(params.get("sigma_pose", 0.7))
    sigma_trans = float(params.get("sigma_trans", 0.7))
    state["go"] = _rotation_filter(state["go"], sigma_rot)
    state["hp"] = _rotation_filter(state["hp"], sigma_pose)
    if sigma_trans > 0:
        state["transl"] = gaussian_filter1d(state["transl"], sigma_trans, axis=0,
                                           mode="nearest", truncate=3.0)
    return seg.mano(state)
