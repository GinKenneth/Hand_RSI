"""Denoise each segment's MANO trajectory with a minimum-jerk prior.

The fit minimizes observation error plus squared third temporal differences.
It preserves constant-acceleration motion, uses both temporal directions, and
has natural boundaries. Rotations are fitted in their matrix embedding and
projected back to SO(3); MANO then keeps all geometry consistent.
"""
from functools import lru_cache
import numpy as np
from scipy.linalg import cho_factor, cho_solve


@lru_cache(maxsize=16)
def _kernel(n, strength):
    d = np.diff(np.eye(n), n=3, axis=0)
    a = np.eye(n) + strength * (d.T @ d)
    return cho_solve(cho_factor(a), np.eye(n))


def _smooth(x, kernel):
    flat = np.asarray(x, np.float64).reshape(len(x), -1)
    good = np.isfinite(flat).all(axis=0)
    out = flat.copy()
    out[:, good] = kernel @ flat[:, good]
    return out.reshape(x.shape)


def _rotations(r, kernel):
    q = _smooth(r, kernel).reshape(-1, 3, 3)
    good = np.isfinite(q).all(axis=(1, 2))
    if good.any():
        u, _, vh = np.linalg.svd(q[good])
        sign = np.linalg.det(u @ vh)
        u[..., :, -1] *= np.where(sign < 0, -1.0, 1.0)[..., None]
        q[good] = u @ vh
    return q.reshape(r.shape)


def apply(state, seg, params):
    n = len(state["transl"])
    if n < 4:
        return state
    kernel = _kernel(n, float(params.get("strength", 1.0)))
    previous_translation = state["transl"].copy()
    for key in ("transl", "joints_2d"):
        if params.get(key, True):
            strength = params.get(key + "_strength")
            key_kernel = kernel if strength is None else _kernel(n, float(strength))
            state[key] = _smooth(state[key], key_kernel)
    changed = False
    for key in ("go", "hp"):
        if params.get(key, True):
            state[key] = _rotations(state[key], kernel)
            changed = True
    if changed:
        import torch
        torch.set_num_threads(1)
        state = seg.mano(state)
    else:
        shift = state["transl"] - previous_translation
        state["joints_cam"] += shift[:, :, None]
        state["verts_cam"] += shift[:, :, None]
    wrist_weight = float(params.get("direct_wrist_weight", 0.0))
    if wrist_weight > 0:
        residual = seg.raw["direct_wrist_cam"][:n] - state["joints_cam"][:, :, 0]
        wrist_kernel = _kernel(n, float(params.get("wrist_strength", 10.0)))
        shift = wrist_weight * _smooth(residual, wrist_kernel)
        state["transl"] += shift
        state["joints_cam"] += shift[:, :, None]
        state["verts_cam"] += shift[:, :, None]
    return state

