"""Minimum-jerk regularization of rotations and direct image anchors.

Uses the temporal-consistency method in the accepted shared result
R1_a6_req_1. Rotations are averaged in their matrix embedding, projected onto
SO(3), and passed through MANO to preserve the surface/parameter convention.
"""
from functools import lru_cache
import numpy as np
from scipy.linalg import cho_factor, cho_solve


@lru_cache(maxsize=16)
def _kernel(n, strength):
    d = np.diff(np.eye(n), n=3, axis=0)
    return cho_solve(cho_factor(np.eye(n) + strength * d.T @ d), np.eye(n))


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
        u[..., :, -1] *= np.where(np.linalg.det(u @ vh) < 0, -1., 1.)[..., None]
        q[good] = u @ vh
    return q.reshape(r.shape)


def apply(state, seg, params):
    n = len(state['transl'])
    if n < 4:
        return state
    kernel = _kernel(n, float(params.get('strength', 1.0)))
    state['joints_2d'] = _smooth(state['joints_2d'], kernel)
    for key in ('go', 'hp'):
        state[key] = _rotations(state[key], kernel)
    import torch
    torch.set_num_threads(1)
    return seg.mano(state)
