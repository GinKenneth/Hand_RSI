"""Fit a small, temporally smooth MANO articulation correction to a second head.

The direct 3D head provides independent wrist-relative joint evidence. A strong
angular prior preserves the MANO head's articulation, and a minimum-jerk fit
keeps the correction smooth within this segment. Global rotation, shape and
root translation are held fixed; MANO recomputes the final mesh and joints.
"""
from functools import lru_cache
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.transform import Rotation
import torch

_TIPS = [744, 320, 443, 554, 671]
_ORDER = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]


@lru_cache(maxsize=8)
def _kernel(n, strength):
    d = np.diff(np.eye(n), n=3, axis=0)
    return cho_solve(cho_factor(np.eye(n) + strength * d.T @ d), np.eye(n))


def _smooth(x, strength):
    flat = np.asarray(x, np.float64).reshape(len(x), -1)
    good = np.isfinite(flat).all(0)
    out = flat.copy()
    out[:, good] = _kernel(len(x), strength) @ flat[:, good]
    return out.reshape(x.shape)


def apply(state, seg, params):
    n = len(state['hp'])
    if n < 4 or 'direct_joints_cam' not in seg.raw:
        return state
    torch.set_num_threads(1)
    from harness.core import mano_models
    models = mano_models()
    direct = _smooth(seg.raw['direct_joints_cam'][:n], float(params.get('target_strength', 1.0)))
    result = state['hp'].copy()
    for side in (0, 1):
        valid = (np.isfinite(state['hp'][:, side]).all((1, 2, 3)) &
                 np.isfinite(state['go'][:, side]).all((1, 2)) &
                 np.isfinite(state['betas'][:, side]).all(1) &
                 np.isfinite(direct[:, side]).all((1, 2)))
        if not valid.any():
            continue
        h0_np = Rotation.from_matrix(state['hp'][valid, side].reshape(-1, 3, 3)).as_rotvec().reshape(-1, 45)
        h0 = torch.tensor(h0_np, dtype=torch.float32)
        go = torch.tensor(Rotation.from_matrix(state['go'][valid, side]).as_rotvec(), dtype=torch.float32)
        shape = torch.tensor(state['betas'][valid, side], dtype=torch.float32)
        target = direct[valid, side] - direct[valid, side, :1]
        target = torch.tensor(target, dtype=torch.float32)
        model = models[side]
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        h = h0.clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [h], lr=1, max_iter=min(20, int(params.get('steps', 20))),
            max_eval=25, history_size=10, tolerance_grad=1e-10,
            tolerance_change=1e-12, line_search_fn='strong_wolfe')
        prior = float(params.get('prior', 0.03))

        def closure():
            optimizer.zero_grad()
            out = model(global_orient=go, hand_pose=h, betas=shape,
                        transl=torch.zeros(len(h), 3))
            joints = torch.cat([out.joints, out.vertices[:, _TIPS]], 1)[:, _ORDER]
            joints = joints - joints[:, :1]
            loss = ((joints - target) ** 2).sum((-1, -2)).mean()
            loss = loss + prior * ((h - h0) ** 2).sum(-1).mean()
            loss.backward()
            return loss

        with torch.enable_grad():
            optimizer.step(closure)
        correction = np.zeros((n, 15, 3))
        delta = (h.detach() - h0).numpy().reshape(-1, 15, 3)
        if not np.isfinite(delta).all():
            continue
        correction[valid] = delta
        correction = _smooth(correction, float(params.get('correction_strength', 10.0)))
        aa = h0_np.reshape(-1, 15, 3) + correction[valid]
        result[valid, side] = Rotation.from_rotvec(aa.reshape(-1, 3)).as_matrix().reshape(-1, 15, 3, 3)
    state['hp'] = result
    state = seg.mano(state)
    # Retain the image evidence for the hand's common position, and use the
    # fitted kinematics to regularize the relative joint configuration.
    uv = np.asarray(state['joints_2d'], np.float64)
    projection = seg.project(state['joints_cam'])
    relative = projection - projection.mean(axis=2, keepdims=True)
    relative -= uv - uv.mean(axis=2, keepdims=True)
    good = (np.isfinite(relative).all((2, 3)) &
            (state['joints_cam'][..., 2] > 0.05).all(2) &
            (np.median(np.linalg.norm(relative, axis=-1), axis=2) < 30.0))
    relative = np.where(good[:, :, None, None], relative, 0.0)
    correction = _smooth(relative, float(params.get('anchor_strength', 1.0)))
    state['joints_2d'] = uv + float(params.get('anchor_weight', 0.5)) * correction
    return state
