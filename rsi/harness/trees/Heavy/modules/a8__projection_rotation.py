"""Small orientation consensus from independent image-space keypoints.

Fit camera-space rotation about the wrist, allowing a nuisance image offset so
root translation errors cannot tilt the hand. The correction is bounded and
regularized. When image evidence vanishes, briefly continue a correction in
the hand's local frame, preserving the network's predicted motion.

Original NumPy/SciPy implementation (BSD-3-Clause).
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation


def apply(state, seg, params):
    if 'exists_2d' not in seg.raw:
        return state
    n = seg.n
    j = np.asarray(state['joints_cam'], np.float64)
    target = np.asarray(state['joints_2d'], np.float64)
    v = np.clip(np.asarray(seg.raw['exists_2d'], np.float64)[:n], 0., 1.)
    if params.get('raw_2d', False):
        target = np.asarray(seg.raw['direct_joints2d'], np.float64)[:n] * [seg.W, seg.H]
    finite_go = np.isfinite(state['go']).all((2, 3))
    go = np.where(finite_go[..., None, None], state['go'], np.eye(3))
    finite = np.isfinite(j).all((2, 3)) & np.isfinite(target).all((2, 3)) & finite_go
    good = finite & (j[..., 2] > .03).all(-1) & (v > float(params.get('visibility', .8)))
    if not good.any():
        return state
    ids = np.asarray(params.get('joints', list(range(21))), int)
    delta = np.zeros((n, 2, 3), np.float64)
    rel = j - j[:, :, :1]
    maximum = np.deg2rad(float(params.get('max_degrees', 10.)))
    for _ in range(int(params.get('steps', 3))):
        rot = Rotation.from_rotvec(delta.reshape(-1, 3)).as_matrix().reshape(n, 2, 3, 3)
        x = rel @ rot.transpose(0, 1, 3, 2)
        xyz = x + j[:, :, :1]
        z = np.maximum(xyz[..., 2], .03)
        pred = seg.project(xyz)
        # d projection / d left camera rotation, rotation is around the wrist.
        px, py, pz = x[..., 0], x[..., 1], x[..., 2]
        jac3 = np.stack([np.zeros_like(px), pz, -py, -pz, np.zeros_like(px), px,
                         py, -px, np.zeros_like(px)], -1).reshape(n, 2, 21, 3, 3)
        proj = np.zeros((n, 2, 21, 2, 3))
        proj[..., 0, 0] = seg.K[0, 0] / z
        proj[..., 1, 1] = seg.K[1, 1] / z
        proj[..., 0, 2] = -seg.K[0, 0] * xyz[..., 0] / z ** 2
        proj[..., 1, 2] = -seg.K[1, 1] * xyz[..., 1] / z ** 2
        jac = proj @ jac3
        residual = target - pred
        # Eliminate the nuisance common image shift analytically.
        jac -= jac[:, :, ids].mean(2, keepdims=True)
        residual -= residual[:, :, ids].mean(2, keepdims=True)
        scale = np.maximum(np.linalg.norm(pred - pred.mean(2, keepdims=True), axis=-1).mean(2), 5.)
        jac /= scale[..., None, None, None]
        residual /= scale[..., None, None]
        a = jac[:, :, ids].reshape(n, 2, -1, 3)
        b = residual[:, :, ids].reshape(n, 2, -1)
        if params.get('free_scale', True):
            scaling = (pred - pred[:, :, ids].mean(2, keepdims=True))[:, :, ids].reshape(n, 2, -1)
            scaling /= np.maximum(np.linalg.norm(scaling, axis=-1), 1e-9)[..., None]
            a -= scaling[..., None] * (a * scaling[..., None]).sum(-2)[..., None, :]
            b -= scaling * (b * scaling).sum(-1)[..., None]
        prior = float(params.get('prior', 5.))
        mat = a.swapaxes(-1, -2) @ a + prior * np.eye(3)
        rhs = (a.swapaxes(-1, -2) @ b[..., None])[..., 0] - prior * delta
        update = np.zeros_like(delta)
        update[good] = np.linalg.solve(mat[good], rhs[good])
        delta += update
        delta *= np.minimum(1., maximum / np.maximum(np.linalg.norm(delta, axis=-1), 1e-9))[..., None]
    alpha = float(params.get('alpha', 1.))
    local = (go.transpose(0, 1, 3, 2) @ delta[..., None])[..., 0]
    tau = float(params.get('carry_frames', 0.))
    for side in (0, 1):
        idx = np.flatnonzero(good[:, side])
        if len(idx):
            sigma = float(params.get('sigma', 1.))
            if sigma > 0:
                w = good[:, side].astype(float)
                local[:, side] = gaussian_filter1d(local[:, side] * w[:, None], sigma, axis=0) / np.maximum(gaussian_filter1d(w, sigma), 1e-9)[:, None]
            low_sigma = float(params.get('low_sigma', 0.))
            if low_sigma > 0:
                w = good[:, side].astype(float)
                bias = gaussian_filter1d(local[:, side] * w[:, None], low_sigma, axis=0) / np.maximum(gaussian_filter1d(w, low_sigma), 1e-9)[:, None]
                local[:, side] -= bias
            if tau > 0:
                distance = np.abs(np.arange(n)[:, None] - idx)
                nearest = idx[distance.argmin(1)]
                fill = local[nearest, side] * np.exp(-distance.min(1) / tau)[:, None]
                local[:, side] = np.where(good[:, side, None], local[:, side], fill)
            else:
                local[~good[:, side], side] = 0.
    delta = (go @ local[..., None])[..., 0] * alpha
    delta = np.where(finite[..., None], delta, 0.)
    delta *= np.minimum(1., maximum / np.maximum(np.linalg.norm(delta, axis=-1), 1e-9))[..., None]
    if float(params.get('boundary_fade', 0.)) > 0:
        uv = seg.project(j)
        border = np.minimum.reduce([uv[..., 0], seg.W - 1 - uv[..., 0], uv[..., 1], seg.H - 1 - uv[..., 1]])
        distance = np.max(border, -1)
        extent = np.linalg.norm(uv - uv.mean(2, keepdims=True), axis=-1).mean(2)
        fade = np.clip(np.abs(distance) / np.maximum(extent * float(params['boundary_fade']), 1.), 0., 1.)
        delta *= fade[..., None]
    delta = np.where(finite[..., None], delta, 0.)
    correction_limit = np.deg2rad(float(params.get('max_correction_degrees', 3.)))
    delta *= np.minimum(1., correction_limit / np.maximum(np.linalg.norm(delta, axis=-1), 1e-9))[..., None]
    rot = Rotation.from_rotvec(delta.reshape(-1, 3)).as_matrix().reshape(n, 2, 3, 3)
    wrist = j[:, :, :1]
    previous = {k: state[k].copy() for k in ('go', 'transl', 'joints_cam', 'verts_cam')}
    state['go'] = rot @ state['go']
    state['transl'] = ((state['transl'] - wrist[:, :, 0])[..., None, :] @
                       rot.transpose(0, 1, 3, 2))[..., 0, :] + wrist[:, :, 0]
    state['joints_cam'] = (j - wrist) @ rot.transpose(0, 1, 3, 2) + wrist
    state['verts_cam'] = (state['verts_cam'] - wrist) @ rot.transpose(0, 1, 3, 2) + wrist
    if params.get('boundary_guard', False):
        def visible(x):
            uv = seg.project(x)
            return ((x[...,2] > .01) & (uv[...,0] >= 0) & (uv[...,0] < seg.W) & (uv[...,1] >= 0) & (uv[...,1] < seg.H)).any(-1)
        crossing = visible(j) != visible(state['joints_cam'])
        for key in previous: state[key][crossing] = previous[key][crossing]
    return state
