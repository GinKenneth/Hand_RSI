"""Fit one small hand-shape correction from the segment's image anchors.

Camera translation and rotation are nuisance variables in the normal equations:
Schur elimination prevents their errors from becoming apparent hand shape.
A Gaussian shape prior and a half-standard-deviation bound keep the fit small.
"""
import numpy as np


def _skew(v):
    out = np.zeros(v.shape + (3,))
    out[..., 0, 1], out[..., 0, 2] = -v[..., 2], v[..., 1]
    out[..., 1, 0], out[..., 1, 2] = v[..., 2], -v[..., 0]
    out[..., 2, 0], out[..., 2, 1] = -v[..., 1], v[..., 0]
    return out


def apply(state, seg, params):
    if 'direct_joints2d' not in seg.raw:
        return state
    import torch
    from harness.core import mano_models
    from ace_eval.mano import forward
    torch.set_num_threads(1)
    n = len(state['transl'])
    ids = np.arange(0, n, max(1, int(params.get('stride', 5))))
    joints = np.asarray(state['joints_cam'], np.float64)
    anchors = np.asarray(seg.raw['direct_joints2d'], np.float64)[:n] * [seg.W, seg.H]
    projection = seg.project(joints)
    margin = np.minimum.reduce([projection[..., 0], seg.W-projection[..., 0],
                                projection[..., 1], seg.H-projection[..., 1]]).min(-1)
    visibility = np.asarray(seg.raw.get('exists_2d', np.ones((n, 2))), np.float64)[:n].reshape(n, 2)
    confidence = np.clip(margin/20.0, 0.0, 1.0) * (state['exists'] > 0.5)
    confidence *= np.nan_to_num(np.clip(visibility, 0.0, 1.0))
    correction = np.zeros((2, state['betas'].shape[-1]))
    prior = max(float(params.get('shape_prior', 0.0003)), 1e-8)
    for side in (0, 1):
        good = ((confidence[ids, side] > 0) & np.isfinite(joints[ids, side]).all((1, 2))
                & np.isfinite(anchors[ids, side]).all((1, 2))
                & np.isfinite(state['go'][ids, side]).all((1, 2))
                & np.isfinite(state['hp'][ids, side]).all((1, 2, 3))
                & np.isfinite(state['betas'][ids, side]).all(1))
        if good.sum() < 3 or confidence[ids[good], side].sum() < 3:
            continue
        index = ids[good]
        go, hp, beta = state['go'][index, side], state['hp'][index, side], state['betas'][index, side]
        base, _ = forward(mano_models(), side, go, hp, beta)
        shape_jac = np.zeros(base.shape + (len(correction[side]),))
        for k in range(len(correction[side])):
            perturbed = beta.copy()
            perturbed[:, k] += 1.0
            changed, _ = forward(mano_models(), side, go, hp, perturbed)
            shape_jac[..., k] = changed-base
        j = joints[index, side]
        z = np.maximum(j[..., 2], 0.1)
        residual = ((anchors[index, side]-seg.K[:2, 2]) / [seg.K[0, 0], seg.K[1, 1]]
                    - j[..., :2]/z[..., None])
        pj = np.zeros(j.shape[:-1] + (2, 3))
        pj[..., 0, 0], pj[..., 1, 1] = 1/z, 1/z
        pj[..., 0, 2], pj[..., 1, 2] = -j[..., 0]/z**2, -j[..., 1]/z**2
        a = pj @ shape_jac
        b = np.concatenate([pj, pj @ -_skew(j-j[:, :1])], -1)
        weight = confidence[index, side, None] / (1+(np.linalg.norm(residual, axis=-1)/0.02)**2)
        aa = np.einsum('fjik,fjil,fj->fkl', a, a, weight)
        ab = np.einsum('fjik,fjil,fj->fkl', a, b, weight)
        bb = np.einsum('fjik,fjil,fj->fkl', b, b, weight) + np.diag([100.0]*3+[1.0]*3)
        ar = np.einsum('fjik,fji,fj->fk', a, residual, weight)
        br = np.einsum('fjik,fji,fj->fk', b, residual, weight)
        # Keep the normal-equation scale independent of missing keyframes.
        normal = (aa-ab @ np.linalg.solve(bb, ab.swapaxes(-1, -2))).sum(0)/len(ids)
        rhs = (ar-(ab @ np.linalg.solve(bb, br)[..., None])[..., 0]).sum(0)/len(ids)
        delta = np.linalg.solve(normal+prior*np.eye(len(correction[side])), rhs)
        correction[side] = delta * min(1.0, float(params.get('max_shape', 0.5))/max(np.linalg.norm(delta), 1e-12))
    if not np.any(correction):
        return state
    candidate = dict(state)
    candidate['betas'] = state['betas'] + correction[None]
    candidate = seg.mano(candidate)

    def present(j, q):
        return ((j[..., 2] > 0.01) & (q[..., 0] >= 0) & (q[..., 0] < seg.W)
                & (q[..., 1] >= 0) & (q[..., 1] < seg.H)).any(-1)

    before = present(joints, projection)
    after = present(candidate['joints_cam'], seg.project(candidate['joints_cam']))
    # Preserve one constant shape per hand, including near image crossings.
    for side in (0, 1):
        if np.any(before[:, side] != after[:, side]):
            for key in ('betas', 'joints_cam', 'verts_cam'):
                candidate[key][:, side] = state[key][:, side]
    return candidate
