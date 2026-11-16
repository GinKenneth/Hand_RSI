"""Reconcile one segment shape with raw shape and independent 3D bone lengths.

Bone distances remove the direct head's root and global rotation errors. A
strong prior retains the current shape, and only visible, finite observations
contribute. The resulting shape is shared through periods out of sight.
Original implementation using supplied MANO and PyTorch/NumPy (BSD-3-Clause).
"""
import numpy as np
from scipy.spatial.transform import Rotation

_TIPS = [744, 320, 443, 554, 671]
_ORDER = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]


def apply(state, seg, params):
    if any(k not in seg.raw for k in ('betas', 'exists_2d', 'direct_joints_cam')):
        return state
    import torch
    from harness.core import mano_models
    torch.set_num_threads(1)
    n = seg.n
    base_geometry = seg.mano(dict(state))
    relative_old = state['joints_cam'] - state['joints_cam'][:, :, :1]
    relative_base = base_geometry['joints_cam'] - base_geometry['joints_cam'][:, :, :1]
    mesh_scale = np.sum(relative_old * relative_base, axis=(2, 3)) / np.maximum(np.sum(relative_base ** 2, axis=(2, 3)), 1e-12)
    mesh_scale = np.clip(np.nan_to_num(mesh_scale, nan=1.0), 0.8, 1.2)
    previous = {k: state[k].copy() for k in ('betas', 'transl', 'joints_cam', 'verts_cam')}
    wrist = state['joints_cam'][:, :, 0].copy()
    raw_beta = np.asarray(seg.raw['betas'], np.float64)[:n]
    raw_vis = np.clip(np.asarray(seg.raw['exists_2d'], np.float64)[:n], 0., 1.)
    valid_shape = np.isfinite(raw_beta).all(-1) & np.isfinite(raw_vis)
    w = np.where(valid_shape, .01 + raw_vis ** 2, 0.)
    shape = (np.nan_to_num(raw_beta) * w[..., None]).sum(0) / np.maximum(w.sum(0)[:, None], 1e-9)
    enough = ((raw_vis > .8) & valid_shape).sum(0) >= 3
    shape = np.where(enough[:, None], shape, state['betas'].mean(0))
    blend = float(params.get('blend', .5))
    state['betas'] = state['betas'] + blend * enough[None, :, None] * (shape[None] - state['betas'])
    vis = np.clip(np.asarray(seg.raw['exists_2d'], np.float64)[:n], 0., 1.)
    direct = np.asarray(seg.raw['direct_joints_cam'], np.float64)[:n]
    pairs = [(0, j) for j in (1, 5, 9, 13, 17)]
    pairs += [(5, 9), (9, 13), (13, 17), (5, 17)]
    if not params.get('palm_only', False):
        pairs += [(k, k+1) for root in (1,5,9,13,17) for k in range(root, root+3)]
    i, j = np.array(pairs).T
    changed = False
    for side, model in mano_models().items():
        if int(params.get('steps', 12)) <= 0:
            continue
        valid = np.flatnonzero((vis[:, side] > .8) & np.isfinite(direct[:, side]).all((1,2))
                              & np.isfinite(state['hp'][:, side]).all((1,2,3)))
        if len(valid) < 3:
            continue
        count = int(params.get('samples', 12))
        valid = valid[np.unique(np.linspace(0,len(valid)-1,min(len(valid),count)).astype(int))]
        hp = torch.tensor(Rotation.from_matrix(state['hp'][valid,side].reshape(-1,3,3)).as_rotvec().reshape(-1,45), dtype=torch.float32)
        go = torch.zeros((len(valid),3))
        beta0 = torch.tensor(state['betas'][:,side].mean(0)[None], dtype=torch.float32)
        beta = beta0.clone().requires_grad_(True)
        target = torch.tensor(np.linalg.norm(direct[valid,side][:,i]-direct[valid,side][:,j],axis=-1), dtype=torch.float32)
        for parameter in model.parameters(): parameter.requires_grad_(False)
        optimizer = torch.optim.LBFGS([beta], max_iter=int(params.get('steps',15)), max_eval=20,
                                     tolerance_grad=1e-9, tolerance_change=1e-10, line_search_fn='strong_wolfe')
        def closure():
            optimizer.zero_grad()
            out = model(global_orient=go,hand_pose=hp,betas=beta.expand(len(valid),-1))
            joints = torch.cat([out.joints,out.vertices[:,_TIPS]],1)[:,_ORDER]
            lengths = torch.linalg.norm(joints[:,i]-joints[:,j],dim=-1)
            residual = (lengths-target)/.05
            loss = (torch.sqrt(1+residual.square()/.01)-1).mean()*.02
            loss = loss + float(params.get('prior',.01))*(beta-beta0).square().mean()
            loss.backward()
            return loss
        with torch.enable_grad(): optimizer.step(closure)
        result = beta.detach().numpy()[0]
        shift = result-beta0.numpy()[0]
        shift *= min(1., float(params.get('max_change',1.))/max(np.linalg.norm(shift),1e-9))
        if np.isfinite(shift).all():
            state['betas'][:,side] += float(params.get('alpha',1.))*shift
            changed = True
    total = state['betas'] - previous['betas']
    limit = float(params.get('max_total', 2.2))
    total *= np.minimum(1., limit / np.maximum(np.linalg.norm(total, axis=-1), 1e-9))[..., None]
    state['betas'] = previous['betas'] + total
    state = seg.mano(state)
    center = state['joints_cam'][:, :, :1].copy()
    state['joints_cam'] = center + mesh_scale[..., None, None] * (state['joints_cam'] - center)
    state['verts_cam'] = center + mesh_scale[..., None, None] * (state['verts_cam'] - center)
    if params.get('keep_wrist', False):
        shift = wrist - state['joints_cam'][:, :, 0]
        state['transl'] += shift
        state['joints_cam'] += shift[:, :, None]
        state['verts_cam'] += shift[:, :, None]
    if params.get('boundary_guard', False):
        # The shape fit has no image evidence for a boundary crossing. Limit
        # its step there so the previous image-supported footprint is retained.
        old = previous['joints_cam']
        new = state['joints_cam']
        def visible(joints):
            uv = seg.project(joints)
            return ((joints[..., 2] > .01) & (uv[..., 0] >= 0) & (uv[..., 0] < seg.W)
                    & (uv[..., 1] >= 0) & (uv[..., 1] < seg.H)).any(-1)
        before = visible(old)
        crossing = before != visible(new)
        lo, hi = np.zeros((n,2)), np.ones((n,2))
        for _ in range(16):
            mid = (lo + hi) / 2
            same = visible(old + mid[...,None,None] * (new-old)) == before
            lo = np.where(same, mid, lo)
            hi = np.where(same, hi, mid)
        step = np.where(crossing, lo * .95, 1.)
        for key in previous:
            weight = step.reshape(step.shape + (1,) * (state[key].ndim - 2))
            state[key] = previous[key] + weight * (state[key] - previous[key])
    return state
