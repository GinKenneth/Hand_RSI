"""Robust perspective finger fit with reliability-weighted image evidence.

Fit only well-supported interior hands. Subtract the robust common image
offset so articulation does not explain placement error. Keep shape, root
and global orientation fixed, with a pose prior and smooth bounded update.

The inherited uniform anchor fit is replaced by a Cauchy data loss and a
reliability-weighted temporal band of the already computed WiLoR crop-camera
keypoints. Image confidence uses centered image/kinematic disagreement,
truncation and a temporal filter. A depth prior retains the video model's
unobserved depth while image evidence corrects articulation. Only this
segment is used; no additional image inference is performed.

The optimization evaluates the 16 kinematic joints and five fingertip
vertices; normal seg.mano supplies the final mesh. Original implementation
using NumPy/SciPy/PyTorch (BSD-3-Clause), calling supplied smplx.lbs functions
(SMPL-X non-commercial scientific research license). MANO: Romero et al.,
ACM TOG 2017. No model data or additional weights are included.
"""
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

TIPS = [744, 320, 443, 554, 671]
ORDER = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
BODY = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]


def _prepare_joints(model, shape, translation, scale, wrist):
    from smplx.lbs import batch_rodrigues, batch_rigid_transform

    with torch.no_grad():
        shaped = model.v_template + torch.einsum('bl,vkl->bvk', shape, model.shapedirs)
        rest = torch.einsum('jv,bvk->bjk', model.J_regressor, shaped)
        tip_shaped = shaped[:, TIPS]
        tip_dirs = model.posedirs.reshape(135, -1, 3)[:, TIPS].reshape(135, 15)
        tip_weights = model.lbs_weights[TIPS]
        identity = torch.eye(3, dtype=shape.dtype)

    def forward(go, hp):
        pose = torch.cat([go, hp], dim=1) + model.pose_mean
        rotations = batch_rodrigues(pose.reshape(-1, 3)).reshape(len(shape), 16, 3, 3)
        joints, transforms = batch_rigid_transform(rotations, rest, model.parents, dtype=shape.dtype)
        feature = (rotations[:, 1:] - identity).reshape(len(shape), 135)
        tips = tip_shaped + (feature @ tip_dirs).reshape(len(shape), 5, 3)
        blend = (tip_weights @ transforms.reshape(len(shape), 16, 16)).reshape(len(shape), 5, 4, 4)
        homogeneous = torch.cat([tips, torch.ones_like(tips[..., :1])], dim=-1)
        tips = (blend @ homogeneous[..., None])[..., :3, 0]
        result = torch.cat([joints, tips], dim=1)[:, ORDER] + translation[:, None]
        return scale[:, None, None] * (result - result[:, :1]) + wrist[:, None]

    return forward



def _image_target(state, seg, projected, anchors, params):
    residual = anchors - projected
    residual -= np.median(residual[:, :, BODY], axis=2, keepdims=True)
    image = state.get('_retained_wilor_2d')
    available = image is not None and np.isfinite(image).all(-1).any()
    if not available:
        return residual, 1 / (1 + (np.linalg.norm(residual, axis=-1) / 8) ** 2), False
    finite = np.isfinite(image).all(-1) & np.isfinite(projected).all(-1)
    discrepancy = np.where(finite[..., None], image - projected, 0.0)
    discrepancy -= np.median(discrepancy[:, :, BODY], axis=2, keepdims=True)
    extent = np.linalg.norm(projected[:, :, 9] - projected[:, :, 0], axis=-1)
    scale = np.maximum(6.0, .25 * extent)
    confidence = 1 / (1 + (np.linalg.norm(discrepancy, axis=-1) / scale[..., None]) ** 2)
    margin = np.minimum.reduce([image[..., 0], seg.W - 1 - image[..., 0],
                                image[..., 1], seg.H - 1 - image[..., 1]])
    confidence *= np.clip(margin / 5.0, 0, 1)
    confidence = np.where(finite, confidence, 0.0)
    confidence = gaussian_filter1d(confidence, 2.0, axis=0, mode='nearest')
    confidence = np.where(finite, confidence, 0.0)
    innovation = discrepancy - residual
    support = confidence * finite
    def smooth(sigma):
        denominator = gaussian_filter1d(support, sigma, axis=0, mode='constant')
        numerator = gaussian_filter1d(innovation * support[..., None], sigma, axis=0, mode='constant')
        return numerator / np.maximum(denominator, 1e-6)[..., None]
    residual += float(params.get('image_weight', .5)) * (smooth(2.0) - smooth(16.0)) * support[..., None]
    weights = 1 / (1 + (np.linalg.norm(residual, axis=-1) / 8) ** 2)
    weights *= .5 + .5 * confidence
    return residual, weights, True


def apply(state, seg, params):
    from harness.core import mano_models

    torch.set_num_threads(1)
    models = mano_models()
    hp = state['hp'].copy()
    old = {k: state[k].copy() for k in ('transl', 'joints_cam', 'verts_cam')}
    base_geometry = seg.mano(dict(state))
    relative = state['joints_cam'] - state['joints_cam'][:, :, :1]
    base_relative = base_geometry['joints_cam'] - base_geometry['joints_cam'][:, :, :1]
    mesh_scale = np.sum(relative * base_relative, axis=(2, 3)) / np.maximum(np.sum(base_relative ** 2, axis=(2, 3)), 1e-12)
    mesh_scale = np.clip(np.nan_to_num(mesh_scale, nan=1.0), .5, 2.0)
    joints = state['joints_cam']
    projected = seg.project(joints)
    anchors = state['joints_2d'].astype(np.float64)
    residual, weights, has_image = _image_target(state, seg, projected, anchors, params)
    target = (projected + residual - seg.K[:2, 2]) / np.array([seg.K[0, 0], seg.K[1, 1]])
    prior = float(params.get('prior', .001)) if has_image else .003
    depth_prior = float(params.get('depth_prior', 1.0)) if has_image else 0.0
    robust_scale = float(params.get('robust_mm', 10.0)) / 1000.0 if has_image else 0.0
    sigma = float(params.get('correction_sigma', 5.0))
    changed = False

    for side in (0, 1):
        image = projected[:, side]
        margin = np.minimum.reduce([image[..., 0], seg.W - 1 - image[..., 0],
                                    image[..., 1], seg.H - 1 - image[..., 1]])
        live = ((state['exists'][:, side] > .5)
                & np.isfinite(joints[:, side]).all((1, 2))
                & (joints[:, side, :, 2] > .05).all(-1)
                & (margin[:, 0] > 5) & ((margin > 5).sum(-1) >= 10)
                & np.isfinite(anchors[:, side]).all((1, 2))
                & np.isfinite(hp[:, side]).all((1, 2, 3))
                & np.isfinite(state['go'][:, side]).all((1, 2))
                & np.isfinite(state['betas'][:, side]).all(-1))
        if not live.any():
            continue

        h0 = torch.tensor(Rotation.from_matrix(hp[live, side].reshape(-1, 3, 3))
                          .as_rotvec().reshape(-1, 45), dtype=torch.float32)
        go = torch.tensor(Rotation.from_matrix(state['go'][live, side]).as_rotvec(), dtype=torch.float32)
        h = h0.clone().requires_grad_(True)
        shape = torch.tensor(state['betas'][live, side], dtype=torch.float32)
        translation = torch.tensor(state['transl'][live, side], dtype=torch.float32)
        target_t = torch.tensor(target[live, side], dtype=torch.float32)
        weight_t = torch.tensor(weights[live, side], dtype=torch.float32)
        depth = torch.tensor(joints[live, side, :, 2], dtype=torch.float32)
        model = models[side]
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        joint_forward = _prepare_joints(model, shape, translation,
                                        torch.tensor(mesh_scale[live, side], dtype=torch.float32),
                                        torch.tensor(joints[live, side, 0], dtype=torch.float32))
        optimizer = torch.optim.LBFGS([h], lr=1, max_iter=16 if has_image else 12, max_eval=20 if has_image else 16,
                                     history_size=5, tolerance_grad=1e-10,
                                     tolerance_change=1e-12, line_search_fn='strong_wolfe')

        def closure():
            optimizer.zero_grad()
            fitted = joint_forward(go, h)
            error = (fitted[..., :2] / fitted[..., 2:].clamp(min=.05) - target_t) * depth[..., None]
            squared = error.square().sum(-1)
            rho = robust_scale ** 2 * torch.log1p(squared / robust_scale ** 2) if robust_scale > 0 else squared
            loss = ((rho * weight_t).sum(-1).mean()
                    + depth_prior * ((fitted[..., 2] - depth).square() * weight_t).sum(-1).mean()
                    + prior * (h - h0).square().sum(-1).mean())
            loss.backward()
            return loss

        with torch.enable_grad():
            optimizer.step(closure)
        correction = np.zeros((seg.n, 45))
        correction[live] = (h - h0).detach().numpy()
        correction[~np.isfinite(correction)] = 0
        correction = gaussian_filter1d(correction, sigma, axis=0, mode='nearest').reshape(seg.n, 15, 3)
        correction *= np.minimum(1, np.radians(5) /
                                 np.maximum(np.linalg.norm(correction, axis=-1, keepdims=True), 1e-9))
        vectors = h0.numpy() + correction[live].reshape(-1, 45)
        hp[live, side] = Rotation.from_rotvec(vectors.reshape(-1, 3)).as_matrix().reshape(-1, 15, 3, 3)
        changed = True

    if changed:
        updated = np.any(hp != state['hp'], axis=(2, 3, 4))
        state['hp'] = hp
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
