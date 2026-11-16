"""MANO helpers in ACE-Ego-Hand's convention (ace_ego_hand/mano_utils.py).

21 joints = smplx's 16 MANO joints + 5 fingertip vertices, reordered to OpenPose;
smplx with flat_hand_mean=False (ACE's setting for HOT3D / ARCTIC / HOI4D), so a
zero hand_pose is MANO's mean pose. The canonical placeholder of (S8) is the hand
with identity global orientation, zero hand_pose, zero betas and tau = 0.
"""
from __future__ import annotations

import os

import numpy as np

MANO_DIR = os.environ.get(       # converted MANO_LEFT.pkl / MANO_RIGHT.pkl (ACE-Ego-Hand scripts/convert_mano_pkls.py)
    "ACE_EVAL_MANO_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "mano"))
TIP_IDS = [744, 320, 443, 554, 671]
MANO_TO_OP = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
CANON_NPZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "canonical_mano.npz")


def load_models(flat_hand_mean: bool = False) -> dict:
    import smplx
    return {s: smplx.create(MANO_DIR, model_type="mano", is_rhand=bool(s), use_pca=False,
                            flat_hand_mean=flat_hand_mean).eval() for s in (0, 1)}


def to_axis_angle(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) rotation matrices or (..., 3) axis-angle -> (..., 3) axis-angle."""
    if R.shape[-1] == 3 and R.shape[-2:] != (3, 3):
        return R
    from scipy.spatial.transform import Rotation
    shp = R.shape[:-2]
    flat = R.reshape(-1, 3, 3)
    out = np.full((len(flat), 3), np.nan)
    ok = np.isfinite(flat).all((1, 2))
    if ok.any():
        out[ok] = Rotation.from_matrix(flat[ok]).as_rotvec()
    return out.reshape(shp + (3,))


def forward(models: dict, side: int, go: np.ndarray, hp: np.ndarray, betas: np.ndarray):
    """Batched MANO forward at tau = 0. go (B,3,3)|(B,3); hp (B,15,3,3)|(B,15,3)|(B,45);
    betas (B,10). Returns joints (B,21,3) and vertices (B,778,3), NaN where any input is NaN."""
    import torch
    B = len(go)
    go_aa = to_axis_angle(np.asarray(go, np.float64).reshape(B, *np.shape(go)[1:])).reshape(B, 3)
    hp = np.asarray(hp, np.float64)
    hp_aa = to_axis_angle(hp.reshape(B, 15, 3, 3)) if hp.shape[-2:] == (3, 3) else hp
    hp_aa = hp_aa.reshape(B, 45)
    betas = np.asarray(betas, np.float64).reshape(B, -1)
    ok = np.isfinite(go_aa).all(1) & np.isfinite(hp_aa).all(1) & np.isfinite(betas).all(1)
    J = np.full((B, 21, 3), np.nan, np.float32)
    V = np.full((B, 778, 3), np.nan, np.float32)
    if ok.any():
        t = lambda x: torch.as_tensor(x[ok], dtype=torch.float32)   # noqa: E731
        with torch.no_grad():
            out = models[side](global_orient=t(go_aa), hand_pose=t(hp_aa), betas=t(betas),
                               transl=torch.zeros(int(ok.sum()), 3))
        j = torch.cat([out.joints, out.vertices[:, TIP_IDS]], 1)[:, MANO_TO_OP]
        J[ok] = j.numpy()
        V[ok] = out.vertices.numpy()
    return J, V


def canonical_joints(flat_hand_mean: bool = False) -> np.ndarray:
    """(2, 21, 3) canonical placeholder joints for slot 0 (left) and 1 (right); cached."""
    key = "flat" if flat_hand_mean else "mean"
    if os.path.exists(CANON_NPZ):
        d = np.load(CANON_NPZ)
        if key in d:
            return d[key]
    out = {}
    for k, flat in (("mean", False), ("flat", True)):
        m = load_models(flat)
        out[k] = np.stack([forward(m, s, np.zeros((1, 3)), np.zeros((1, 45)), np.zeros((1, 10)))[0][0]
                           for s in (0, 1)])
    np.savez(CANON_NPZ, **out)
    return out[key]
