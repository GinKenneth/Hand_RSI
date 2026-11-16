"""Retain direct image evidence after repeated geometric anchor fusion.

Successive projection blends share the same MANO placement error. A small
contribution from the original image head restores independent localization
evidence without another temporal filter or a perspective depth assumption.
"""
import numpy as np


def apply(state, seg, params):
    if "direct_joints2d" not in seg.raw:
        return state
    anchors = np.asarray(state["joints_2d"])
    direct = np.asarray(seg.raw["direct_joints2d"], np.float64)[:len(anchors)]
    direct = direct.reshape(anchors.shape) * np.array([seg.W, seg.H])
    valid = np.isfinite(direct).all(-1) & np.isfinite(anchors).all(-1)
    alpha = np.clip(float(params.get("alpha", 0.25)), 0.0, 1.0)
    balanced = anchors + alpha * (direct - anchors)
    state["joints_2d"] = np.where(valid[..., None], balanced, anchors).astype(anchors.dtype)
    return state
