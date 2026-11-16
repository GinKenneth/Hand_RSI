"""Fuse two estimates of the same image-space joints.

The direct 2D head supplies image evidence; the placed MANO projection supplies
the articulated geometry constraint. Equal weighting avoids selecting either
head as a privileged source when both provide finite, forward-facing estimates.
"""
import numpy as np


def apply(state, seg, params):
    anchors = np.asarray(state['joints_2d'])
    joints = np.asarray(state['joints_cam'])
    projected = seg.project(joints)
    good = (np.isfinite(projected).all(-1)
            & np.isfinite(anchors).all(-1) & (joints[..., 2] > 0.01))
    weight = float(params.get('projection_weight', 0.5))
    fused = anchors + weight * (projected - anchors)
    state['joints_2d'] = np.where(good[..., None], fused, anchors)
    return state
