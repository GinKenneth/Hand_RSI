"""Retain the initial calibrated projection as an independent 2D reference.

Runs before geometry refinement, using only the current segment's state.
Original implementation; NumPy BSD-3-Clause.
"""
import numpy as np


def apply(state, seg, params):
    points = np.asarray(state['joints_cam'])
    reference = seg.project(points)
    valid = np.isfinite(points).all(-1) & (points[..., 2] > .01)
    state['_anchor_initial_projection'] = np.where(valid[..., None], reference, np.nan)
    return state
