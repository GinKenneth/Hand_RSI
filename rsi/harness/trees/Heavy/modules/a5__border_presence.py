"""a5__border_presence: image-side presence check for kept hands that touch the image border.

A hand placed with only a few joints inside the image is scored as present (TP or FP) by those joints alone, so at the
border the decision hinges on whether a hand is really there. Geometry alone cannot tell (HOT3D dev: hands with 1-3
joints in view are ~2:1 TP:FP). This module asks the image twice, using only evidence that already exists:
  * WiLoR (arXiv:2409.12259), run once by the `wilor` module on a crop around ACE's hand, regresses 2D keypoints. If a
    hand is in the crop, they land on ACE's hand (median disagreement ~0.25 of the hand extent); if the crop holds
    no hand, WiLoR invents one elsewhere (~0.45). `wilor` stores its ungated keypoints and ACE's projected joints at
    that time in state['_wilor_kp2d_all'] / state['_wilor_uv_at'] (no second pass).
  * the network's own per-frame visibility score exists_2d (e2).
A kept hand with 1..max_in joints in view (final placement) whose WiLoR disagreement is >= dis_min and whose e2 is
<= e2_max gets its existence lowered to exist_floor. Placement is untouched. Hands without WiLoR evidence pass through.
Deterministic, CPU only, per segment.
"""
import sys

import numpy as np


def _proj(K, X):
    z = np.maximum(X[..., 2:3], 1e-6)
    return np.concatenate([K[0, 0] * X[..., :1] / z + K[0, 2], K[1, 1] * X[..., 1:2] / z + K[1, 2]], -1)


def apply(state, seg, params):
    kw = state.get('_wilor_kp2d_all')
    ua = state.get('_wilor_uv_at')
    if kw is None or ua is None:
        print('a5__border_presence: no WiLoR keypoints in state, pass-through', file=sys.stderr)
        return state
    if 'exists_2d' not in seg.raw:
        print('a5__border_presence: no exists_2d, pass-through', file=sys.stderr)
        return state
    F = state['exists'].shape[0]
    K, W, H = np.asarray(seg.K, np.float64), seg.W, seg.H
    e2 = np.asarray(seg.raw['exists_2d'], np.float64)[:F].reshape(F, 2)
    max_in = int(params.get('max_in', 3))
    dis_min = float(params.get('dis_min', 0.5))
    e2_max = float(params.get('e2_max', 0.2))
    floor = float(params.get('exist_floor', 0.5))
    kw = np.asarray(kw, np.float64)
    ua = np.asarray(ua, np.float64)
    J = np.asarray(state['joints_cam'], np.float64)
    fin = np.isfinite(J).all((-1, -2)) & (J[..., 2] > 0.01).all(-1)
    uv = _proj(K, np.where(fin[..., None, None], J, 1.0))
    nin = ((uv[..., 0] > 0) & (uv[..., 0] < W) & (uv[..., 1] > 0) & (uv[..., 1] < H)).sum(-1)
    have = np.isfinite(kw).all((-1, -2)) & np.isfinite(ua).all((-1, -2))
    ext = np.maximum(np.ptp(np.where(have[..., None, None], ua, 0.0), axis=-2).max(-1), 1.0)
    dis = np.where(have, np.median(np.linalg.norm(np.where(have[..., None, None], kw - ua, 0.0), axis=-1), -1) / ext, np.nan)
    kept = fin & (state['exists'] > 0.5) & (nin >= 1) & (nin <= max_in)
    if kept.any() and not have.any():
        print('a5__border_presence: WiLoR produced no keypoints (model unavailable?), pass-through', file=sys.stderr)
        return state
    drop = kept & have & (dis >= dis_min) & np.isfinite(e2) & (e2 <= e2_max)
    if drop.any():
        ex = np.array(state['exists'], copy=True)
        ex[drop] = np.minimum(ex[drop], floor)
        state['exists'] = ex.astype(np.asarray(state['exists']).dtype)
    return state
