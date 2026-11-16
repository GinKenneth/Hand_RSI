"""a1__border_decide: one on/off decision for hands at the image border, at the end of the chain.

A predicted hand counts only if >= 1 of its joints projects into the image, so at the border a few pixels of placement
decide between TP / FN (GT on screen) and FP / TN (GT off screen). The network's 2D head never places keypoints outside
the image (HOT3D dev: all 12.8k kept hands placed fully off-screen have raw 2D joints 2-80 px inside), so the modules
that fit placement to 2D anchors pull truncated hands inward, and later placement modules can undo an earlier rescue.
This module re-decides at the end of the chain from two cues that do not depend on the final 3D placement: the
distance of the hand to the border relative to its projected size (rel), and the network's own per-frame visibility
score exists_2d (e2, optionally Gaussian-smoothed over time; sigma = 0 uses the raw per-frame score):
  rescue: a kept hand placed fully outside with rel <= r and e2 > e  -> translate it parallel to the image plane so its
          nearest joint lands margin_px inside (pose, shape, depth untouched; shift bounded by max_shift_m)
  drop:   a kept hand with only 1..max_in joints inside, its most-inside joint <= m_px from the border and e2 <= e
          -> existence lowered to exist_floor (placement untouched)
Every correction is bounded; missing inputs pass through. Deterministic, CPU only, per segment.
"""
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d


def _proj(K, X):
    z = np.maximum(X[..., 2:3], 1e-6)
    return np.concatenate([K[0, 0] * X[..., :1] / z + K[0, 2], K[1, 1] * X[..., 1:2] / z + K[1, 2]], -1)


def _sd(uv, W, H):
    """Signed distance of each point to the image border: > 0 inside, < 0 outside (pixels)."""
    du = np.minimum(uv[..., 0], W - uv[..., 0])
    dv = np.minimum(uv[..., 1], H - uv[..., 1])
    ins = (du > 0) & (dv > 0)
    return np.where(ins, np.minimum(du, dv), -np.hypot(np.minimum(du, 0), np.minimum(dv, 0)))


def apply(state, seg, params):
    if 'exists_2d' not in seg.raw:
        print('a1__border_decide: no exists_2d, pass-through', file=sys.stderr)
        return state
    F = state['exists'].shape[0]
    K, W, H = np.asarray(seg.K, np.float64), seg.W, seg.H
    e2 = np.asarray(seg.raw['exists_2d'], np.float64)[:F].reshape(F, 2)
    sig = float(params.get('sigma', 0.0))
    if sig > 0:
        e2 = gaussian_filter1d(e2, sig, axis=0, mode='nearest')
    rescue = params.get('rescue', [])            # [[rel_max, e2_min], ...]
    drop = params.get('drop', [])                # [[max_in, e2_max, m_px_max], ...]
    margin = float(params.get('margin_px', 1.0))
    max_shift = float(params.get('max_shift_m', 0.05))
    floor = float(params.get('exist_floor', 0.5))
    J = np.asarray(state['joints_cam'], np.float64)
    fin = np.isfinite(J).all((-1, -2)) & (J[..., 2] > 0.01).all(-1) & np.isfinite(e2)
    Jf = np.where(fin[..., None, None], J, 1.0)
    uv = _proj(K, Jf)
    sd = _sd(uv, W, H)
    nin = (sd > 0).sum(-1)                       # joints inside the image
    m = sd.max(-1)                               # most-inside joint: > 0 inside, < 0 = distance outside
    size = np.linalg.norm(uv.max(-2) - uv.min(-2), axis=-1)
    rel = -m / np.maximum(size, 1.0)             # outside distance relative to the projected hand size
    kept = fin & (state['exists'] > 0.5)
    shift = np.zeros((F, 2, 3))
    ex = np.array(state['exists'], copy=True)
    n_res = n_drop = 0
    for t in range(F):
        for s in (0, 1):
            if not kept[t, s]:
                continue
            if nin[t, s] == 0:
                if not any(rel[t, s] <= r and e2[t, s] > e for r, e in rescue):
                    continue
                j = int(np.argmax(sd[t, s]))
                u, v = uv[t, s, j]
                tu, tv = np.clip(u, margin, W - 1 - margin), np.clip(v, margin, H - 1 - margin)
                z = J[t, s, j, 2]
                d = np.array([(tu - u) * z / K[0, 0], (tv - v) * z / K[1, 1], 0.0])
                if np.linalg.norm(d) <= max_shift:
                    shift[t, s] = d
                    n_res += 1
            elif any(nin[t, s] <= mi and e2[t, s] <= e and m[t, s] <= mp for mi, e, mp in drop):
                ex[t, s] = min(ex[t, s], floor)
                n_drop += 1
    if n_res:
        state['transl'] = (state['transl'] + shift).astype(np.float32)
        state['joints_cam'] = (state['joints_cam'] + shift[:, :, None]).astype(np.float32)
        state['verts_cam'] = (state['verts_cam'] + shift[:, :, None]).astype(np.float32)
    state['exists'] = ex.astype(np.asarray(state['exists']).dtype)
    return state
