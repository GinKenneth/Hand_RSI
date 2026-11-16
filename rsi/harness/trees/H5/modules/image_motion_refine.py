"""Bounded image-motion correction of temporally fused 2D anchors.

Bidirectional Lucas-Kanade tracks at gaps 1, 2 and 4 constrain the differences
of a zero-prior correction trajectory. Median non-tip motion reduces texture
noise. A bounded temporal unsharp residual restores attenuated motion.
Only this segment's frames and current 2D anchors are used.

Adapted from public accepted R3_a7_req_1 anchor_consensus_flow and
anchor_motion_restore: original a7 NumPy/SciPy code, BSD-3-Clause; OpenCV
Apache-2.0; pyramidal LK, Bouguet 2001. No added weights or datasets.
"""
import sys

import numpy as np
from scipy.linalg import solveh_banded


def _motion_correction(base, seg, strength, distances, quality, shared=0.0):
    import cv2

    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    frames = seg.frames()
    n = min(len(base), len(frames))
    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames[:n]]
    width = max(distances)
    band = np.zeros((width + 1, n, 2, 21))
    band[-1] = 1.0
    rhs = np.zeros((n, 2, 21, 2))
    common_band = np.zeros((width + 1, n, 2))
    common_band[-1] = 1.0
    common_rhs = np.zeros((n, 2, 2))
    body = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
    opt = dict(winSize=(21, 21), maxLevel=3,
               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, .01),
               minEigThreshold=1e-4, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    finite = np.isfinite(base[:n]).all(-1)
    points = np.where(finite[..., None], base[:n], 0).astype(np.float32)

    def track(a, b):
        p = points[a].reshape(-1, 1, 2)
        q, status, error = cv2.calcOpticalFlowPyrLK(
            gray[a], gray[b], p, points[b].reshape(-1, 1, 2).copy(), **opt)
        back, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            gray[b], gray[a], q, p.copy(), **opt)
        uv = p[:, 0]
        shift = points[b].reshape(-1, 2) - q[:, 0]
        good = (finite[a].ravel() & finite[b].ravel()
                & (uv[:, 0] >= 0) & (uv[:, 0] < seg.W)
                & (uv[:, 1] >= 0) & (uv[:, 1] < seg.H)
                & (status[:, 0] > 0) & (backward_status[:, 0] > 0)
                & (np.linalg.norm(back[:, 0] - uv, axis=-1) < 1.0)
                & (np.linalg.norm(shift, axis=-1) < 20.0)
                & np.isfinite(shift).all(-1) & (error[:, 0] < quality))
        return np.where(good[:, None], shift, 0).reshape(2, 21, 2), good.reshape(2, 21)

    for distance in distances:
        for a in range(n - distance):
            b = a + distance
            forward, good_forward = track(a, b)
            backward, good_backward = track(b, a)
            count = good_forward.astype(float) + good_backward
            weight = strength * count / 2.0
            target = (-forward + backward) / np.maximum(count, 1)[..., None]
            band[-1, a] += weight
            band[-1, b] += weight
            band[width - distance, b] -= weight
            rhs[a] -= weight[..., None] * target
            rhs[b] += weight[..., None] * target
            if shared > 0:
                shifts = np.zeros((2, 2, 2))
                accepted = np.zeros((2, 2))
                for direction, (values, reliable) in enumerate(((forward, good_forward), (backward, good_backward))):
                    for side in (0, 1):
                        visible = reliable[side, body]
                        if np.count_nonzero(visible) >= 4:
                            shifts[direction, side] = np.median(values[side, body][visible], axis=0)
                            accepted[direction, side] = 1.0
                count_common = accepted.sum(0)
                weight_common = strength * count_common / 2.0
                target_common = (-shifts[0] + shifts[1]) / np.maximum(count_common, 1)[:, None]
                common_band[-1, a] += weight_common
                common_band[-1, b] += weight_common
                common_band[width - distance, b] -= weight_common
                common_rhs[a] -= weight_common[:, None] * target_common
                common_rhs[b] += weight_common[:, None] * target_common
    correction = np.zeros_like(base)
    for side in (0, 1):
        for joint in range(21):
            correction[:n, side, joint] = solveh_banded(
                band[:, :, side, joint], rhs[:, side, joint], check_finite=False)
    if shared > 0:
        common = np.zeros((len(base), 2, 1, 2))
        for side in (0, 1):
            common[:n, side, 0] = solveh_banded(
                common_band[:, :, side], common_rhs[:, side], check_finite=False)
        correction = (1.0 - shared) * correction + shared * common
    return correction


def apply(state, seg, params):
    from scipy.ndimage import gaussian_filter1d
    anchors = np.asarray(state['joints_2d'])
    valid = np.isfinite(anchors).all(-1)
    refined = anchors.copy()
    if len(anchors) >= 3 and float(params.get('flow_strength', .3)) > 0:
        try:
            distances = tuple(int(d) for d in params.get('distances', [1, 2, 4])
                              if 0 < int(d) < len(anchors))
            if distances:
                correction = _motion_correction(
                    anchors, seg, float(params.get('flow_strength', .3)), distances,
                    float(params.get('flow_quality', 10.0)),
                    float(params.get('motion_share', .5)))
                correction *= np.minimum(
                    1.0, float(params.get('flow_cap', 8.0)) /
                    np.maximum(np.linalg.norm(correction, axis=-1, keepdims=True), 1e-9))
                refined = anchors + correction
        except Exception as exc:
            sys.stderr.write(f'[image_motion_refine] image tracking unavailable ({exc!r}); using temporal anchors\n')
    values = np.where(valid[..., None], refined, 0).astype(np.float64)
    sigma = float(params.get('sigma', 2.0))
    den = gaussian_filter1d(valid.astype(float), sigma, axis=0, mode='nearest')
    smooth = gaussian_filter1d(values, sigma, axis=0, mode='nearest') / np.maximum(den, 1e-8)[..., None]
    correction = values - smooth
    correction *= np.minimum(
        1.0, float(params.get('motion_cap', 8.0)) /
        np.maximum(np.linalg.norm(correction, axis=-1, keepdims=True), 1e-9))
    result = values + float(params.get('weight', .5)) * correction
    state['joints_2d'] = np.where(valid[..., None], result, anchors).astype(anchors.dtype)
    return state
