"""Bounded complementary correction from existing crop-camera 2D evidence.

Retain the middle temporal band of the image disagreement. A common hand
shift is allowed; inconsistent articulation reduces confidence. Only finite,
agreed image predictions are updated. No extra image inference.
Original NumPy/SciPy implementation (BSD-3-Clause), complementary filter idea
Mahony et al., IEEE TAC 2008. Supplied WiLoR arXiv:2409.12259, model
CC BY-NC-ND 4.0. No weights are included.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

BODY = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]


def apply(state, seg, params):
    image = state.get('_retained_wilor_2d')
    if image is None:
        return state
    anchors = state['joints_2d']
    valid = np.isfinite(image).all(-1) & np.isfinite(anchors).all(-1)
    if not valid.any():
        return state
    residual = np.where(valid[..., None], image - anchors, 0).astype(np.float64)
    common = np.median(residual[:, :, BODY], axis=2, keepdims=True)
    agreement = float(params.get('agreement_px', 12))
    weights = valid * np.exp(-.5 * (np.linalg.norm(residual - common, axis=-1) / agreement) ** 2)

    def smooth(sigma):
        den = gaussian_filter1d(weights, sigma, axis=0, mode='constant')
        num = gaussian_filter1d(residual * weights[..., None], sigma, axis=0, mode='constant')
        return num / np.maximum(den, 1e-6)[..., None], den

    near, confidence = smooth(float(params.get('sigma_near', 1)))
    far, _ = smooth(float(params.get('sigma_far', 8)))
    correction = (near - far) * np.minimum(confidence, 1)[..., None]
    cap = float(params.get('cap', 8))
    correction *= np.minimum(1, cap / np.maximum(np.linalg.norm(correction, axis=-1, keepdims=True), 1e-9))
    correction *= float(params.get('weight', .25))
    state['joints_2d'] = np.where(valid[..., None], anchors + correction, anchors).astype(anchors.dtype)
    return state
