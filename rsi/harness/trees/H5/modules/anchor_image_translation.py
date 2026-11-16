"""Small 2D corrections from a geometric reference and image hand placement.

The initial calibrated MANO projection supplies a bounded reference that has
not accumulated the later geometric refinements. Smooth its bounded residual,
not the moving hand itself. WiLoR crop-camera evidence then estimates a common
translation from non-tip joints, rejecting inconsistent hand shape. This
recovers image placement without copying WiLoR's articulation error.

Original implementation; NumPy and SciPy BSD-3-Clause. No new model inference.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d


_BODY = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]


def _bound(v, radius):
    return v * np.minimum(1.0, radius / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9))


def apply(state, seg, params):
    source = np.asarray(state['joints_2d'])
    anchors = source.astype(np.float64)
    reference = state.pop('_anchor_initial_projection', None)
    image = state.get('_retained_wilor_2d')
    if reference is not None:
        valid = np.isfinite(reference).all(-1) & np.isfinite(anchors).all(-1)
        residual = _bound(np.where(valid[..., None], reference - anchors, 0.0),
                          float(params.get('reference_cap', 2.0)))
        sigma = float(params.get('reference_sigma', 3.0))
        if sigma > 0:
            residual = gaussian_filter1d(residual, sigma, axis=0, mode='nearest')
        anchors += float(params.get('reference_weight', .25)) * np.where(valid[..., None], residual, 0.0)
    if image is not None:
        residual = np.asarray(image, np.float64)[:, :, _BODY] - anchors[:, :, _BODY]
        valid = np.isfinite(residual).all((2, 3))
        offset = np.zeros((*anchors.shape[:2], 1, 2), np.float64)
        if np.any(valid):
            values = residual[valid]
            common = np.median(values, axis=1, keepdims=True)
            shape_error = np.median(np.linalg.norm(values - common, axis=-1), axis=-1)
            common = _bound(common, float(params.get('image_cap', 10.0)))
            offset[valid] = np.where((shape_error < float(params.get('shape_gate', 8.0)))[:, None, None], common, 0.0)
        anchors += float(params.get('image_weight', .25)) * offset
    state['joints_2d'] = np.where(np.isfinite(anchors), anchors, source).astype(source.dtype)
    return state
