"""Shared helpers of the classical post-processing baselines.

Standard filters applied to every per-frame quantity the evaluator scores: MANO global orientation and finger
rotations (filtered element-wise, then projected back to SO(3)), camera translation, and the network's 2D keypoints.
No GT, no image, no other network; one 81-frame segment at a time, like every harness module.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

FREQ = 30.0                                   # frame rate assumed by One-Euro (HOT3D / ARCTIC are 30 fps)
SCALE = {"rot": 1.0, "transl": 0.1, "px": 100.0}   # One-Euro: signals normalised to ~unit speed before filtering


def so3(A):
    u, _, vt = np.linalg.svd(A)
    u[..., :, -1] *= np.linalg.det(u @ vt)[..., None]
    return u @ vt


def gauss(x, sigma):
    return x if sigma <= 0 else gaussian_filter1d(x, sigma, axis=0, mode="nearest", truncate=3.0)


def savgol(x, window, order):
    if window <= order or len(x) < window:
        return x
    return savgol_filter(x, window, order, axis=0, mode="interp")


def one_euro(x, mincutoff, beta, dcutoff=1.0, freq=FREQ, scale=1.0):
    """Casiez et al., CHI 2012 (causal), element-wise over all trailing dims."""
    x = np.asarray(x, np.float64) / scale
    te = 1.0 / freq
    alpha = lambda c: 1.0 / (1.0 + (1.0 / (2 * np.pi * c)) / te)
    y = np.empty_like(x); y[0] = x[0]
    edx = np.zeros_like(x[0]); ad = alpha(dcutoff)
    for t in range(1, len(x)):
        dx = (x[t] - y[t - 1]) * freq
        edx = ad * dx + (1 - ad) * edx
        a = alpha(mincutoff + beta * np.abs(edx))
        y[t] = a * x[t] + (1 - a) * y[t - 1]
    return y * scale


def apply_all(state, seg, f):
    """f(x, kind) -> filtered x, applied per hand slot; non-finite series are left untouched."""
    for key, kind in (("go", "rot"), ("hp", "rot"), ("transl", "transl"), ("joints_2d", "px")):
        x = np.asarray(state[key])
        out = x.copy()
        for s in (0, 1):
            xs = x[:, s].astype(np.float64)
            if not np.isfinite(xs).all():
                continue
            ys = f(xs, kind)
            if kind == "rot":
                ys = so3(ys)
            out[:, s] = ys.astype(x.dtype)
        state[key] = out
    return seg.mano(state)
