"""Keyframe-aware root re-interpolation + L1 acceleration smoothing of the hand (a5, R4).

Why (HOT3D / ARCTIC dev, measured on the network's raw outputs): the network decodes an 81-frame segment at temporal
keyframes every `stride` = 4 frames (0, 4, ..., 80) and interpolates linearly in between. Its raw root translation has
its acceleration on the keyframes (HOT3D: 6.41 mm/frame^2 at frames 4k vs 0.83 elsewhere; direct-wrist head 5.87 vs
0.14; ground truth is flat over the phase). After the R3 chain the root still has the kinks (2.64 vs 0.80) and its
error is lowest on the keyframes (CT 10.52 mm at 4k vs 10.75 / 10.88 / 10.72 at 4k+1..3). Two consequences:
  1. linear interpolation between keyframes cuts the corners of a curved trajectory: sampled at the same keyframes, a
     C2 cubic spline reproduces the ground-truth root with 1.05 mm error vs 2.10 mm for the linear one;
  2. but an interpolating spline through noisy keyframes overshoots, which raises the 3D acceleration (Jitter) a lot.
What:
  a) the chain's translation x(t) = L(t) + r(t), with L the piecewise-linear interpolant through its own keyframe
     values; the target is y(t) = x(t) + beta * (S(t) - L(t)), S a not-a-knot cubic spline through the same keyframes
     (the per-frame residual r the chain added is kept);
  b) one rigid translation offset d(t) per frame is then found by
         min_d  1/2 sum_t ||y(t) + d(t) - x(t) - c(t)||^2  +  lam * sum_t mean_k || D2 (J_k + c + d)(t) ||
     i.e. L1 (total variation of velocity) on the second differences of all 21 joints of the corrected hand, with the
     fidelity term pulling to the spline target. The group-L1 is exactly the Jitter definition, so the smoother removes
     the overshoot of (a) and the remaining keyframe kinks while keeping the corner correction. Solved by ADMM
     (Boyd et al., "Distributed optimization ... ADMM", FnT ML 2011; L1 trend filtering: Kim, Koh, Boyd, Gorinevsky,
     SIAM Review 2009). Original NumPy/SciPy code, BSD-3-Clause.
  c) noise-adaptive strength: the size of a run's keyframe kinks measures how noisy its keyframe estimates are, so
     lam_run = lam * clip(kink / adapt, lo, hi) ** pw, kink = mean |D2 x| on keyframes - mean |D2 x| between them
     (a discrepancy-principle style choice of the regularisation weight: noisier tracks are smoothed more, clean ones
     less). Dev: HOT3D kink median 1.38 mm/frame^2, ARCTIC 0.81 (the more accurate network output is left closer to
     itself: ARCTIC CT -1.01 % vs -1.40 % with a fixed lam at the same HOT3D gain).
  The hand is translated rigidly (transl, joints_cam, verts_cam); orientation, articulation, shape and the 2D anchors
  are unchanged. Only runs of consecutive frames with a predicted (exists > 0.5), finite hand are used. A hand-frame
  whose in-image status would flip is left unchanged (detection is not this module's business).
"""
import numpy as np
from scipy.interpolate import CubicSpline


def _in_image(seg, X):
    uv = seg.project(X)
    return ((X[..., 2] > .01) & (uv[..., 0] >= 0) & (uv[..., 0] < seg.W) & (uv[..., 1] >= 0) & (uv[..., 1] < seg.H)).any(-1)


def _runs(mask):
    out, i, n = [], 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        out.append((i, j + 1))
        i = j + 1
    return out


def _spline_correction(x, a, b, stride, min_knots):
    """x (F,3) trajectory, run [a,b) -> (b-a,3) cubic-spline minus linear interpolant through the keyframes."""
    t = np.arange(a, b)
    knots = [k for k in range(a, b) if k % stride == 0]
    if len(knots) < min_knots:
        return np.zeros((b - a, 3))
    pts = np.asarray(sorted(set(knots) | {a, b - 1}))
    y = x[pts]
    lin = np.stack([np.interp(t, pts, y[:, c]) for c in range(3)], -1)
    return CubicSpline(pts, y, axis=0, bc_type='not-a-knot')(t) - lin


def _l1_rigid(J, lam, rho, iters):
    """J (T,K,3) joints. d (T,3) minimising 1/2 sum_t ||d_t||^2 + lam sum_t mean_k ||D2(J_k + d)_t|| (ADMM)."""
    T, nk = J.shape[:2]
    D = np.zeros((T - 2, T))
    i = np.arange(T - 2)
    D[i, i], D[i, i + 1], D[i, i + 2] = 1., -2., 1.
    Ai = np.linalg.inv(np.eye(T) + rho * nk * D.T @ D)
    a = np.einsum('st,tkc->skc', D, J)
    z, u = a.copy(), np.zeros_like(a)
    thr = lam / nk / rho
    d = np.zeros((T, 3))
    for _ in range(iters):
        d = Ai @ (rho * D.T @ (z - a - u).sum(1))
        v = (D @ d)[:, None, :] + a + u
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        z = v * np.maximum(0., 1. - thr / np.maximum(n, 1e-15))
        u = v - z
    return d


def apply(state, seg, params):
    stride = int(params.get('stride', 4))
    beta = float(params.get('beta', 0.5))
    lam = float(params.get('lam', 0.012))
    rho = float(params.get('rho', 1.0))
    iters = int(params.get('iters', 300))
    min_knots = int(params.get('min_knots', 4))
    min_run = int(params.get('min_run', 8))
    max_shift = float(params.get('max_shift', 0.02))
    guard = bool(params.get('guard', True))
    adapt = float(params.get('adapt', 0.))
    lo, hi, pw = float(params.get('lo', .25)), float(params.get('hi', 4.)), float(params.get('pw', 1.))
    tr = np.asarray(state['transl'], np.float64)
    J = np.asarray(state['joints_cam'], np.float64)
    F = tr.shape[0]
    ok = np.isfinite(tr).all(-1) & np.isfinite(J).all((-1, -2)) & (np.asarray(state['exists']) > 0.5)
    delta = np.zeros((F, 2, 3))
    for s in (0, 1):
        for a, b in _runs(ok[:, s]):
            if b - a < min_run:
                continue
            lam_r = lam
            if adapt > 0 and b - a >= 12:
                acc = np.linalg.norm(tr[a + 2:b, s] - 2 * tr[a + 1:b - 1, s] + tr[a:b - 2, s], axis=-1)
                f = np.arange(a + 1, b - 1)
                kink = max(acc[f % stride == 0].mean() - acc[f % stride != 0].mean(), 0.)
                lam_r = lam * float(np.clip(kink / adapt, lo, hi)) ** pw
            c = beta * _spline_correction(tr[:, s], a, b, stride, min_knots) if beta else np.zeros((b - a, 3))
            d = _l1_rigid(J[a:b, s] + c[:, None, :], lam_r, rho, iters) if lam_r > 0 else 0.
            delta[a:b, s] = c + d
    n = np.linalg.norm(delta, axis=-1)
    delta *= np.minimum(1., max_shift / np.maximum(n, 1e-12))[..., None]
    if not np.any(delta):
        return state
    j_new = J + delta[:, :, None, :]
    if guard:
        flip = _in_image(seg, J) != _in_image(seg, j_new)
        delta[flip] = 0.
        j_new = J + delta[:, :, None, :]
    state['transl'] = (tr + delta).astype(np.float32)
    state['joints_cam'] = j_new.astype(np.float32)
    state['verts_cam'] = (np.asarray(state['verts_cam'], np.float64) + delta[:, :, None, :]).astype(np.float32)
    return state
