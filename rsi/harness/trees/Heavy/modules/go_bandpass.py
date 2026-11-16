"""Band-pass fusion of the global orientation with WiLoR's per-frame orientation (complementary filtering).

Diagnosis (HOT3D dev, the R1 chain): the network's global orientation (GO) has no time lag vs the GT but under-states
the hand's rotation speed (mean 2.17 vs 2.48 deg/frame, 5.7 vs 6.9 on fast turns), and its error grows with the
angular speed (4.0 deg on slow vs 5.9 on fast hands). GO is the main lever for MPJPE-p: the oracle GO cuts the
wrist-aligned error from 9.7 to 7.3 mm. WiLoR (run on every hand by the wilor module) gives an independent per-frame GO
from the image crop: noisier (median ~10 deg vs GT) and with its own slowly-varying bias, but it sees the real
frame-to-frame motion. A complementary filter takes from each source the band where it is better:

  D(t)  = log(R_ace(t)^T R_wilor(t))      body-frame discrepancy, per hand track
  w(t)  = exp(-(|D| / tau)^2 / 2)         soft agreement weight (WiLoR outliers / flips get ~0 weight)
  A_s   = weighted Gaussian mean of D over time with sigma s (normalised by the smoothed weight)
  R'(t) = R_ace(t) exp(beta (A_s1 - A_s2) c1 + lp_w A_s2 c2)

A_s1 - A_s2 is a band-pass of the discrepancy. The low band (sigma >= s2) is WiLoR's bias, dropped. The high band
(sigma < s1) is WiLoR's per-frame noise, which would show up as jitter, also dropped. What is left is the mid-band motion
the network smoothed away. c1, c2 = min(1, smoothed weight) fade the correction where WiLoR is missing.
Rotations are about the wrist joint (MANO rotates about its root), so joints_cam / verts_cam are rotated about joint 0
and transl is unchanged (the wrist does not move). The 2D anchors are left as they are.

WiLoR: arXiv 2409.12259 (Potamias et al.), via wilor_mini (CC BY-NC-ND 4.0 model; the module only consumes the wilor
module's outputs). Complementary filtering: the classical idea (e.g. Mahony et al., IEEE TAC 2008), own implementation.
"""
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation


def _wsmooth(x, w, sig):
    num = gaussian_filter1d(x * w[:, None], sig, axis=0, mode="constant")
    den = gaussian_filter1d(w, sig, mode="constant")
    return num / np.maximum(den, 1e-6)[:, None], den


def apply(state, seg, params):
    s1 = float(params.get("s1", 2.0))
    s2 = float(params.get("s2", 12.0))
    beta = float(params.get("beta", 0.5))
    tau = float(params.get("tau", 15.0))
    lp_w = float(params.get("lp_w", 0.0))
    min_frames = int(params.get("min_frames", 3))
    gw = state.get("wilor_go")
    if gw is None:
        sys.stderr.write("[go_bandpass] no WiLoR orientation in the state (wilor module missing); GO left unchanged\n")
        return state
    if not np.isfinite(gw).any():
        return state                     # WiLoR could not run on this segment (the wilor module reports why)
    go = state["go"]
    J, V = state["joints_cam"], state["verts_cam"]
    F = go.shape[0]
    for s in (0, 1):
        Ra = go[:, s].astype(np.float64)
        live = (state["exists"][:, s] > 0.5) & np.isfinite(Ra).all((1, 2)) & np.isfinite(J[:, s]).all((1, 2)) \
            & np.isfinite(V[:, s]).all((1, 2))
        ok = live & np.isfinite(gw[:, s]).all(-1)
        if ok.sum() < min_frames:
            continue
        D = np.zeros((F, 3))
        Rw = Rotation.from_rotvec(gw[ok, s]).as_matrix()
        D[ok] = Rotation.from_matrix(np.einsum('nji,njk->nik', Ra[ok], Rw)).as_rotvec()
        mag = np.degrees(np.linalg.norm(D, axis=1))
        w = ok * np.exp(-0.5 * (mag / tau) ** 2)
        if w.sum() < 1e-3:
            continue
        A1, d1 = _wsmooth(D, w, s1)
        A2, d2 = _wsmooth(D, w, s2)
        corr = beta * (A1 - A2) * np.minimum(d1, 1.0)[:, None]
        if lp_w > 0:
            corr = corr + lp_w * A2 * np.minimum(d2, 1.0)[:, None]
        idx = np.nonzero(live)[0]
        if not len(idx):
            continue
        dB = Rotation.from_rotvec(corr[idx]).as_matrix()               # body-frame correction
        Rn = np.einsum('nij,njk->nik', Ra[idx], dB)
        dR = np.einsum('nij,nkj->nik', Rn, Ra[idx])                      # camera-frame: Rn = dR Ra
        j0 = J[idx, s, :1].astype(np.float64)
        J[idx, s] = (np.einsum('nij,nkj->nki', dR, J[idx, s].astype(np.float64) - j0) + j0).astype(J.dtype)
        V[idx, s] = (np.einsum('nij,nkj->nki', dR, V[idx, s].astype(np.float64) - j0) + j0).astype(V.dtype)
        go[idx, s] = Rn.astype(go.dtype)
    state["go"], state["joints_cam"], state["verts_cam"] = go, J, V
    return state
