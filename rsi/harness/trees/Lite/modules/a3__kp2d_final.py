"""a3__kp2d_final: last-stage 2D keypoint output (EPE2D-p only; nothing else is touched).

Integrated in R4 from a3's R4 sub1 (bucket R4_a3_req_1); integration added the per-frame clamp `max_px`.

The exported 2D anchors (joints_2d) and the projection P of the final 3D hand are two estimates of the same image
points. Best dev setting {"w_proj": 0, "w_lo": 0.2, "lo_sigma": 4, "clip": true}:
    joints_2d += 0.2 * G_4(P - joints_2d)      G_4 = Gaussian (sigma 4 frames) within each run of existing hands
then clip to the image. The per-frame (high) band of P - joints_2d hurts (the anchors carry real per-frame image
motion), and the very-low band (sigma >= 8, i.e. the projection's per-track bias) hurts ARCTIC; the mid band helps.
Dev (parent preds, full dev): HOT3D EPE2D-p +1.39 % (S +0.139, CI90 [+0.112, +0.165]), 14/15 recordings better;
ARCTIC EPE2D-p +0.93 % (S +0.093), 34/37 recordings better. Other columns unchanged (exactly 0).
Original NumPy/SciPy implementation (BSD-3-Clause numpy/scipy); no paper-specific code.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d


def apply(state, seg, params):
    w = float(params.get("w_proj", 0.0))
    sig = float(params.get("sigma", 0.0))
    clip = bool(params.get("clip", False))
    K = np.asarray(seg.K, np.float64)
    d = np.asarray(state["joints_2d"], np.float64).copy()
    J = np.asarray(state["joints_cam"], np.float64)
    w_lo = float(params.get("w_lo", w))
    lo_sig = float(params.get("lo_sigma", 0.0))
    w_vlo = float(params.get("w_vlo", w_lo))
    vlo_sig = float(params.get("vlo_sigma", 0.0))
    if w > 0 or w_lo > 0:
        z = np.maximum(J[..., 2:3], 1e-6)
        P = np.concatenate([K[0, 0] * J[..., :1] / z + K[0, 2], K[1, 1] * J[..., 1:2] / z + K[1, 2]], -1)
        ok = np.isfinite(P).all(-1) & np.isfinite(d).all(-1) & (J[..., 2] > 0.01)
        diff = np.where(ok[..., None], P - d, 0.0)
        lo = np.zeros_like(diff); vlo = np.zeros_like(diff)
        if lo_sig > 0:
            okh = ok.all(-1) & (np.asarray(state["exists"]) > 0.5)
            F = d.shape[0]
            for s in range(d.shape[1]):
                t = 0
                while t < F:
                    if not okh[t, s]:
                        t += 1; continue
                    e = t
                    while e < F and okh[e, s]:
                        e += 1
                    lo[t:e, s] = gaussian_filter1d(diff[t:e, s], lo_sig, axis=0, mode="nearest")
                    if vlo_sig > 0:
                        vlo[t:e, s] = gaussian_filter1d(diff[t:e, s], vlo_sig, axis=0, mode="nearest")
                    t = e
        if vlo_sig <= 0:
            vlo = lo
        w_edge = float(params.get("w_edge", 0.0)); tau_e = float(params.get("edge_tau", 5.0))
        F = d.shape[0]; tt = np.arange(F, dtype=np.float64)
        we = (w_edge * (np.exp(-tt / tau_e) + np.exp(-(F - 1 - tt) / tau_e)))[:, None, None, None]
        corr = np.where(ok[..., None], w_vlo * vlo + w_lo * (lo - vlo) + w * (diff - lo) + we * lo, 0.0)
        max_px = float(params.get("max_px", 0.0))
        if max_px > 0:   # integration (R4): bound the per-frame, per-joint 2D correction
            nrm = np.linalg.norm(corr, axis=-1, keepdims=True)
            corr = corr * np.minimum(1.0, max_px / np.maximum(nrm, 1e-9))
        d = d + corr
    if sig > 0:
        F = d.shape[0]
        for s in range(d.shape[1]):
            ok = np.isfinite(d[:, s]).all((-1, -2)) & (np.asarray(state["exists"])[:, s] > 0.5)
            t = 0
            while t < F:
                if not ok[t]:
                    t += 1; continue
                e = t
                while e < F and ok[e]:
                    e += 1
                if e - t >= 3:
                    d[t:e, s] = gaussian_filter1d(d[t:e, s], sig, axis=0, mode="nearest")
                t = e
    if clip:
        d[..., 0] = np.clip(d[..., 0], 0, seg.W - 1e-3)
        d[..., 1] = np.clip(d[..., 1], 0, seg.H - 1e-3)
    state["joints_2d"] = d.astype(np.asarray(state["joints_2d"]).dtype)
    return state
