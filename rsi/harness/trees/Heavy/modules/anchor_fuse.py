"""2D anchor fusion: the network's direct 2D keypoints and the projection of its own MANO joints are two partly
independent estimates of the same image points; their weighted mean is a better 2D anchor. If the wilor module left
image-space WiLoR keypoints in state["wilor_2d"] (crop camera mapped back to the image), they are blended in with
finger-dependent weights (wrist 0 -> fingertips highest), following shared accepted R1_a7_req_7 / R1_a4_req_2."""
import numpy as np


def apply(state, seg, params):
    w = float(params.get("w_direct", 0.5))
    P = seg.project(state["joints_cam"].astype(np.float64))
    d = state["joints_2d"].astype(np.float64)
    ok = np.isfinite(P).all(-1) & np.isfinite(d).all(-1) & (state["joints_cam"][..., 2] > 0.01)
    out = np.where(ok[..., None], w * d + (1 - w) * P, d)
    wl = state.pop("wilor_2d", None)
    if wl is not None:
        wts = np.asarray(params.get("wilor_weights", [0.0, 0.15, 0.3, 0.4, 0.45]), np.float64)
        jw = np.r_[wts[0], np.tile(wts[1:5], 5)][:, None]
        v = np.isfinite(wl).all(-1) & np.isfinite(out).all(-1)
        out = np.where(v[..., None], (1 - jw) * out + jw * np.where(v[..., None], wl, 0.0), out)
    state["joints_2d"] = out.astype(state["joints_2d"].dtype)
    return state
