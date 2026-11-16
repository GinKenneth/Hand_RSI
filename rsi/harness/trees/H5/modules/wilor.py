"""wilor: image-based finger articulation and 2D evidence from WiLoR, fused into ACE's hands.

For every hand ACE says exists, the frame is cropped around ACE's own projected MANO mesh (box 2.5x the mesh extent,
256 px, left hands mirrored, anti-alias blur as in WiLoR's demo) and WiLoR regresses MANO hand_pose and 3D keypoints.
  * articulation: per joint, ACE's full finger rotation (hand_pose + MANO mean) is slerped a fraction `alpha` towards
    WiLoR's (temporally smoothed over the segment) — two independent image models averaged in rotation space.
    Global orientation, translation and shape stay ACE's.
  * 2D: WiLoR's crop-camera keypoints mapped back to the image (scale-free, no depth/focal assumption) are stored in
    state["wilor_2d"] for fuse2d.
A hand-frame whose WiLoR 2D disagrees with ACE's 2D by more than `gate` x the hand box size (median over joints) is
treated as a WiLoR failure (wrong hand / occlusion) and not used.

WiLoR: Potamias et al., "WiLoR: End-to-end 3D hand localization and reconstruction in-the-wild", CVPR 2025,
arXiv:2409.12259 (github.com/rolpotamias/WiLoR, CC BY-NC-ND 4.0); wrapper wilor_mini (github.com/warmshao/WiLoR-mini).
Crop / crop-camera code follows wilor_mini.utils.generate_image_patch_cv2 and ace_eval/baselines/diag_wilor2d.py.
Optional perspective rectification follows Yu et al., Perspective Crop Layers,
arXiv:2011.13607 (paper CC BY 4.0); original NumPy/OpenCV implementation.
"""
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

_CACHE = {}


def _root():
    """Directory holding wilor_mini/ with its pretrained weights: $WILOR_MINI_ROOT, else an importable wilor_mini, else
    <sys.prefix>/share/wilor_root. None if no weights are found (the module is then a no-op)."""
    if "root" not in _CACHE:
        cands = [os.environ.get("WILOR_MINI_ROOT", "")]
        try:
            import importlib.util
            sp = importlib.util.find_spec("wilor_mini")
            if sp is not None and sp.origin:
                cands.append(os.path.dirname(os.path.dirname(sp.origin)))
        except Exception:  # noqa: BLE001
            pass
        cands += [os.path.join(sys.prefix, "share", "wilor_root")]
        _CACHE["root"] = None
        for c in cands:
            pre = os.path.join(c, "wilor_mini", "pretrained_models") if c else ""
            if c and all(os.path.isfile(os.path.join(pre, f)) for f in ("wilor_final.ckpt", "MANO_RIGHT.pkl", "mano_mean_params.npz")):
                _CACHE["root"] = c
                break
        if _CACHE["root"] is None:
            sys.stderr.write("[wilor] WiLoR weights not found (WILOR_MINI_ROOT); module disabled\n")
    return _CACHE["root"]


def _model():
    if "m" in _CACHE:
        return _CACHE["m"]
    root = _root()
    if root is None:
        _CACHE["m"] = None
        return None
    if root not in sys.path:
        sys.path.insert(0, root)
    for name, value in {"bool": np.bool_, "int": np.int_, "float": np.float64, "complex": np.complex128,
                        "object": np.object_, "unicode": np.str_, "str": np.str_}.items():
        if not hasattr(np, name):
            setattr(np, name, value)
    try:
        import torch
        from wilor_mini.models.wilor import WiLor
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        pre = os.path.join(root, "wilor_mini", "pretrained_models")
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = WiLor(mano_model_path=os.path.join(pre, "MANO_RIGHT.pkl"), mano_mean_path=os.path.join(pre, "mano_mean_params.npz"),
                  focal_length=5000, image_size=256)
        m.load_state_dict(torch.load(os.path.join(pre, "wilor_final.ckpt"), map_location="cpu")["state_dict"], strict=False)
        m.eval()
        # Autocast already uses half precision for the image network. Store its
        # weights in that precision before transfer to reduce shared-card OOM.
        if dev.type == "cuda":
            m.half()
        m.to(dev)
        _CACHE["m"] = (m, dev)
    except Exception as e:  # noqa: BLE001
        failure = (str(e).splitlines() or [repr(e)])[0].split("Process ")[0][:240]
    else:
        return _CACHE["m"]
    # Leave the exception handler before freeing a partially transferred model:
    # its traceback retains tensors until the exception itself is released.
    # A busy card can recover later in the replay, so do not cache this failure.
    if "m" in locals():
        del m
    import gc
    gc.collect()
    if "torch" in locals() and torch.cuda.is_available():
        torch.cuda.empty_cache()
    sys.stderr.write(f"[wilor] model unavailable ({failure}); passing through this segment; will retry\n")
    return None


def _mano_mean():
    if "mean" not in _CACHE:
        import rsi_paths  # noqa: F401  (puts ace_eval on sys.path)
        from ace_eval.mano import load_models                              # the models seg.mano uses
        ms = load_models(flat_hand_mean=False)
        _CACHE["mean"] = {s: ms[s].hand_mean.detach().cpu().numpy().reshape(15, 3).astype(np.float64) for s in (0, 1)}
    return _CACHE["mean"]


def _patch(img, cx, cy, size, flip, out=256):
    """WiLoR's crop: blur (anti-alias) then affine-warp a (size x size) box at (cx, cy) to out x out; only the box
    region of the image is blurred (equivalent to blurring the whole frame, far cheaper)."""
    import cv2
    H, W = img.shape[:2]
    ds = size / out / 2.0
    sig = (ds - 1) / 2 if ds > 1.1 else 0.0
    pad = int(np.ceil(4 * sig)) + 2
    x0 = int(max(0, np.floor(cx - size / 2) - pad)); x1 = int(min(W, np.ceil(cx + size / 2) + pad))
    y0 = int(max(0, np.floor(cy - size / 2) - pad)); y1 = int(min(H, np.ceil(cy + size / 2) + pad))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    sub = img[y0:y1, x0:x1].astype(np.float32)
    if sig > 0:
        sub = cv2.GaussianBlur(sub, (0, 0), sig, borderType=cv2.BORDER_REPLICATE)
    c_x, c_y = cx - x0, cy - y0
    if flip:
        sub = sub[:, ::-1]
        c_x = sub.shape[1] - c_x - 1
    src = np.array([[c_x, c_y], [c_x, c_y + size / 2], [c_x + size / 2, c_y]], np.float32)
    dst = np.array([[out / 2, out / 2], [out / 2, out], [out, out / 2]], np.float32)
    trans = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(np.ascontiguousarray(sub), trans, (out, out), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _perspective_patch(img, vertices, K, scale, flip, out=256):
    """Rotate a virtual camera toward the hand before applying its square crop.

    A homography corrects the off-axis perspective shear without changing the
    optical centre. Image keypoints are subsequently mapped through its inverse.
    Perspective Crop Layers (arXiv:2011.13607), original NumPy/OpenCV code.
    """
    import cv2
    ray = vertices.mean(0).astype(np.float64)
    ray /= max(np.linalg.norm(ray), 1e-12)
    x = np.cross([0.0, 1.0, 0.0], ray)
    x /= max(np.linalg.norm(x), 1e-12)
    y = np.cross(ray, x)
    rotation = np.stack([x, y, ray])
    virtual = vertices @ rotation.T
    if virtual[:, 2].min() < 0.02:
        return None
    points = virtual[:, :2] / virtual[:, 2:]
    lo, hi = points.min(0), points.max(0)
    center = (lo + hi) / 2.0
    extent = scale * (hi - lo).max()
    focal = out / max(extent, 1e-12)
    if not (10 < focal < 20000):
        return None
    intrinsic = np.array([[focal, 0, out/2-focal*center[0]],
                          [0, focal, out/2-focal*center[1]], [0, 0, 1]])
    transform = intrinsic @ rotation @ np.linalg.inv(K)
    patch = cv2.warpPerspective(img, transform, (out, out), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if flip:
        patch = patch[:, ::-1]
    return np.ascontiguousarray(patch, dtype=np.float32), np.linalg.inv(transform), rotation


def _run(state, seg, params):
    """WiLoR on every existing hand: hp (F,2,15,3) full articulation in the hand's own frame, kp2d (F,2,21,2) px."""
    import torch
    F = seg.n
    hp = np.full((F, 2, 15, 3), np.nan)
    kp2d = np.full((F, 2, 21, 2), np.nan)
    go = np.full((F, 2, 3), np.nan)          # WiLoR's own global orientation (axis-angle, ACE camera frame, un-mirrored)
    k3w = np.full((F, 2, 21, 3), np.nan)     # WiLoR's metric 3D keypoints, wrist-relative (its own MANO shape)
    state["wilor_k3"] = k3w
    V = state["verts_cam"]
    ok = (state["exists"] > float(params.get("exists_min", 0.5))) & np.isfinite(V).all((2, 3)) & (V[..., 2] > 0.02).all(-1)
    if not ok.any() or _model() is None:
        return hp, kp2d, ok, go
    try:
        frames = seg.frames()
    except Exception as exc:
        sys.stderr.write(f"[wilor] frames unavailable ({exc!r}); image evidence skipped\n")
        return hp, kp2d, ok, go
    uv = seg.project(np.where(ok[..., None, None], V, 1.0))
    jobs, patches = [], []
    go_src = str(params.get("go_source", "rotate"))      # rotate | raw | plain (perspective crops only)
    go_plain = bool(params.get("perspective_crop", False)) and go_src == "plain"
    go_scale = float(params.get("go_box_scale", params.get("box_scale", 2.5)))
    for f, s in zip(*np.nonzero(ok)):
        if f >= len(frames):
            continue
        lo, hi = uv[f, s].min(0), uv[f, s].max(0)
        c = (lo + hi) / 2
        size = float(params.get("box_scale", 2.5)) * float((hi - lo).max())
        if not (8 <= size <= 4 * max(seg.W, seg.H)):
            continue
        inverse, vrot = None, None
        if params.get("perspective_crop", False):
            item = _perspective_patch(frames[f, :, :, ::-1], V[f, s], seg.K,
                                      float(params.get("box_scale", 2.5)), s == 0)
            if item is None:
                continue
            p, inverse, vrot = item
        else:
            p = _patch(frames[f, :, :, ::-1], c[0], c[1], size, s == 0)       # WiLoR expects BGR (it flips channels)
        if p is None:
            continue
        jobs.append((f, s, c, size, inverse, vrot, "main" if go_plain else "both")); patches.append(p)
        if go_plain:                       # WiLoR's GO is read from a plain (un-rectified) crop: it agrees far better
            gsz = go_scale * float((hi - lo).max())                              # with ACE's GO than the rectified one
            if 8 <= gsz <= 4 * max(seg.W, seg.H):
                q = _patch(frames[f, :, :, ::-1], c[0], c[1], gsz, s == 0)
                if q is not None:
                    jobs.append((f, s, c, gsz, None, None, "go")); patches.append(q)
    if not jobs:
        return hp, kp2d, ok, go
    m, dev = _model()
    f_crop = 5000.0 / 256.0
    amp = dev.type == "cuda" and bool(params.get("fp16", True))

    def forward(x_np):
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
            o = m(torch.from_numpy(x_np).to(dev, dtype=torch.float32))
            return (o["hand_pose"].float().cpu().numpy().reshape(-1, 15, 3), o["pred_keypoints_3d"].float().cpu().numpy(),
                    o["pred_cam"].float().cpu().numpy(), o["global_orient"].float().cpu().numpy().reshape(-1, 3))

    def safe_forward(x):
        try:
            return forward(x)
        except Exception as exc:
            failure = repr(exc)
        # Leave the exception scope before retrying. Its traceback otherwise
        # retains the failed forward's GPU tensors while smaller batches run.
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        if len(x) > 1:
            middle = len(x) // 2
            left, right = safe_forward(x[:middle]), safe_forward(x[middle:])
            return tuple(np.concatenate((a, b), axis=0) for a, b in zip(left, right))
        sys.stderr.write(f"[wilor] inference failed ({failure}); crop evidence skipped\n")
        return (np.full((1, 15, 3), np.nan), np.full((1, 21, 3), np.nan), np.full((1, 3), np.nan),
                np.full((1, 3), np.nan))

    bs = min(16, max(1, int(params.get("batch", 8))))
    for b0 in range(0, len(patches), bs):
        x = np.stack(patches[b0:b0 + bs])
        hpo, k3, cam, gor = safe_forward(x)
        for n, (f, s, c, size, inverse, vrot, kind) in enumerate(jobs[b0:b0 + bs]):
            sc, tx, ty = cam[n]
            X = k3[n].astype(np.float64) + np.array([tx, ty, 2 * f_crop / (sc + 1e-9)])
            u = f_crop * X[:, :2] / X[:, 2:3]
            h = hpo[n].astype(np.float64)
            g = gor[n].astype(np.float64)
            if s == 0:                                                       # un-mirror the left hand
                h[:, 1:] *= -1
                g[1:] *= -1
            if vrot is not None and go_src == "rotate" and np.isfinite(g).all():   # virtual camera -> ACE camera
                g = Rotation.from_matrix(vrot.T @ Rotation.from_rotvec(g).as_matrix()).as_rotvec()
            if kind != "main":
                go[f, s] = g
            if kind == "go":
                continue
            k3w[f, s] = k3[n] - k3[n][:1]
            if inverse is None:
                if s == 0:
                    u[:, 0] = -u[:, 0]
                kp2d[f, s] = c + u * size
            else:
                pixel = 128.0 + 256.0 * u
                if s == 0:
                    pixel[:, 0] = 255.0 - pixel[:, 0]
                homogeneous = np.c_[pixel, np.ones(len(pixel))] @ inverse.T
                kp2d[f, s] = homogeneous[:, :2] / homogeneous[:, 2:]
            hp[f, s] = h
    # Release temporary allocations once per segment. Releasing every crop
    # batch forces repeated driver allocation across concurrent GPU workers.
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return hp, kp2d, ok, go


def _smooth_quat(q, w, sig):
    q = q.copy()
    for t in range(1, len(q)):
        sgn = np.sign((q[t] * q[t - 1]).sum(-1, keepdims=True)); sgn[sgn == 0] = 1; q[t] *= sgn
    if sig > 0:
        num = gaussian_filter1d(q * w[:, None, None], sig, axis=0, mode="nearest")
        den = gaussian_filter1d(w, sig, mode="nearest")
        q = num / np.maximum(den, 1e-8)[:, None, None]
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)


def apply(state, seg, params):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _apply(state, seg, params)


def _apply(state, seg, params):
    alpha = float(params.get("alpha", 0.3))
    gate = float(params.get("gate", 0.25))
    sig = float(params.get("sig", 1.5))
    hpW, kp2d, ok, goW = _run(state, seg, params)
    state["wilor_go"] = goW                  # consumed by go_bandpass (ungated; it weights by agreement itself)
    F = seg.n
    # gate: WiLoR 2D vs ACE's own 2D anchors, median joint distance relative to the hand box
    d2 = state["joints_2d"].astype(np.float64)
    J = state["joints_cam"]
    uvJ = seg.project(np.where(np.isfinite(J), J, 1.0))
    ext = (np.nanmax(uvJ, 2) - np.nanmin(uvJ, 2)).max(-1)
    dis = np.nanmedian(np.linalg.norm(kp2d - d2, axis=-1), axis=-1) / np.maximum(ext, 1.0)
    good = np.isfinite(hpW).all((2, 3)) & np.isfinite(kp2d).all((2, 3)) & (dis < gate)
    # ungated image evidence for later presence checks (a5__border_presence); nothing below reads these
    state["_wilor_kp2d_all"] = kp2d
    state["_wilor_uv_at"] = uvJ
    sys.stderr.write(f"[wilor] coverage eligible={int(ok.sum())} "
                     f"image={int(np.isfinite(kp2d).all((2, 3)).sum())} "
                     f"accepted={int(good.sum())} go={int(np.isfinite(goW).all(-1).sum())}\n")
    state["wilor_2d"] = np.where(good[..., None, None], kp2d, np.nan)
    state["wilor_hp"] = hpW
    state["wilor_good"] = good
    state["_retained_wilor_2d"] = state["wilor_2d"].copy()
    if alpha <= 0:
        return state
    mean = _mano_mean()
    changed = False
    for s in (0, 1):
        g = good[:, s]
        if not g.any():
            continue
        w = g.astype(np.float64)
        h = np.where(g[:, None, None], hpW[:, s], 0.0)
        qW = Rotation.from_rotvec(h.reshape(-1, 3)).as_quat().reshape(F, 15, 4)
        qW = _smooth_quat(qW, w, sig)
        idx = np.nonzero(g)[0]
        hA = state["hp"][idx, s]
        okA = np.isfinite(hA).all((1, 2, 3))
        idx = idx[okA]
        if not len(idx):
            continue
        RA = Rotation.from_rotvec((Rotation.from_matrix(state["hp"][idx, s].reshape(-1, 3, 3)).as_rotvec().reshape(-1, 15, 3)
                                   + mean[s]).reshape(-1, 3))
        RW = Rotation.from_quat(qW[idx].reshape(-1, 4))
        Rb = RA * Rotation.from_rotvec(alpha * (RA.inv() * RW).as_rotvec())
        hp_new = Rb.as_rotvec().reshape(-1, 15, 3) - mean[s]
        state["hp"][idx, s] = Rotation.from_rotvec(hp_new.reshape(-1, 3)).as_matrix().reshape(-1, 15, 3, 3)
        changed = True
    return seg.mano(state) if changed else state
