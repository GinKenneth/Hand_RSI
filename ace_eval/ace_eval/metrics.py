"""Per-segment scoring for the ViDiHand / ACE-Ego-Hand coverage-aware protocol.

Re-implementation of ACE-Ego-Hand (arXiv 2608.20308) Appendix A.1 from its text;
the authors' evaluator is not public. Equation numbers (S1)-(S11) refer to that
appendix. Every choice the text leaves open is a field of `Config` and is listed
in README.md; the defaults were fixed before any reproduction run.

Conventions (see README.md for the file format):
  slot 0 = left hand, slot 1 = right hand; 21 joints in OpenPose order
  (wrist, thumb x4, index x4, middle x4, ring x4, pinky x4); camera frame,
  metres; pixels in the evaluated image (W x H) with intrinsics K.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Config:
    z_min: float = 0.01              # (S1) on-screen depth threshold, metres
    exist_thr: float = 0.5           # active iff score > exist_thr
    box_dilate: float = 1.10         # GT box width/height scale about its centre
    box_clip: bool = False           # clip projected boxes to the image
    pa_scale: bool = True            # Procrustes with scale (S5)
    epe_pool: str = "joint"          # EPE2D-p averaged over on-screen joints ("joint") or hands ("hand")
    jitter_norm: str = "joint_mean"  # per-frame value: mean of per-joint norms, or "frobenius"
    jitter_pool: str = "frame"       # pool second differences over all frames ("frame") or average runs ("run")
    jitter_min_run: int = 3
    oos_missing: str = "canonical"   # pred slot is NaN in the GT-gated pass: "canonical" or "skip"
    canonical_flat: bool = False     # (S8) "zero articulation": MANO mean pose (False) or flat hand (True)
    # Detection rule for hands without ground truth: a prediction on a hand slot whose GT is not annotated in that frame
    # (valid = False) is left out of detection, neither a TP nor a FP. HOT3D has untracked stretches where the hand is
    # plainly visible (one test recording: 44 % of its hand slots); every such "FP" we inspected was a real hand.
    # Off-screen hands keep their GT (valid = True), so a hand pushed into the frame by a wrong depth is still a FP.
    # Only FAcc and F1 depend on it: such a prediction can never be a TP, so Recall and the pose columns are unaffected.


# ── geometry ────────────────────────────────────────────────────────────

def project(X: np.ndarray, K: np.ndarray):
    """Pinhole projection of (..., 3) camera points -> (..., 2) pixels and (...) depth."""
    z = X[..., 2]
    q = X @ K.T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = q[..., :2] / q[..., 2:3]
    return uv, z


def joints_in_frame(J: np.ndarray, K: np.ndarray, W: int, H: int, z_min: float) -> np.ndarray:
    """(21,) bool: joint projects inside [0,W) x [0,H) at depth > z_min (S1)."""
    uv, z = project(J, K)
    return (z > z_min) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)


def proj_box(P: np.ndarray, K: np.ndarray, W: int, H: int, cfg: Config):
    """Bounding box (x0, y0, x1, y1) of the projected points in front of the camera, or None."""
    uv, z = project(P, K)
    uv = uv[z > cfg.z_min]
    if len(uv) == 0:
        return None
    b = np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()])
    if cfg.box_clip:
        b = np.array([max(b[0], 0), max(b[1], 0), min(b[2], W), min(b[3], H)])
        if b[2] <= b[0] or b[3] <= b[1]:
            return None
    return b


def dilate(b: np.ndarray, s: float) -> np.ndarray:
    c = (b[:2] + b[2:]) / 2
    h = (b[2:] - b[:2]) / 2 * s
    return np.concatenate([c - h, c + h])


def iou(a: np.ndarray, b: np.ndarray) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def procrustes(S1: np.ndarray, S2: np.ndarray, scale: bool = True) -> np.ndarray:
    """Similarity (or rigid) transform of S1 onto S2, both (N, 3); returns aligned S1."""
    mu1, mu2 = S1.mean(0), S2.mean(0)
    X1, X2 = S1 - mu1, S2 - mu2
    U, _, Vt = np.linalg.svd(X1.T @ X2)
    Z = np.eye(3)
    Z[-1, -1] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ Z @ U.T
    s = np.trace(R @ X1.T @ X2) / (X1 ** 2).sum() if scale else 1.0
    return s * X1 @ R.T + mu2


def geodesic_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    c = (np.trace(R1.T @ R2) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def wrist_mpjpe(Jp: np.ndarray, Jg: np.ndarray) -> float:
    """(S4)/(S9): mean joint error after subtracting each set's wrist, in the input unit."""
    return float(np.linalg.norm((Jp - Jp[0]) - (Jg - Jg[0]), axis=-1).mean())


# ── accumulator ─────────────────────────────────────────────────────────

@dataclass
class Acc:
    """Running sums for one benchmark; `add` merges another accumulator."""
    frames: int = 0
    frames_ok: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    pen: dict = field(default_factory=lambda: {k: 0.0 for k in ("mpjpe", "pa", "go", "ct")})
    matched: dict = field(default_factory=lambda: {k: 0.0 for k in ("mpjpe", "pa", "go", "ct", "epe")})
    n_go_missing: int = 0            # TP/FN hands without a usable rotation (excluded from GO)
    go_n_matched: int = 0
    epe_sum: float = 0.0             # penalised EPE2D numerator
    epe_n: float = 0.0               # its denominator (joints or hands, per cfg.epe_pool)
    epe_matched_sum: float = 0.0
    epe_matched_n: float = 0.0
    jit_sum: float = 0.0
    jit_n: int = 0
    jit_runs: list = field(default_factory=list)
    oos: dict = field(default_factory=lambda: {"iv_sum": 0.0, "iv_n": 0, "oos_sum": 0.0, "oos_n": 0, "missing": 0})

    def add(self, o: "Acc") -> None:
        for k in ("frames", "frames_ok", "tp", "fp", "fn", "n_go_missing", "go_n_matched", "epe_sum", "epe_n",
                  "epe_matched_sum", "epe_matched_n", "jit_sum", "jit_n"):
            setattr(self, k, getattr(self, k) + getattr(o, k))
        for d in ("pen", "matched", "oos"):
            for k in getattr(self, d):
                getattr(self, d)[k] += getattr(o, d)[k]
        self.jit_runs += o.jit_runs


# ── per-segment scoring ─────────────────────────────────────────────────

def _finite(*a) -> bool:
    return all(x is not None and np.all(np.isfinite(x)) for x in a)


def _match(gts: list, preds: list) -> tuple[list, int, list]:
    """Appendix A.1 'Detection': strictly-positive-overlap test on the matching side.

    gts / preds: dicts with 'side' and 'box' (GT box already dilated). Each prediction
    takes its highest-overlap GT hand over both sides; zero overlap or an opposite-side
    best is a FP; two same-side predictions on one GT hand -> higher overlap wins.
    Returns (tp pairs [(gi, pi)], n_fp, fn gt indices)."""
    best = {}
    n_fp = 0
    for pi, p in enumerate(preds):
        ious = [iou(p["box"], g["box"]) for g in gts]
        if not ious or max(ious) <= 0:
            n_fp += 1
            continue
        top = max(ious)
        same = [gi for gi, g in enumerate(gts) if g["side"] == p["side"] and ious[gi] == top]
        if not same:                      # best overlap falls on the opposite side
            n_fp += 1
            continue
        gi = same[0]
        if gi in best:
            n_fp += 1                     # one of the two same-side predictions loses
            if top > best[gi][0]:
                best[gi] = (top, pi)
        else:
            best[gi] = (top, pi)
    tp = [(gi, pi) for gi, (_, pi) in best.items()]
    fn = [gi for gi in range(len(gts)) if gi not in best]
    return tp, n_fp, fn


def score_segment(gt: dict, pred: dict, canon: np.ndarray, cfg: Config) -> Acc:
    """Score one segment. gt / pred follow README.md; canon is (2, 21, 3) canonical MANO
    joints (identity orientation, zero articulation, mean shape, tau = 0) per slot."""
    K = np.asarray(gt["K"], np.float64)
    W, H = int(gt["W"]), int(gt["H"])
    diag = float(np.hypot(W, H))
    F = gt["valid"].shape[0]
    a = Acc()
    a.frames = F
    g_has_verts = "verts_cam" in gt
    p_has_verts = "verts_cam" in pred
    runs = {0: [], 1: []}                 # per GT side: list of (frame, J~) for matched frames

    for f in range(F):
        gts, preds = [], []
        for s in (0, 1):
            Jg = gt["joints_cam"][f, s]
            if bool(gt["valid"][f, s]) and _finite(Jg):
                vis = joints_in_frame(Jg, K, W, H, cfg.z_min)
                if vis.any():
                    pts = gt["verts_cam"][f, s] if g_has_verts else Jg
                    box = proj_box(pts, K, W, H, cfg)
                    if box is not None:
                        gts.append({"side": s, "box": dilate(box, cfg.box_dilate), "vis": vis})
            Jp = pred["joints_cam"][f, s]
            if not bool(gt["valid"][f, s]):
                continue                                             # no GT annotated for this hand here: not scored
            if float(pred["exists"][f, s]) > cfg.exist_thr and _finite(Jp):
                if joints_in_frame(Jp, K, W, H, cfg.z_min).any():   # OOS predictions are not FPs
                    pts = pred["verts_cam"][f, s] if p_has_verts else Jp
                    box = proj_box(pts, K, W, H, cfg)
                    if box is not None:
                        preds.append({"side": s, "box": box})
        tp, n_fp, fn = _match(gts, preds)
        a.tp += len(tp)
        a.fp += n_fp
        a.fn += len(fn)
        a.frames_ok += int(n_fp == 0 and len(fn) == 0)

        for gi, pi in tp:
            s = gts[gi]["side"]
            vis = gts[gi]["vis"]
            Jg, Jp = gt["joints_cam"][f, s], pred["joints_cam"][f, s]
            e = {"mpjpe": wrist_mpjpe(Jp, Jg) * 1e3,
                 "pa": float(np.linalg.norm(procrustes(Jp, Jg, cfg.pa_scale) - Jg, axis=-1).mean()) * 1e3,
                 "ct": float(np.linalg.norm(pred["transl"][f, s] - gt["transl"][f, s]))}
            for k, v in e.items():
                a.pen[k] += v
                a.matched[k] += v
            Rp = pred["rot"][f, s] if "rot" in pred else None
            Rg = gt["rot"][f, s] if "rot" in gt else None
            if _finite(Rp, Rg):
                v = geodesic_deg(Rp, Rg)
                a.pen["go"] += v
                a.matched["go"] += v
                a.go_n_matched += 1
            else:
                a.n_go_missing += 1
            # (S6): the model's own 2D anchors; without them (K-free) re-project under the GT K
            p2 = pred["joints_2d"][f, s] if "joints_2d" in pred else None
            if not _finite(p2):
                p2 = project(Jp, K)[0]
            g2 = project(Jg, K)[0]
            err = np.linalg.norm(p2[vis] - g2[vis], axis=-1)
            if cfg.epe_pool == "joint":
                a.epe_sum += err.sum()
                a.epe_n += len(err)
            else:
                a.epe_sum += err.mean()
                a.epe_n += 1
            a.epe_matched_sum += err.mean()
            a.epe_matched_n += 1
            runs[s].append((f, (Jp - Jp[0] + pred["transl"][f, s]) * 1e3))

        for gi in fn:                     # (S8): canonical MANO hand at the camera origin
            s = gts[gi]["side"]
            vis = gts[gi]["vis"]
            Jg, Jc = gt["joints_cam"][f, s], canon[s]
            a.pen["mpjpe"] += wrist_mpjpe(Jc, Jg) * 1e3
            a.pen["pa"] += float(np.linalg.norm(procrustes(Jc, Jg, cfg.pa_scale) - Jg, axis=-1).mean()) * 1e3
            a.pen["ct"] += float(np.linalg.norm(gt["transl"][f, s]))
            if "rot" in gt and _finite(gt["rot"][f, s]):
                a.pen["go"] += geodesic_deg(np.eye(3), gt["rot"][f, s])
            else:
                a.n_go_missing += 1
            if cfg.epe_pool == "joint":
                a.epe_sum += diag * int(vis.sum())
                a.epe_n += int(vis.sum())
            else:
                a.epe_sum += diag
                a.epe_n += 1

    # Jitter: matched frames only, maximal consecutive runs, no run crosses the segment.
    for s in (0, 1):
        seq = runs[s]
        i = 0
        while i < len(seq):
            j = i
            while j + 1 < len(seq) and seq[j + 1][0] == seq[j][0] + 1:
                j += 1
            if j - i + 1 >= cfg.jitter_min_run:
                Jt = np.stack([x[1] for x in seq[i:j + 1]])          # (T, 21, 3) mm
                d2 = Jt[2:] - 2 * Jt[1:-1] + Jt[:-2]
                if cfg.jitter_norm == "joint_mean":
                    v = np.linalg.norm(d2, axis=-1).mean(-1)
                else:
                    v = np.linalg.norm(d2.reshape(len(d2), -1), axis=-1)
                a.jit_sum += float(v.sum())
                a.jit_n += len(v)
                a.jit_runs.append(float(v.mean()))
            i = j + 1

    # (S9)-(S10): second pass gated by the ground truth alone; no existence gate, no matching.
    for f in range(F):
        for s in (0, 1):
            Jg = gt["joints_cam"][f, s]
            if not (bool(gt["valid"][f, s]) and _finite(Jg)):
                continue
            Jp = pred["joints_cam"][f, s]
            if _finite(Jp):
                eps = wrist_mpjpe(Jp, Jg) * 1e3
            elif cfg.oos_missing == "canonical":
                eps = wrist_mpjpe(canon[s], Jg) * 1e3
                a.oos["missing"] += 1
            else:
                a.oos["missing"] += 1
                continue
            if joints_in_frame(Jg, K, W, H, cfg.z_min).any():
                a.oos["iv_sum"] += eps
                a.oos["iv_n"] += 1
            else:
                a.oos["oos_sum"] += eps
                a.oos["oos_n"] += 1
    return a


def summarize(a: Acc, cfg: Config) -> dict:
    """Table-1 metrics plus the unpenalised matched-only means and the raw counts."""
    div = lambda x, n: float(x / n) if n else float("nan")   # noqa: E731
    p = div(a.tp, a.tp + a.fp)
    r = div(a.tp, a.tp + a.fn)
    n_pen = a.tp + a.fn
    n_go = n_pen - a.n_go_missing
    jitter = div(a.jit_sum, a.jit_n) if cfg.jitter_pool == "frame" else (
        float(np.mean(a.jit_runs)) if a.jit_runs else float("nan"))
    o = a.oos
    return {
        "FAcc": div(a.frames_ok, a.frames),
        "Recall": r,
        "F1": 0.0 if a.tp == 0 and a.tp + a.fn > 0 else (div(2 * p * r, p + r) if p == p and r == r else float("nan")),
        "MPJPE-p": div(a.pen["mpjpe"], n_pen),
        "PA-p": div(a.pen["pa"], n_pen),
        "MPJPE+OOS": div(o["iv_sum"] + o["oos_sum"], o["iv_n"] + o["oos_n"]),
        "EPE2D-p": div(a.epe_sum, a.epe_n),
        "GO-p": div(a.pen["go"], n_go),
        "CT-p": div(a.pen["ct"], n_pen),
        "Jitter": jitter,
        "extra": {
            "Precision": p,
            "MPJPE_OOS": div(o["oos_sum"], o["oos_n"]),
            "MPJPE_IV": div(o["iv_sum"], o["iv_n"]),
            "MPJPE_matched": div(a.matched["mpjpe"], a.tp),
            "PA_matched": div(a.matched["pa"], a.tp),
            "EPE2D_matched_per_hand": div(a.epe_matched_sum, a.epe_matched_n),
            "GO_matched": div(a.matched["go"], a.go_n_matched),
            "CT_matched": div(a.matched["ct"], a.tp),
        },
        "counts": {"frames": a.frames, "TP": a.tp, "FP": a.fp, "FN": a.fn,
                   "hand_frames_IV": o["iv_n"], "hand_frames_OOS": o["oos_n"],
                   "oos_pass_missing_pred": o["missing"], "go_missing": a.n_go_missing,
                   "jitter_frames": a.jit_n, "jitter_runs": len(a.jit_runs)},
    }
