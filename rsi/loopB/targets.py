"""Teacher targets of one round: the teacher harness H_k* replayed on θ_{k-1}'s raw captures of the training split,
assembled per frame for every training sequence.

  python -m loopB.targets --tree TREE --cache captures/<node>/hot3d_train --view data/views/hot3d_train --name R1 [--jobs 32]
-> loopB/cache/<sid>/teacher_<name>.npz  {exists (n,2), joints_cam, transl, go, hp, betas, joints_2d}; frames outside any
   81-frame segment (the dropped tail) have exists = 0 and NaN poses.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loopB import common as C  # noqa: E402

KEYS = dict(exists=(2,), joints_cam=(2, 21, 3), transl=(2, 3), rot=(2, 3, 3), hp=(2, 15, 3, 3), betas=(2, 10), joints_2d=(2, 21, 2))
# Pseudo-label quality weight per hand-frame (parameters fixed before use). Literature: select / weight
# pseudo-labels by agreement with an independent model rather than by the teacher's own confidence, and drop temporal
# outliers (MINT's outlier removal; flow-consistency filtering). weight = w_temp * w_agree, 0 where the teacher claims no hand.
#   w_temp  = exp(-(a / 15 mm)^2), a = the teacher's wrist acceleration |x[t+1] - 2 x[t] + x[t-1]| (normal jitter ~3 mm)
#   w_agree = exp(-(d / 0.15)^2), d = median 2D distance over the 16 non-tip joints, teacher vs WiLoR's image-space
#             keypoints, / teacher hand-box diagonal; 0 if d > 0.5. Only where WiLoR (wilor_final + its YOLO detector,
#             conf 0.3) has a box for that hand: where it does not (occluded or out-of-view hands, exactly where the teacher
#             is most useful) w_agree = 1, so those frames are not dropped.
#   Reference = WiLoR's crop-camera 2D mapped to the full image (data/wilor_ref2d.py), NOT its 3D projected through K:
#   on HOT3D that projection is ~23 px median off the GT (weak-perspective depth at fx = 208), so the first reference
#   (data/hot3d_nogt_wilor) disagreed with every teacher frame and the weights carried no signal (median weight 0.06;
#   weight-0 and weight-0.7+ frames had the same teacher error). Checked on 4 HOT3D train recordings with GT
#   (30k hand-frames): WiLoR image-2D error 5.5 px median (teacher 4.1); d vs the teacher's true 2D error Spearman 0.63;
#   teacher error 2.7 px at d < 0.05, 14.3 px at 0.2-0.5, 34.8 px at d > 0.5 (1.1% of frames, dropped).
WILOR_REF = C.S / "data" / "hot3d_nogt_wilor2d"    # per-segment WiLoR image-space 2D, where computed
BODY = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]     # joints without the 5 tips (tip vertex ids differ)
TAU_ACC, SIGMA, DROP = 0.015, 0.15, 0.5


def _uv(J, K):
    z = np.maximum(J[..., 2], 1e-4)
    return np.stack([K[0, 0] * J[..., 0] / z + K[0, 2], K[1, 1] * J[..., 1] / z + K[1, 2]], -1)


def quality_weight(out: dict, d: dict) -> tuple[np.ndarray, dict]:
    n = int(d["frames"]); K = np.asarray(d["K"], float)
    J = out["joints_cam"].astype(np.float64)
    ex = out["exists"] > 0.5
    wr = J[:, :, 0]
    acc = np.zeros((n, 2))
    if n >= 3:
        acc[1:-1] = np.linalg.norm(wr[2:] - 2 * wr[1:-1] + wr[:-2], axis=-1)
    w_temp = np.exp(-(np.nan_to_num(acc, nan=0.0) / TAU_ACC) ** 2)
    w_agree = np.ones((n, 2)); has_ref = np.zeros((n, 2), bool); dist = np.full((n, 2), np.nan)
    ref = np.full((n, 2, 21, 2), np.nan)
    for sg in d["segments"]:
        f = WILOR_REF / f"{sg['name']}.npz"
        if f.is_file():
            ref[sg["start"]:sg["end"]] = np.load(f)["kp2d"]
    uv_t = _uv(J, K)
    ok = ex & np.isfinite(J).all((-1, -2)) & np.isfinite(ref).all((-1, -2))
    if ok.any():
        span = np.linalg.norm(np.nanmax(uv_t, -2) - np.nanmin(uv_t, -2), axis=-1).clip(min=20.0)
        dd = np.median(np.linalg.norm(uv_t[:, :, BODY] - ref[:, :, BODY], axis=-1), -1) / span
        dist[ok] = dd[ok]; has_ref = ok
        w_agree[ok] = np.exp(-(dd[ok] / SIGMA) ** 2)
        w_agree[ok & (dd > DROP)] = 0.0
    w = (w_temp * w_agree).astype(np.float32); w[~ex] = 0.0
    st = dict(teacher_hand_frames=int(ex.sum()), with_wilor_ref=int(has_ref.sum()),
              dropped_by_agreement=int((has_ref & (np.nan_to_num(dist) > DROP)).sum()),
              mean_weight=float(w[ex].mean()) if ex.any() else None)
    return w, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True, type=Path)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--view", required=True, type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--pred", type=Path, default=None, help="harness output dir (default loopB/teacher_pred/<name>)")
    a = ap.parse_args()
    __import__("os").umask(0o022)          # new trees / prediction dirs must be readable inside the sandbox
    pred = a.pred or (C.S / "loopB" / "teacher_pred" / a.name)
    # the teacher tree is agent-written: run it in the sandbox, exactly like the gate does
    sys.path.insert(0, str(C.S / "loopA"))
    import sandbox
    view = a.view.resolve()
    ro = [str(a.tree.resolve()), str(a.cache.resolve()), str(view / "segments"), str(view / "videos")]
    pred.mkdir(parents=True, exist_ok=True)
    import os
    os.chown(pred, 23001, 23001)
    r = sandbox.run([C.rsi_paths.PY, "-m", "harness.run", "--tree", str(a.tree.resolve()),
                     "--cache", str(a.cache.resolve()), "--view", str(view), "--out", str(pred), "--jobs", str(a.jobs)],
                    ro=list(dict.fromkeys(ro)), rw=[str(pred)], cwd=str(C.S), env=dict(PYTHONPATH=str(C.S)),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, net=False)     # no network at run time
    print((r.stdout or "")[-2000:])
    if r.returncode:
        sys.exit(f"teacher replay failed (rc {r.returncode})")
    tot = {}
    for sj in sorted((view / "segments").glob("*.json")):
        d = json.loads(sj.read_text()); n = int(d["frames"])
        out = {k: np.full((n,) + s, np.nan, np.float32) for k, s in KEYS.items()}
        out["exists"][:] = 0.0
        for sg in d["segments"]:
            p = np.load(pred / f"{sg['name']}.npz"); sl = slice(sg["start"], sg["end"])
            for k in KEYS:
                out[k][sl] = p[k]
        out["weight"], qs = quality_weight(out, d)
        for k_, v_ in qs.items():
            tot[k_] = tot.get(k_, 0) + (v_ or 0) * (1 if k_ != "mean_weight" else qs["teacher_hand_frames"])
        dst = C.CACHE / sj.stem; dst.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst / f"teacher_{a.name}.npz", **out)
    if tot.get("teacher_hand_frames"):
        tot["mean_weight"] = tot["mean_weight"] / tot["teacher_hand_frames"]
    print(f"teacher_{a.name}: {len(list((view / 'segments').glob('*.json')))} sequences; pseudo-label quality {json.dumps(tot)}")


if __name__ == "__main__":
    main()
