"""WiLoR baseline on protocol segments -> prediction npz for ace_eval.evaluate.

  python -m ace_eval.baselines.run_wilor --data HOT3D_ACE/ --out PRED_WILOR/ --device cuda:0 [--seqs a,b] [--max-frames N]

ACE A.3: "WiLoR uses wilor_final.ckpt with its own YOLO detector", run without
retraining. We drive that checkpoint and detector through wilor_mini (pip
package wilor-mini, its weights in $WILOR_MINI_ROOT),
replicating WiLorHandPose3dEstimationPipeline.predict() step for step, batched.

Choices this runner makes (the paper does not state them):
  - detector confidence 0.3 (the wilor_mini default); per frame and side the most
    confident box is kept, since the evaluator has one slot per side; a kept box
    is an active hand (exists = 1);
  - crop -> full-image translation uses the clip's true focal length (fx), not
    WiLoR's default 5000 / 256 * max(W, H). The wilor_mini docstring asks for
    this when the camera is known, and a default-focal translation would be tens of
    metres off at 480 px;
  - joints and vertices come from WiLoR's MANO parameters through
    ace_eval.mano.forward (ACE's 21-joint convention) with flat_hand_mean=True,
    which reproduces WiLoR's own mesh exactly (see to_pred), and tau = WiLoR's
    full-image camera translation;
  - no 2D anchors are written: WiLoR's 2D keypoints are the projection of its 3D
    joints, so the evaluator re-projects joints_cam under the GT K.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

import os

PLUGIN = os.environ.get("WILOR_MINI_ROOT", "third_party/wilor_root")   # dir holding wilor_mini/


def load_pipe(device: str):
    sys.path.insert(0, PLUGIN)
    # WiLoR's MANO pickle imports chumpy, which still expects NumPy's old aliases (same shim as
    # the plugin's ace_ego_hand/wilor_finger_refine.py).
    for name, value in {"bool": np.bool_, "int": np.int_, "float": np.float64, "complex": np.complex128,
                        "object": np.object_, "unicode": np.str_, "str": np.str_}.items():
        if not hasattr(np, name):
            setattr(np, name, value)
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
    return WiLorHandPose3dEstimationPipeline(device=torch.device(device), dtype=torch.float32, verbose=False)


def read_frames(video: str, max_frames: int | None) -> list:
    cap = cv2.VideoCapture(video)
    out = []
    while True:
        ok, f = cap.read()
        if not ok or (max_frames and len(out) >= max_frames):
            break
        out.append(f)
    cap.release()
    return out


@torch.no_grad()
def run_sequence(pipe, frames: list, fx: float, color: str, conf: float, det_batch: int, net_batch: int) -> dict:
    from skimage.filters import gaussian
    from wilor_mini.utils import utils as wu
    if color == "rgb":
        frames = [np.ascontiguousarray(f[..., ::-1]) for f in frames]
    F = len(frames)
    best = {}                                            # (frame, slot) -> (conf, xyxy)
    for i in range(0, F, det_batch):
        for j, r in enumerate(pipe.hand_detector(frames[i:i + det_batch], conf=conf, verbose=False)):
            if len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)  # 1 = right, 0 = left = our slot index
            cf = r.boxes.conf.cpu().numpy()
            for k in range(len(cf)):
                key = (i + j, int(cls[k]))
                if key not in best or cf[k] > best[key][0]:
                    best[key] = (float(cf[k]), xyxy[k])
    keys = sorted(best)
    H, W = frames[0].shape[:2]
    img_size = np.array([W, H], np.float64)
    go = np.full((F, 2, 3), np.nan)
    hp = np.full((F, 2, 15, 3), np.nan)
    betas = np.full((F, 2, 10), np.nan)
    tau = np.full((F, 2, 3), np.nan)
    score = np.zeros((F, 2), np.float32)
    for c0 in range(0, len(keys), net_batch):
        chunk = keys[c0:c0 + net_batch]
        patches, meta = [], []
        for f, s in chunk:
            box = best[(f, s)][1]
            center = (box[2:4] + box[0:2]) / 2.0
            size = (2.5 * (box[2:4] - box[0:2])).max()   # rescale_factor 2.5, as predict()
            img = frames[f]
            ds = size / 256 / 2.0
            if ds > 1.1:
                img = gaussian(img, sigma=(ds - 1) / 2, channel_axis=2, preserve_range=True)
            patch, _ = wu.generate_image_patch_cv2(img, center[0], center[1], size, size, 256, 256,
                                                   s == 0, 1.0, 0, border_mode=cv2.BORDER_CONSTANT)
            patches.append(patch)
            meta.append((f, s, center, size))
        x = torch.from_numpy(np.stack(patches)).to(pipe.device, dtype=pipe.dtype)
        out = {k: v.cpu().float().numpy() for k, v in pipe.wilor_model(x).items()}
        for n, (f, s, center, size) in enumerate(meta):
            cam = out["pred_cam"][n:n + 1].copy()
            cam[:, 1] *= (2 * s - 1)
            g = out["global_orient"][n].reshape(3).copy()
            h = out["hand_pose"][n].reshape(15, 3).copy()
            if s == 0:                                   # mirror back to a left-hand MANO
                g[1:] *= -1
                h[:, 1:] *= -1
            go[f, s], hp[f, s], betas[f, s] = g, h, out["betas"][n]
            tau[f, s] = wu.cam_crop_to_full(cam, center[None], size, img_size[None], fx)[0]
            score[f, s] = 1.0
    return {"go": go, "hp": hp, "betas": betas, "tau": tau, "exists": score,
            "det_conf": np.array([[best.get((f, s), (0.0,))[0] for s in (0, 1)] for f in range(F)], np.float32)}


def to_pred(r: dict, models: dict) -> dict:
    """models: smplx MANO with flat_hand_mean=True. WiLoR's MANOLayer poses the hand from rotation
    matrices without adding MANO's mean hand pose, so its hand_pose is relative to the flat hand;
    with True our forward reproduces WiLoR's own pred_vertices exactly (checked: 0.00 mm,
    vs 81 mm with False). Left hands are WiLoR's right-hand mesh mirrored in x, as its pipeline does."""
    from ace_eval.mano import forward
    F = len(r["exists"])
    J = np.full((F, 2, 21, 3), np.nan, np.float32)
    V = np.full((F, 2, 778, 3), np.nan, np.float32)
    rot = np.full((F, 2, 3, 3), np.nan, np.float32)
    for s in (0, 1):
        if s == 1:
            J[:, s], V[:, s] = forward(models, 1, r["go"][:, s], r["hp"][:, s], r["betas"][:, s])
        else:                                            # undo the left-hand mirroring, pose a right hand, mirror
            g, h = r["go"][:, 0].copy(), r["hp"][:, 0].copy()
            g[..., 1:] *= -1
            h[..., 1:] *= -1
            J[:, s], V[:, s] = forward(models, 1, g, h, r["betas"][:, 0])
            J[:, s, :, 0] *= -1
            V[:, s, :, 0] *= -1
        ok = np.isfinite(r["go"][:, s]).all(-1)
        if ok.any():
            rot[ok, s] = Rotation.from_rotvec(r["go"][ok, s]).as_matrix()
    tau = r["tau"].astype(np.float32)
    return {"exists": r["exists"], "joints_cam": J + tau[:, :, None], "verts_cam": V + tau[:, :, None],
            "transl": tau, "rot": rot, "det_conf": r["det_conf"]}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path, help="dir with segments/<seq>.json and videos/")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seqs", default="", help="comma-separated subset of sequence ids")
    ap.add_argument("--color", choices=("bgr", "rgb"), default="bgr", help="channel order fed to wilor_mini")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--det-batch", type=int, default=32)
    ap.add_argument("--net-batch", type=int, default=64)
    ap.add_argument("--max-frames", type=int, default=0, help="debug: only the first N frames")
    ap.add_argument("--raw", action="store_true",
                    help="only run the network: write <seq>.raw.npz (MANO params per frame), no MANO / segments")
    ap.add_argument("--from-raw", type=Path, help="build segment predictions from a dir of <seq>.raw.npz")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    seq_files = sorted((a.data / "segments").glob("*.json"))
    if a.seqs:
        keep = set(a.seqs.split(","))
        seq_files = [p for p in seq_files if p.stem in keep]
    pipe = None if a.from_raw else load_pipe(a.device)
    models = None
    if not a.raw:
        from ace_eval.mano import load_models
        models = load_models(flat_hand_mean=True)      # see to_pred()
    for p in seq_files:
        meta = json.loads(p.read_text())
        t0 = time.time()
        if a.raw:
            if (a.out / f"{p.stem}.raw.npz").exists():
                continue
        else:
            todo = [sg for sg in meta["segments"] if not (a.out / f"{sg['name']}.npz").exists()]
            if not todo:
                continue
        if a.from_raw:
            r = dict(np.load(a.from_raw / f"{p.stem}.raw.npz"))
            n_frames = len(r["exists"])
        else:
            video = a.data / "videos" / Path(meta["video"]).name       # the data dir may have moved
            frames = read_frames(str(video if video.exists() else meta["video"]), a.max_frames or None)
            n_frames = len(frames)
            r = run_sequence(pipe, frames, float(meta["K"][0][0]), a.color, a.conf, a.det_batch, a.net_batch)
            del frames
            if a.raw:
                np.savez_compressed(a.out / f"{p.stem}.raw.npz", **r)
                print(f"{p.stem}: {n_frames} frames, {int(r['exists'].sum())} hands, {time.time() - t0:.0f}s", flush=True)
                continue
        pred = to_pred(r, models)
        n = 0
        for sg in todo:
            if sg["end"] > n_frames:
                continue
            sl = slice(sg["start"], sg["end"])
            np.savez_compressed(a.out / f"{sg['name']}.npz", **{k: v[sl] for k, v in pred.items()})
            n += 1
        print(f"{p.stem}: {n_frames} frames, {n} segments, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
