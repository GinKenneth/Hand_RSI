"""Diagnostic: WiLoR's crop-space 2D keypoints vs the GT projection, per sequence.

The crop-space 2D keypoints come from WiLoR's weak-perspective crop camera
(pred_cam) and do not depend on any full-image depth or focal assumption. They
are image evidence independent of ACE. A constant offset shared by WiLoR and ACE
on a sequence points at GT/image alignment; an offset only ACE has points at ACE.

  python -m ace_eval.baselines.diag_wilor2d --data HOT3D_ACE --seqs a,b --device cuda:0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .run_wilor import load_pipe, read_frames

BODY = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]   # OpenPose joints without the 5 tips (tip vertex ids differ)


@torch.no_grad()
def crop_2d(pipe, frames, conf=0.3, batch=64):
    """{(frame, slot): (21, 2) full-image px} from WiLoR's crop camera, best box per side."""
    from skimage.filters import gaussian
    from wilor_mini.utils import utils as wu
    best = {}
    for i in range(0, len(frames), 32):
        for j, r in enumerate(pipe.hand_detector(frames[i:i + 32], conf=conf, verbose=False)):
            for k in range(len(r.boxes)):
                key = (i + j, int(r.boxes.cls[k]))
                c = float(r.boxes.conf[k])
                if key not in best or c > best[key][0]:
                    best[key] = (c, r.boxes.xyxy[k].cpu().numpy())
    keys = sorted(best)
    out = {}
    f_crop = pipe.FOCAL_LENGTH / pipe.IMAGE_SIZE                       # crop focal in crop-size units
    for c0 in range(0, len(keys), batch):
        chunk = keys[c0:c0 + batch]
        patches, meta = [], []
        for f, s in chunk:
            box = best[(f, s)][1]
            center = (box[2:4] + box[0:2]) / 2.0
            size = (2.5 * (box[2:4] - box[0:2])).max()
            img = frames[f]
            ds = size / 256 / 2.0
            if ds > 1.1:
                img = gaussian(img, sigma=(ds - 1) / 2, channel_axis=2, preserve_range=True)
            patch, _ = wu.generate_image_patch_cv2(img, center[0], center[1], size, size, 256, 256, s == 0, 1.0, 0,
                                                   border_mode=cv2.BORDER_CONSTANT)
            patches.append(patch)
            meta.append((f, s, center, size))
        x = torch.from_numpy(np.stack(patches)).to(pipe.device, dtype=pipe.dtype)
        o = {k: v.cpu().float().numpy() for k, v in pipe.wilor_model(x).items()}
        for n, (f, s, center, size) in enumerate(meta):
            sc, tx, ty = o["pred_cam"][n]
            X = o["pred_keypoints_3d"][n] + np.array([tx, ty, 2 * f_crop / (sc + 1e-9)])
            uv = f_crop * X[:, :2] / X[:, 2:3]                        # crop units, centre 0, flipped crop for left
            if s == 0:
                uv[:, 0] = -uv[:, 0]
            out[(f, s)] = center + uv * size
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--seqs", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(argv)
    pipe = load_pipe(a.device)
    for seq in a.seqs.split(","):
        meta = json.loads((a.data / "segments" / f"{seq}.json").read_text())
        frames = read_frames(str(a.data / "videos" / f"{seq}.mp4"), None)
        kp = crop_2d(pipe, frames)
        K = np.asarray(meta["K"])
        d_all = []
        for sg in meta["segments"]:
            g = np.load(a.data / "gt" / f"{sg['name']}.npz")
            J = g["joints_cam"]
            uv = J[..., :2] / J[..., 2:3] * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
            for t in range(J.shape[0]):
                for s in (0, 1):
                    k = (sg["start"] + t, s)
                    if k not in kp or not g["valid"][t, s]:
                        continue
                    ok = (J[t, s, BODY, 2] > 0.01) & (uv[t, s, BODY] >= 0).all(-1) & (uv[t, s, BODY] < meta["W"]).all(-1)
                    d = kp[k][BODY] - uv[t, s, BODY]
                    if ok.sum() >= 8 and np.median(np.linalg.norm(d[ok], axis=-1)) < 25:   # same hand, not a swap
                        d_all.append(d[ok])
        d = np.concatenate(d_all)
        print(f"{seq}: WiLoR crop-2D vs GT, {len(d)} joints: mean offset ({d[:, 0].mean():+.2f}, {d[:, 1].mean():+.2f}) px, "
              f"median ({np.median(d[:, 0]):+.2f}, {np.median(d[:, 1]):+.2f}), mean |err| {np.linalg.norm(d, axis=-1).mean():.2f}",
              flush=True)


if __name__ == "__main__":
    main()
