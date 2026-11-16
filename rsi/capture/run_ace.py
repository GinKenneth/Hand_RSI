"""ACE-Ego-Hand (official inference code: rsi_paths.ACE_OFFICIAL, commit 59da83b) on the ACE-protocol segments of a
dataset view (data/build_views.py).

Benchmark setting: K-given model with the segment's ground-truth intrinsics, native 480 x 480 encode, predict_video()
in "tiled" mode. --context segment (default): every 81-frame test segment is its own clip (the paper "decodes the fixed
81-frame test segments shared with all baselines"; 81 frames = 21 latent frames = one training window).
--context whole: the whole sequence is encoded once and tiled 22-latent windows anchored at each 81-frame segment
(predict_video's own long-video path), then sliced per segment.
Output: one npz per segment with the raw, UNGATED predict_video() arrays (DUMP_KEYS) for ace_eval.adapters.ace_dump.

  python capture/run_ace.py --data data/views/hot3d_test --video-dir data/views/hot3d_test/videos \
      --out captures/00/hot3d_test --ckpt $ACE_CKPT --gpu 0 [--shard 0/4] [--context whole]   (or capture/capture.sh)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import rsi_paths  # noqa: E402
ACE = rsi_paths.ACE_OFFICIAL
sys.path.insert(0, str(ACE)); sys.path.insert(0, str(ACE / 'third_party'))
from ace_ego_hand.inference import DUMP_KEYS, predict_video  # noqa: E402
from ace_ego_hand.models.geodit_model import GeoDitModel  # noqa: E402
from ace_ego_hand.video_vae import encode, load_vae  # noqa: E402


def read_frames(path: str, n: int, W: int, H: int) -> np.ndarray:
    cap = cv2.VideoCapture(path); fr = []
    while len(fr) < n:
        ok, f = cap.read()
        if not ok:
            break
        if f.shape[1] != W or f.shape[0] != H:
            f = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
        fr.append(f[:, :, ::-1])                                     # BGR -> RGB, as decode_video
    cap.release()
    if len(fr) < n:
        raise RuntimeError(f'{path}: decoded {len(fr)} < {n}')
    return np.stack(fr)


def to_pixels(a: np.ndarray) -> torch.Tensor:
    """uint8 (F,H,W,3) -> (F,3,H,W) float32 in [-1, 1], padded to 4k+1 frames. Adapted from ACE-Ego-Hand's decode_video
    and infer_video.py padding (MIT licence, Copyright (c) 2026 ACE Robotics)."""
    n = len(a); n_enc = max(5, 4 * ((n + 2) // 4) + 1)
    x = torch.from_numpy(a.astype(np.float32) / 255.0 * 2.0 - 1.0).permute(0, 3, 1, 2).contiguous()
    if n_enc > n:
        x = torch.cat([x, x[-1:].repeat(n_enc - n, 1, 1, 1)])
    return x


def _save(path: Path, **arrays) -> None:
    """temp file + rename: two workers on one segment never leave a torn npz"""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--ckpt', required=True); ap.add_argument('--opt', default=str(ACE / 'options' / 'ace_ego_hand_k.yml'))
    ap.add_argument('--gpu', type=int, default=0); ap.add_argument('--shard', default='0/1')
    ap.add_argument('--context', choices=('segment', 'whole'), default='segment')
    ap.add_argument('--tap', type=int, default=15); ap.add_argument('--tile-w', type=int, default=22)
    ap.add_argument('--reverse', action='store_true', help='walk this shard last-to-first (a 2nd worker)')
    ap.add_argument('--video-dir', type=Path, default=None, help='read <video-dir>/<file name> instead of the json path')
    ap.add_argument('--encode-w', type=int, default=0,
                    help='0: native size (official rule, never upscales). N: scale the width to N '
                         '(also upscales), K rescaled with the frames')
    a = ap.parse_args()
    a.data, a.out = a.data.resolve(), a.out.resolve()      # os.chdir(ACE) below
    if a.video_dir:
        a.video_dir = a.video_dir.resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    i, k = map(int, a.shard.split('/'))
    seqs = sorted((a.data / 'segments').glob('*.json'))[i::k]
    if a.reverse:
        seqs = seqs[::-1]
    os.chdir(ACE)                                       # option paths (ckpt/..., third_party/...) are repo-relative
    device = torch.device('cuda', a.gpu); torch.cuda.set_device(device)
    model = GeoDitModel(yaml.safe_load(open(a.opt)), device)
    model.load_inference(str(Path(a.ckpt).resolve()))
    vae = load_vae(device)
    for sj in seqs:
        d = json.loads(sj.read_text())
        todo = [s for s in d['segments'] if not (a.out / f"{s['name']}.npz").is_file()]
        if not todo:
            continue
        K = np.asarray(d['K']); W, H = int(d['W']), int(d['H'])
        EW, EH = W, H
        if a.encode_w:
            sc = a.encode_w / float(W); EW, EH = (max(32, int(round(v * sc / 32)) * 32) for v in (W, H))
        sx, sy = EW / W, EH / H
        intr = {'fx': float(K[0, 0]) * sx, 'fy': float(K[1, 1]) * sy, 'cx': float(K[0, 2]) * sx, 'cy': float(K[1, 2]) * sy,
                'image_width': EW, 'image_height': EH}
        vpath = str(a.video_dir / Path(d['video']).name) if a.video_dir else d['video']
        frames = read_frames(vpath, max(s['end'] for s in d['segments']), EW, EH)
        if a.context == 'whole':
            ctrl = encode(vae, to_pixels(frames), device).float()
            pred = predict_video(model, a.tap, ctrl, intr, 'tiled', device, tile_w=a.tile_w)
            for s in todo:
                _save(a.out / f"{s['name']}.npz",
                                    **{kk: np.asarray(pred[kk][s['start']:s['end']], np.float32) for kk in DUMP_KEYS})
        else:
            for s in todo:
                n = s['end'] - s['start']
                ctrl = encode(vae, to_pixels(frames[s['start']:s['end']]), device).float()
                pred = predict_video(model, a.tap, ctrl, intr, 'tiled', device, tile_w=a.tile_w)
                _save(a.out / f"{s['name']}.npz", **{kk: np.asarray(pred[kk][:n], np.float32) for kk in DUMP_KEYS})
        print(f"done {d['sequence_id']}: {len(todo)} segments", flush=True)
        del frames
    print('ACE_RUN_DONE', flush=True)


if __name__ == '__main__':
    main()
