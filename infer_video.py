"""HandRSI on one video: the ACE-Ego-Hand network θ (K-given) followed by the evolved harness, segment by segment.

  python infer_video.py --video clip.mp4 --intrinsics fx,fy,cx,cy --ckpt checkpoints/handrsi_r3 --out results/clip
      [--camera cam.json] [--harness rsi/harness/trees/H5] [--encode_w 832] [--device cuda:0]

The video is cut into 81-frame segments (the ACE protocol's test segments; one training window). Each segment is
encoded and decoded on its own, and the harness post-processes each segment's raw network outputs on their own,
exactly as in our evaluation. A tail shorter than 81 frames comes from one more segment that ends at the last frame.

--ckpt     a directory with ace_ego_hand_k.pt + ace_ego_hand_k.yml (our θ), or "release" for ACE-Ego-Hand's own
           K-given checkpoint (rsi_paths.ACE_CKPT): with "--harness empty" that is the ACE baseline (00).
--harness  a harness tree (chain.json + modules/); "empty" = the network alone.
Camera as in ACE-Ego-Hand's infer_video.py: --camera cam.json or --intrinsics fx,fy,cx,cy (pinhole, in the video's
pixels; rescaled with the frames). Frames wider than --encode_w are downscaled; narrower ones keep their size.

Output <out>/<video stem>.npz, per frame (F) and hand (slot 0 = left, 1 = right), camera frame, metres / pixels of
the encoded frame size (W, H):
  exists (F,2)  joints_cam (F,2,21,3)  joints_2d (F,2,21,2)  verts_cam (F,2,778,3)  transl (F,2,3)  rot (F,2,3,3)
  hand_pose (F,2,15,3,3)  betas (F,2,10)  K (3,3)  W  H
A hand is predicted in a frame when exists > 0.5 (the evaluator also requires it to project into the image).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'rsi'))
import rsi_paths  # noqa: E402
from harness.core import Segment, load_chain, run_segment  # noqa: E402

SEG = 81              # protocol segment = one training window (21 latent frames)
GRID = 32             # VAE grid
TAP, TILE_W = 15, 22  # predict_video settings of the evaluation (capture/run_ace.py)


def encode_size(w: int, h: int, max_w: int | None) -> tuple[int, int]:
    """ACE-Ego-Hand's rule: keep the aspect, downscale to max_w (never upscale), snap both axes to the VAE grid.
    Adapted from ACE-Ego-Hand's infer_video.py (_encode_size), MIT licence, Copyright (c) 2026 ACE Robotics."""
    tw = min(w, max_w) if max_w else w
    s = tw / float(w)
    return max(GRID, int(round(w * s / GRID)) * GRID), max(GRID, int(round(h * s / GRID)) * GRID)


def segments(n: int) -> list[tuple[int, int, int]]:
    """(start, end, keep_from): non-overlapping 81-frame segments, plus one ending at the last frame for the tail."""
    if n <= SEG:
        return [(0, n, 0)]
    out = [(s, s + SEG, 0) for s in range(0, n - SEG + 1, SEG)]
    r = n - out[-1][1]
    if r:
        out.append((n - SEG, n, SEG - r))
    return out


def read_frames(video: str, n: int, W: int, H: int) -> np.ndarray:
    """(n, H, W, 3) uint8 RGB, decoded from the start in order (no seeking: frame seeks are unreliable in some codecs)
    and resized to the encode size. The whole clip is held in memory, as in ACE-Ego-Hand's infer_video.py."""
    cap = cv2.VideoCapture(video)
    out = []
    while len(out) < n:
        ok, f = cap.read()
        if not ok:
            break
        if f.shape[1] != W or f.shape[0] != H:
            f = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
        out.append(f[:, :, ::-1])
    cap.release()
    if len(out) < n:
        raise RuntimeError(f'{video}: decoded {len(out)} < {n} frames')
    return np.ascontiguousarray(np.stack(out))


def to_pixels(a: np.ndarray):
    """uint8 (F,H,W,3) -> (F,3,H,W) float32 in [-1, 1], padded to 4k+1 frames. Adapted from ACE-Ego-Hand's decode_video
    and infer_video.py padding (MIT licence, Copyright (c) 2026 ACE Robotics)."""
    import torch
    n = len(a)
    n_enc = max(5, 4 * ((n + 2) // 4) + 1)
    x = torch.from_numpy(a.astype(np.float32) / 255.0 * 2.0 - 1.0).permute(0, 3, 1, 2).contiguous()
    if n_enc > n:
        x = torch.cat([x, x[-1:].repeat(n_enc - n, 1, 1, 1)])
    return x


def load_opt(path: Path, official: Path) -> dict:
    """The checkpoint's options; a path in them that does not exist on this machine (our checkpoints were saved with
    absolute paths) falls back to the value of the official options file, which is relative to the ACE-Ego-Hand repo."""
    import yaml
    opt, ref = yaml.safe_load(open(path)), yaml.safe_load(open(official))

    def fix(d, r):
        for k, v in d.items():
            if isinstance(v, dict) and isinstance(r.get(k), dict):
                fix(v, r[k])
            elif isinstance(v, str) and v.startswith('/') and not os.path.exists(v) and k in r:
                d[k] = r[k]
    fix(opt, ref)
    return opt


def camera(a, src_w: int, src_h: int, enc_w: int, enc_h: int) -> dict:
    if bool(a.camera) == bool(a.intrinsics):
        raise SystemExit('pass exactly one of --camera cam.json or --intrinsics fx,fy,cx,cy (the K-given model needs K)')
    if a.camera:
        cam = json.load(open(a.camera))
        k = cam['frames'][0]['intrinsics']
        fx, fy, cx, cy = (float(k[x]) for x in ('fx', 'fy', 'cx', 'cy'))
        cal_w, cal_h = float(cam.get('image_width', src_w)), float(cam.get('image_height', src_h))
    else:
        fx, fy, cx, cy = (float(v) for v in a.intrinsics.split(','))
        cal_w, cal_h = float(src_w), float(src_h)
    sx, sy = enc_w / cal_w, enc_h / cal_h
    return {'fx': fx * sx, 'fy': fy * sy, 'cx': cx * sx, 'cy': cy * sy, 'image_width': enc_w, 'image_height': enc_h}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--video', required=True)
    ap.add_argument('--camera', default='')
    ap.add_argument('--intrinsics', default='')
    ap.add_argument('--ckpt', required=True, help='directory with ace_ego_hand_k.pt / .yml, or "release"')
    ap.add_argument('--harness', default=str(HERE / 'rsi' / 'harness' / 'trees' / 'H5'))
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--encode_w', type=int, default=832)
    ap.add_argument('--max_frames', type=int, default=0)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    a.out = a.out.resolve()
    video = str(Path(a.video).resolve())
    tree = HERE / 'rsi' / 'harness' / 'trees' / 'empty' if a.harness == 'empty' else Path(a.harness).resolve()
    os.environ.setdefault('WILOR_MINI_ROOT', rsi_paths.WILOR_ROOT)          # image-evidence modules (WiLoR)
    os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')

    cap = cv2.VideoCapture(video)
    src_w, src_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n <= 0:
        raise SystemExit(f'{video}: no frames')
    n = min(n, a.max_frames) if a.max_frames else n
    W, H = encode_size(src_w, src_h, a.encode_w or None)
    intr = camera(a, src_w, src_h, W, H)
    K = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1.0]])
    print(f'[video] {video} {src_w}x{src_h}, {n} frames -> {W}x{H}, {len(segments(n))} segments; harness {tree}', flush=True)

    ace = rsi_paths.ACE_OFFICIAL
    sys.path[:0] = [str(ace), str(ace / 'third_party')]
    import torch
    from ace_ego_hand.inference import DUMP_KEYS, predict_video
    from ace_ego_hand.models.geodit_model import GeoDitModel
    from ace_ego_hand.video_vae import encode, load_vae
    if a.ckpt == 'release':
        ckpt, opt_path = Path(rsi_paths.ACE_CKPT).resolve(), ace / 'options' / 'ace_ego_hand_k.yml'
    else:
        ckpt, opt_path = Path(a.ckpt).resolve() / 'ace_ego_hand_k.pt', Path(a.ckpt).resolve() / 'ace_ego_hand_k.yml'
    opt = load_opt(opt_path, ace / 'options' / 'ace_ego_hand_k.yml')
    chain = load_chain(tree)
    cwd = os.getcwd()
    os.chdir(ace)                                     # ACE-Ego-Hand's option paths are relative to its repo
    device = torch.device(a.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    model = GeoDitModel(opt, device)
    model.load_inference(str(ckpt))
    vae = load_vae(device)
    os.chdir(cwd)

    allf = read_frames(video, n, W, H)
    out = None
    t0 = time.time()
    for start, end, keep in segments(n):
        frames = allf[start:end]
        with torch.no_grad():
            ctrl = encode(vae, to_pixels(frames), device).float()
            pred = predict_video(model, TAP, ctrl, intr, 'tiled', device, tile_w=TILE_W)
        raw = {k: np.asarray(pred[k][:end - start], np.float32) for k in DUMP_KEYS}
        seg = Segment(raw, K, W, H, n=end - start)
        seg._frames = frames                         # the harness's image modules read the encoded frames
        p = run_segment(chain, seg)
        if out is None:
            out = {k: np.full((n,) + v.shape[1:], np.nan, np.float32) for k, v in p.items()}
        for k, v in p.items():
            out[k][start + keep:end] = v[keep:]
        print(f'  frames {start + keep}-{end - 1} done ({time.time() - t0:.0f} s)', flush=True)
    out['hand_pose'] = out.pop('hp')
    a.out.mkdir(parents=True, exist_ok=True)
    dst = a.out / f'{Path(video).stem}.npz'
    np.savez_compressed(dst, **out, K=K, W=W, H=H)
    on = (out['exists'] > 0.5).sum(0)
    print(f'[done] {dst}: left {on[0]}/{n}, right {on[1]}/{n} frames with exists > 0.5', flush=True)


if __name__ == '__main__':
    main()
