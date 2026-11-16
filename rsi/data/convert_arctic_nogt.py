"""ARCTIC subject s03 (the official test subject: images public, MANO / egocam withheld) -> ACE-protocol layout for
teacher-only training (ARCTIC-domain video without GT, like the 62 HOT3D no-GT recordings).

Same image pipeline as data/convert_arctic.py (not modified): view 0 cropped 840 x 600 -> 672 x 480, 30 fps, every
frame, the first 10 frames dropped ("images may be entirely black"), non-overlapping 81-frame segments, shorter tail
dropped. No GT: every GT file has valid = False and NaN poses (the evaluator never sees s03; Loop B's GT term ignores it).
Camera: s03's egocam intrinsics are not released, so K is the per-element median of the scaled (x 0.24) ego K over every
other ARCTIC recording (fx ~579; the principal point varies by ~+-10 px between recordings, recorded in meta).
Output (--out): videos/<sid>.mp4, gt/<sid>__s<k>.npz, segments/<sid>.json, meta/<sid>.json with sid = ar__s03_<seq>.

  python data/convert_arctic_nogt.py --images $RSI_DATA_RAW/ARCTIC/cropped_images/s03 \
      --raw $RSI_DATA_RAW/ARCTIC/raw_seqs --out data/arctic_s03_ace --jobs 8
"""
from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

W, H, SCALE, SEG, TRIM_START = 672, 480, 0.24, 81, 10


def median_K(raw: Path) -> tuple[np.ndarray, dict]:
    Ks = []
    for f in sorted(raw.glob('s*/*.egocam.dist.npy')):
        K = np.asarray(np.load(f, allow_pickle=True).item()['intrinsics'], np.float64).copy()
        K[:2] *= SCALE
        Ks.append(K)
    Ks = np.stack(Ks)
    K = np.median(Ks, 0)
    spread = dict(n=len(Ks), fx=[float(Ks[:, 0, 0].min()), float(Ks[:, 0, 0].max())],
                  cx=[float(Ks[:, 0, 2].min()), float(Ks[:, 0, 2].max())], cy=[float(Ks[:, 1, 2].min()), float(Ks[:, 1, 2].max())])
    return K, spread


def convert(images: Path, seq: str, out: Path, K: np.ndarray, spread: dict) -> dict:
    import imageio_ffmpeg
    sid = f'ar__s03_{seq}'
    for d in ('videos', 'gt', 'segments', 'meta'):
        (out / d).mkdir(parents=True, exist_ok=True)
    imgs = sorted((images / seq / '0').glob('*.jpg'), key=lambda p: int(p.stem))
    idx = np.array([int(p.stem) for p in imgs])
    assert len(imgs) and (np.diff(idx) == 1).all(), f'{sid}: image indices are not contiguous'
    n = len(imgs)
    vid = out / 'videos' / f'{sid}.mp4'
    enc = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                            '-s', f'{W}x{H}', '-r', '30', '-i', '-', '-c:v', 'libx264', '-threads', '4', '-preset', 'medium',
                            '-crf', '18', '-pix_fmt', 'yuv420p', str(vid)], stdin=subprocess.PIPE)
    for p in imgs:
        im = cv2.imread(str(p))
        assert im is not None and im.shape[:2] == (600, 840), f'{p}: {None if im is None else im.shape}'
        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
        enc.stdin.write(np.ascontiguousarray(im[:, :, ::-1]).tobytes())
    enc.stdin.close()
    assert enc.wait() == 0, f'{sid}: ffmpeg failed'
    segs = []
    for old in (out / 'gt').glob(f'{sid}__s*.npz'):
        old.unlink()
    for k in range(max(0, n - TRIM_START) // SEG):
        a0, a1 = TRIM_START + k * SEG, TRIM_START + (k + 1) * SEG
        name = f'{sid}__s{k:03d}'
        np.savez_compressed(out / 'gt' / f'{name}.npz', K=K.astype(np.float32), W=np.int32(W), H=np.int32(H),
                            valid=np.zeros((SEG, 2), bool), joints_cam=np.full((SEG, 2, 21, 3), np.nan, np.float32),
                            transl=np.full((SEG, 2, 3), np.nan, np.float32), rot=np.full((SEG, 2, 3, 3), np.nan, np.float32),
                            verts_cam=np.full((SEG, 2, 778, 3), np.nan, np.float32))
        segs.append(dict(name=name, start=a0, end=a1))
    (out / 'segments' / f'{sid}.json').write_text(json.dumps(dict(sequence_id=sid, video=str(vid), frames=n, fps=30.0,
                                                                   K=K.tolist(), W=W, H=H, segments=segs), indent=1))
    meta = dict(sequence_id=sid, subject='s03', seq=seq, frames=n, segments=len(segs), first_image=int(idx[0]),
                trim=[TRIM_START, 0], hand_frames_annotated=0, gt='none (teacher only)', K=K.tolist(),
                K_source='median of the other subjects\' scaled ego K', K_spread=spread)
    (out / 'meta' / f'{sid}.json').write_text(json.dumps(meta, indent=1))
    return meta


def _job(args):
    try:
        return convert(*args)
    except Exception as e:                                          # noqa: BLE001
        return dict(sequence_id=f'ar__s03_{args[1]}', error=repr(e))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--images', type=Path, required=True)
    ap.add_argument('--raw', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--jobs', type=int, default=8)
    a = ap.parse_args()
    K, spread = median_K(a.raw)
    print('K', K.round(2).tolist(), spread, flush=True)
    seqs = sorted(p.name for p in a.images.iterdir() if (p / '0').is_dir())
    jobs = [(a.images, s, a.out, K, spread) for s in seqs if not (a.out / 'meta' / f'ar__s03_{s}.json').is_file()]
    print(f'{len(jobs)} sequences to convert', flush=True)
    with ProcessPoolExecutor(a.jobs) as ex:
        for i, m in enumerate(ex.map(_job, jobs), 1):
            print(f"[{i}/{len(jobs)}] {json.dumps({k: m.get(k) for k in ('sequence_id', 'frames', 'segments', 'error')})}", flush=True)


if __name__ == '__main__':
    main()
