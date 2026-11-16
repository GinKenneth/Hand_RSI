"""HOI4D -> the ACE-Ego-Hand evaluation protocol (zero-shot, sealed test only).

Protocol (ACE arXiv 2608.20308, App. A.2 and Table 2): 166 recordings, 49,800 frames (all 300 frames of each), 498
segments (3 x 81 frames from frame 0), 832 x 480, every frame, treated as 30 fps (HOI4D is natively 15 fps; ACE does not
resample). ACE's 166 recordings are not published (they come from ViDiHand), so we draw 166 with a fixed seed
(20261029) from the released recordings that have a 300-frame video and any hand-pose annotation. The draw was fixed
before any HOI4D score was seen.

Ground truth: the released annotations as they are. A hand-frame is valid where HOI4D ships its
pickle; elsewhere the hand has no GT, so a detection there counts as a false positive. ACE's own HOI4D row (recall 0.996,
F1 0.974) shows the same pattern.

Conventions (HOI4D-Instructions README):
- manopth ManoLayer(use_pca=False, flat_hand_mean=True): poseCoeff = 3 global + 45 ABSOLUTE axis-angle pose; joints in
  mm + trans (m), camera frame.
- Mapped to the evaluator's smplx (flat_hand_mean=False): hand_pose = pose45 - hands_mean, transl = trans,
  global_orient = poseCoeff[:3]. No left-hand shapedirs fix: manopth loads the same pkl.
  Checked: median reprojection error 0.9 px against HOI4D's kps2D on a sample (fingertips 2-6 px, from a different
  tip-vertex choice).
- Size and intrinsics: ACE's own infer_video.py snaps both axes to the VAE's 32-pixel grid with a maximum width of 832,
  which turns HOI4D's 1920 x 1080 into 832 x 480 (an earlier note said 854 x 480; the VAE rejects that). The
  intrinsics camera_params/<camera>/intrin.npy are scaled by 832/1920 and 480/1080.
Each meta/<sid>.json records the reprojection error against kps2D: an independent check of the conversion.

  python data/convert_hoi4d.py --root $RSI_DATA_RAW/HOI4D --out data/hoi4d_ace --jobs 24
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import pickle
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rsi_paths  # noqa: E402,F401  (puts ace_eval on sys.path)

W, H, SEG, NSEG, NFR, N_REC, SEED = 832, 480, 81, 3, 300, 166, 20261029
SIDES = (('left', 0), ('right', 1))


def ffmpeg_exe() -> str:
    exe = shutil.which('ffmpeg')
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def index(root: Path, cache: Path) -> dict:
    """{seq: {'left': [frames], 'right': [frames]}} for every recording with a released video."""
    if cache.is_file():
        return json.loads(cache.read_text())
    hp = zipfile.ZipFile(root / 'Hand_pose.zip')
    idx = defaultdict(lambda: {'left': [], 'right': []})
    for n in hp.namelist():
        if not n.endswith('.pickle'):
            continue
        parts = n.split('/')                      # Hand_pose/handpose_<side>_hand/<7 levels>/<frame>.pickle
        side = 'left' if 'left' in parts[1] else 'right'
        idx['/'.join(parts[2:9])][side].append(int(parts[9][:-7]))
    vids = {n.split('/', 1)[1].rsplit('/align_rgb/', 1)[0] for n in zipfile.ZipFile(root / 'HOI4D_release.zip').namelist()
            if n.endswith('align_rgb/image.mp4')}
    out = {s: {k: sorted(v) for k, v in d.items()} for s, d in idx.items() if s in vids}
    cache.write_text(json.dumps(out))
    return out


def mano_mean() -> dict:
    from ace_eval import mano as AM
    out = {}
    for s, nm in ((0, 'MANO_LEFT.pkl'), (1, 'MANO_RIGHT.pkl')):
        f = (glob.glob(f'{AM.MANO_DIR}/**/{nm}', recursive=True) or glob.glob(f'{AM.MANO_DIR}/{nm}'))[0]
        out[s] = np.asarray(pickle.load(open(f, 'rb'), encoding='latin1')['hands_mean'], np.float64)
    return out


def convert(root: Path, seq: str, frames: dict, out: Path) -> dict:
    from ace_eval import mano as AM
    sid = 'hd__' + seq.replace('/', '_')
    for d in ('videos', 'gt', 'segments', 'meta'):
        (out / d).mkdir(parents=True, exist_ok=True)
    K0 = np.load(io.BytesIO(zipfile.ZipFile(root / 'camera_params.zip').read(f"camera_params/{seq.split('/')[0]}/intrin.npy")))
    with tempfile.NamedTemporaryFile(suffix='.mp4') as tmp:
        tmp.write(zipfile.ZipFile(root / 'HOI4D_release.zip').read(f'HOI4D_release/{seq}/align_rgb/image.mp4')); tmp.flush()
        cap = cv2.VideoCapture(tmp.name)
        W0, H0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid = out / 'videos' / f'{sid}.mp4'
        enc = subprocess.Popen([ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
                                '-r', '30', '-i', '-', '-c:v', 'libx264', '-threads', '4', '-preset', 'medium', '-crf', '18',
                                '-pix_fmt', 'yuv420p', str(vid)], stdin=subprocess.PIPE)
        n = 0
        while n < NFR:
            ok, im = cap.read()
            if not ok:
                break
            enc.stdin.write(np.ascontiguousarray(cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)[:, :, ::-1]).tobytes())
            n += 1
        enc.stdin.close()
        assert enc.wait() == 0 and n == NFR, f'{sid}: {n} frames decoded'
    K = K0.astype(np.float64).copy(); K[0] *= W / W0; K[1] *= H / H0
    models, mean = AM.load_models(flat_hand_mean=False), mano_mean()
    hz = zipfile.ZipFile(root / 'Hand_pose.zip')
    J = np.full((n, 2, 21, 3), np.nan, np.float32); V = np.full((n, 2, 778, 3), np.nan, np.float32)
    tau = np.full((n, 2, 3), np.nan, np.float32); rot = np.full((n, 2, 3, 3), np.nan, np.float32)
    valid = np.zeros((n, 2), bool); reproj = []
    for side, s in SIDES:
        fr = [f for f in frames.get(side, []) if 0 <= f < n]
        if not fr:
            continue
        P = []
        for f in fr:
            d = pickle.load(io.BytesIO(hz.read(f'Hand_pose/handpose_{side}_hand/{seq}/{f}.pickle')), encoding='latin1')
            P.append((np.asarray(d['poseCoeff'], np.float64), np.asarray(d['beta'], np.float64),
                      np.asarray(d['trans'], np.float64), np.asarray(d['kps2D'], np.float64)))
        pc = np.stack([p[0] for p in P]); b = np.stack([p[1] for p in P]); t = np.stack([p[2] for p in P]); k2 = np.stack([p[3] for p in P])
        j0, v0 = AM.forward(models, s, pc[:, :3], pc[:, 3:] - mean[s], b)
        fr = np.asarray(fr)
        J[fr, s] = j0 + t[:, None]; V[fr, s] = v0 + t[:, None]; tau[fr, s] = t
        rot[fr, s] = Rotation.from_rotvec(pc[:, :3]).as_matrix()
        valid[fr, s] = np.isfinite(J[fr, s]).all((1, 2))
        X = J[fr, s].astype(np.float64)                             # reprojection in the ORIGINAL camera vs kps2D
        uv = np.stack([K0[0, 0] * X[..., 0] / X[..., 2] + K0[0, 2], K0[1, 1] * X[..., 1] / X[..., 2] + K0[1, 2]], -1)
        e = np.linalg.norm(uv - k2, axis=-1)
        reproj.append(dict(side=side, frames=len(fr), median_px=float(np.median(e)),
                           median_px_no_tips=float(np.median(e[:, [i for i in range(21) if i not in (4, 8, 12, 16, 20)]]))))
    segs = []
    for k in range(NSEG):
        a0, a1 = k * SEG, (k + 1) * SEG
        name = f'{sid}__s{k:03d}'
        np.savez_compressed(out / 'gt' / f'{name}.npz', K=K.astype(np.float32), W=np.int32(W), H=np.int32(H), valid=valid[a0:a1],
                            joints_cam=J[a0:a1], transl=tau[a0:a1], rot=rot[a0:a1], verts_cam=V[a0:a1])
        segs.append(dict(name=name, start=a0, end=a1))
    (out / 'segments' / f'{sid}.json').write_text(json.dumps(dict(sequence_id=sid, video=str(vid), frames=n, fps=30.0,
                                                                   K=K.tolist(), W=W, H=H, segments=segs), indent=1))
    meta = dict(sequence_id=sid, source=seq, frames=n, segments=len(segs), source_size=[W0, H0], native_fps=15,
                hand_frames_annotated=int(valid.sum()), hand_frames_annotated_in_segments=int(valid[:NSEG * SEG].sum()),
                reprojection_vs_kps2D=reproj, K=K.tolist())
    (out / 'meta' / f'{sid}.json').write_text(json.dumps(meta, indent=1))
    return meta


def _job(args):
    try:
        return convert(*args)
    except Exception as e:                                          # noqa: BLE001
        return dict(sequence_id='hd__' + args[1].replace('/', '_'), error=repr(e))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--jobs', type=int, default=24)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    idx = index(a.root, a.out / 'annotation_index.json')
    pool = sorted(s for s, d in idx.items() if d['left'] or d['right'])
    pick = sorted(np.random.default_rng(SEED).choice(pool, N_REC, replace=False).tolist())
    (a.out / 'selection.json').write_text(json.dumps(dict(rule=f'{N_REC} of the {len(pool)} released recordings with any hand '
                                                          f'annotation, numpy default_rng({SEED}).choice', recordings=pick), indent=1))
    print(f'{len(idx)} released recordings with a video and a hand-pose folder; pool {len(pool)}; picked {len(pick)}', flush=True)
    jobs = [(a.root, s, idx[s], a.out) for s in pick if not (a.out / 'meta' / f"hd__{s.replace('/', '_')}.json").is_file()]
    with ProcessPoolExecutor(a.jobs) as ex:
        for i, m in enumerate(ex.map(_job, jobs), 1):
            print(f"[{i}/{len(jobs)}] {json.dumps({k: m.get(k) for k in ('sequence_id', 'frames', 'hand_frames_annotated', 'error')})} "
                  f"reproj {[round(r['median_px_no_tips'], 2) for r in m.get('reprojection_vs_kps2D', [])]}", flush=True)
    (Path(__file__).resolve().parent / 'splits' / 'hoi4d_test.txt').write_text(
        '\n'.join('hd__' + s.replace('/', '_') for s in pick) + '\n')


if __name__ == '__main__':
    main()
