"""ARCTIC (egocentric view 0) -> the ACE-Ego-Hand evaluation protocol, one sequence per call.

Protocol (ACE arXiv 2608.20308, App. A.1-A.2 and Table 2): 672 x 480, 30 fps, every frame, non-overlapping 81-frame
segments (a shorter tail is dropped); test subject s05 = 34 recordings, 24,863 frames, 291 segments; the EPE2D miss
penalty is the image diagonal (826 px).

Read-only source (--root): cropped_images/<sub>/<seq>/0/%05d.jpg (840 x 600 = 0.3 x the 2800 x 2000 ego image),
raw_seqs/<sub>/<seq>.{mano,egocam.dist}.npy, meta/misc.json.
- Frame alignment: image <i>.jpg shows mocap frame i - ioi_offset[sub] (ARCTIC src/datasets/arctic_dataset.py).
  Images whose mocap index falls outside the sequence are dropped; they can only sit at the ends.
- Camera: ARCTIC's own convention. The undistorted pinhole K_ego (egocam 'intrinsics', defined on 2800 x 2000) is
  scaled by 0.24 and used with the (distorted) ego image, as ARCTIC's benchmarks do ("kp2d, kp3d are in undistorted
  space").
- MANO: smplx, flat_hand_mean=False, axis-angle pose relative to the mean. No left-hand shapedirs fix: ARCTIC's body
  models do not apply one, and neither does ACE's MANO.
- Frame range: ACE's counts (s05 24,863 frames / 291 segments, train 184,373 frames) are matched exactly
  by dropping 30 frames per recording, and by nothing else we tried (0: 25,883 / 302; ARCTIC's own 10 + 10: 25,203 /
  295). We drop the first 10 (ARCTIC src/arctic/split.py: "images may be entirely black") and the last 20; which end
  the other 20 come from is a guess. The video keeps every frame; segments start at frame 10.
- Camera-frame GT, as in data/convert_hot3d.py: rot = R_cw R, tau = R_cw (J0 + t) + t_cw - J0, where J0 is
  the rest wrist joint (smplx rotates about it).
Output (under --out): videos/<sid>.mp4 (672 x 480, every kept frame), gt/<sid>__s<k>.npz, segments/<sid>.json,
meta/<sid>.json, with sid = ar__<sub>_<seq>.

  python data/convert_arctic.py --root $RSI_DATA_RAW/ARCTIC --sub s05 --seq box_grab_01 --out data/arctic_ace
  python data/convert_arctic.py --root ... --all --out data/arctic_ace --jobs 24
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rsi_paths  # noqa: E402,F401  (puts ace_eval on sys.path)

W, H, SCALE, SEG = 672, 480, 0.24, 81
TRIM_START, TRIM_END = 10, 20


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def convert(root: Path, sub: str, seq: str, out: Path) -> dict:
    from ace_eval import mano as AM
    sid = f"ar__{sub}_{seq}"
    for d in ("videos", "gt", "segments", "meta"):
        (out / d).mkdir(parents=True, exist_ok=True)
    off = int(json.loads((root / "meta" / "misc.json").read_text())[sub]["ioi_offset"])
    mano = np.load(root / "raw_seqs" / sub / f"{seq}.mano.npy", allow_pickle=True).item()
    ego = np.load(root / "raw_seqs" / sub / f"{seq}.egocam.dist.npy", allow_pickle=True).item()
    n_mocap = len(mano["right"]["rot"])
    imgs = sorted((root / "cropped_images" / sub / seq / "0").glob("*.jpg"), key=lambda p: int(p.stem))
    keep = [(int(p.stem) - off, p) for p in imgs if 0 <= int(p.stem) - off < n_mocap]
    vids = np.array([v for v, _ in keep])
    assert len(vids) and (np.diff(vids) == 1).all(), f"{sid}: kept mocap indices are not contiguous"
    n = len(vids)

    K = np.asarray(ego["intrinsics"], np.float64).copy()
    K[:2] *= SCALE
    vid = out / "videos" / f"{sid}.mp4"
    if vid.is_file() and int(cv2.VideoCapture(str(vid)).get(cv2.CAP_PROP_FRAME_COUNT)) == n:
        enc = None                                                  # already encoded (re-run of the GT only)
    else:
        enc = subprocess.Popen([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{W}x{H}", "-r", "30", "-i", "-", "-c:v", "libx264", "-threads", "4",
                            "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", str(vid)], stdin=subprocess.PIPE)
    for _, p in (keep if enc else []):
        im = cv2.imread(str(p))
        assert im is not None and im.shape[:2] == (600, 840), f"{p}: {None if im is None else im.shape}"
        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
        enc.stdin.write(np.ascontiguousarray(im[:, :, ::-1]).tobytes())
    if enc:
        enc.stdin.close()
        assert enc.wait() == 0, f"{sid}: ffmpeg failed"

    models = AM.load_models(flat_hand_mean=False)
    Rcw = np.asarray(ego["R_k_cam_np"], np.float64)[vids]
    tcw = np.asarray(ego["T_k_cam_np"], np.float64).reshape(-1, 3)[vids]
    J = np.full((n, 2, 21, 3), np.nan, np.float32); V = np.full((n, 2, 778, 3), np.nan, np.float32)
    tau = np.full((n, 2, 3), np.nan, np.float32); rot = np.full((n, 2, 3, 3), np.nan, np.float32)
    valid = np.zeros((n, 2), bool)
    for s, hand in enumerate(("left", "right")):
        h = mano[hand]
        aa = np.asarray(h["rot"], np.float64)[vids]; hp = np.asarray(h["pose"], np.float64)[vids]
        tw = np.asarray(h["trans"], np.float64)[vids]
        B = np.repeat(np.asarray(h["shape"], np.float64).reshape(-1)[None, :10], n, 0)
        ok = np.isfinite(aa).all(1) & np.isfinite(hp).all(1) & np.isfinite(tw).all(1) & np.isfinite(Rcw).all((1, 2))
        rc = np.full((n, 3, 3), np.nan)
        rc[ok] = Rcw[ok] @ Rotation.from_rotvec(aa[ok]).as_matrix()
        for k in range(0, n, 512):
            ii = np.arange(k, min(n, k + 512))
            ii = ii[ok[ii]]
            if not len(ii):
                continue
            j0, v0 = AM.forward(models, s, rc[ii], hp[ii], B[ii])
            j00 = j0[:, 0].astype(np.float64)                      # OpenPose joint 0 = wrist = rest root J0
            t0 = np.einsum("nij,nj->ni", Rcw[ii], j00 + tw[ii]) + tcw[ii] - j00
            J[ii, s] = j0 + t0[:, None]; V[ii, s] = v0 + t0[:, None]; tau[ii, s] = t0
        rot[:, s] = rc
        valid[:, s] = ok & np.isfinite(J[:, s]).all((1, 2))

    segs = []
    for old in (out / "gt").glob(f"{sid}__s*.npz"):
        old.unlink()
    for k in range(max(0, n - TRIM_START - TRIM_END) // SEG):
        a0, a1 = TRIM_START + k * SEG, TRIM_START + (k + 1) * SEG
        name = f"{sid}__s{k:03d}"
        np.savez_compressed(out / "gt" / f"{name}.npz", K=K.astype(np.float32), W=np.int32(W), H=np.int32(H),
                            valid=valid[a0:a1], joints_cam=J[a0:a1], transl=tau[a0:a1], rot=rot[a0:a1],
                            verts_cam=V[a0:a1])
        segs.append(dict(name=name, start=a0, end=a1))
    (out / "segments" / f"{sid}.json").write_text(json.dumps(dict(sequence_id=sid, video=str(vid), frames=n, fps=30.0,
                                                                   K=K.tolist(), W=W, H=H, segments=segs), indent=1))
    # on-screen share (any joint in frame, z > 1 cm), for the record
    z = J[..., 2]; u = K[0, 0] * J[..., 0] / np.maximum(z, 1e-4) + K[0, 2]; v = K[1, 1] * J[..., 1] / np.maximum(z, 1e-4) + K[1, 2]
    on = ((u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0.01)).any(-1) & valid
    meta = dict(sequence_id=sid, subject=sub, seq=seq, frames=n, frames_used=n - TRIM_START - TRIM_END,
                trim=[TRIM_START, TRIM_END], segments=len(segs), images=len(imgs), ioi_offset=off,
                mocap_frames=n_mocap, first_mocap=int(vids[0]), hand_frames_annotated=int(valid.sum()),
                hand_frames_on_screen=int(on.sum()), K=K.tolist())
    (out / "meta" / f"{sid}.json").write_text(json.dumps(meta, indent=1))
    return meta


def _job(args):
    try:
        return convert(*args)
    except Exception as e:                                          # noqa: BLE001
        return dict(sequence_id=f"ar__{args[1]}_{args[2]}", error=repr(e))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sub"); ap.add_argument("--seq")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--subjects", default="", help="comma list with --all (default: every subject present)")
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    if not a.all:
        print(json.dumps(convert(a.root, a.sub, a.seq, a.out)))
        return
    subs = a.subjects.split(",") if a.subjects else sorted(p.name for p in (a.root / "cropped_images").iterdir())
    jobs = [(a.root, s, q.name, a.out) for s in subs for q in sorted((a.root / "cropped_images" / s).iterdir())
            if q.is_dir() and not (a.out / "meta" / f"ar__{s}_{q.name}.json").is_file()]
    print(f"{len(jobs)} sequences to convert", flush=True)
    with ProcessPoolExecutor(a.jobs) as ex:
        for i, m in enumerate(ex.map(_job, jobs), 1):
            print(f"[{i}/{len(jobs)}] {json.dumps({k: m.get(k) for k in ('sequence_id', 'frames', 'segments', 'error')})}",
                  flush=True)


if __name__ == "__main__":
    main()
