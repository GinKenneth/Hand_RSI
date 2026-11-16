"""HOT3D -> the ACE-Ego-Hand / ViDiHand evaluation protocol, one sequence per call.

Protocol (ViDiHand arXiv 2606.30308, Table "Dataset # train # test ..."; ACE arXiv 2608.20308, A.2):
  480 x 480 linear (pinhole) camera, fx = fy = 207.8, cx = cy = 239.8 after the upright (cw 90) rotation; 30 fps, every
  frame; non-overlapping 81-frame segments, a tail shorter than 81 frames is dropped.
Ground truth per segment in the format of ace_eval (README): K, W, H, valid (annotation exists, in view or
not), joints_cam = J + tau, transl = tau (MANO root translation, not the wrist), rot (global orient, camera frame),
verts_cam. MANO goes through the evaluator's own smplx forward (flat_hand_mean=False), so ground truth and predictions
share one convention. HOT3D's wrist_xform is passed to smplx as global_orient and transl (hand_tracking_toolkit mano_hand_model.py), so its
translation t is the MANO root translation in the world frame; in the camera frame tau = R_cw (J0 + t) + t_cw - J0, with
J0 the rest wrist joint of the sequence's betas (smplx rotates about it). HOT3D's fix of the smplx left-hand shapedirs (smplx issue 48) is applied, as HOT3D's
own ground truth uses it.

Read-only input: an official HOT3D Aria sequence directory, $RSI_DATA_RAW/HOT3D/<seq> (recording.vrs,
camera_models.json, mano_hand_pose_trajectory.jsonl, headset_trajectory.csv, timecode_devicetime_mapping.csv).
Output (under --out): videos/<sid>.mp4 (480x480, all frames), gt/<sid>__s<k>.npz, segments/<sid>.json, meta/<sid>.json.

  <python with projectaria_tools, smplx, torch> data/convert_hot3d.py --seq <dir> --out data/hot3d_ace [--max-frames N]
(all HOT3D sequences with public MANO GT go to data/hot3d_ace; data/build_views.py then links the splits)
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rsi_paths  # noqa: E402,F401  (puts ace_eval on sys.path)

SIZE, F_PX, C_PX, SEG = 480, 207.8, 239.8, 81



def quat_to_R(w, x, y, z):
    n = np.sqrt(w * w + x * x + y * y + z * z); w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def load_tracks(seq: Path):
    hands = []          # (timecode, {hand: (pca, R, t)})
    betas = None
    for line in open(seq / 'mano_hand_pose_trajectory.jsonl'):
        r = json.loads(line); hp = {}
        for idx, hand in (('0', 'left'), ('1', 'right')):
            if idx in r['hand_poses']:
                h = r['hand_poses'][idx]
                q, t = h['wrist_xform']['q_wxyz'], h['wrist_xform']['t_xyz']
                hp[hand] = (np.asarray(h['pose'], np.float64), quat_to_R(*q), np.asarray(t, np.float64))
                if betas is None and 'betas' in h:
                    betas = np.asarray(h['betas'], np.float64)
        hands.append((int(r['timestamp_ns']), hp))
    hands.sort(key=lambda x: x[0])
    head = []
    with open(seq / 'headset_trajectory.csv') as f:
        for row in csv.DictReader(f):
            T = np.eye(4)
            T[:3, :3] = quat_to_R(float(row['q_wo_w']), float(row['q_wo_x']), float(row['q_wo_y']), float(row['q_wo_z']))
            T[:3, 3] = [float(row['t_wo_x[m]']), float(row['t_wo_y[m]']), float(row['t_wo_z[m]'])]
            head.append((int(row['timestamp[ns]']), T))
    head.sort(key=lambda x: x[0])
    tc = {}
    with open(seq / 'timecode_devicetime_mapping.csv') as f:
        for row in csv.DictReader(f):
            tc[int(row['devicetime_ns'])] = int(row['timecode_ns'])
    return hands, betas, head, tc


def nearest(track, ts, t, tol_ns=2_000_000):
    """record of a time-sorted track (timestamps ts) nearest to t, if within tol_ns"""
    i = bisect.bisect_left(ts, t)
    best = min((j for j in (i - 1, i) if 0 <= j < len(ts)), key=lambda j: abs(ts[j] - t), default=None)
    return track[best] if best is not None and abs(ts[best] - t) <= tol_ns else None

def mano_models():
    from ace_eval import mano as AM
    models = AM.load_models(flat_hand_mean=False)
    l, r = models[0], models[1]
    if (l.shapedirs[:, 0, :] - r.shapedirs[:, 0, :]).abs().sum() < 1:       # HOT3D / smplx issue 48
        l.shapedirs[:, 0, :] *= -1
    import glob, pickle
    comps = {}
    for s, name in ((0, 'MANO_LEFT.pkl'), (1, 'MANO_RIGHT.pkl')):   # PCA basis of the same model files (HOT3D: 15 coefficients)
        f = (glob.glob(f'{AM.MANO_DIR}/**/{name}', recursive=True) or glob.glob(f'{AM.MANO_DIR}/{name}'))[0]
        comps[s] = np.asarray(pickle.load(open(f, 'rb'), encoding='latin1')['hands_components'], np.float64)[:15]
    return AM, models, comps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--max-frames', type=int, default=0); ap.add_argument('--sid-prefix', default='h3__')
    ap.add_argument('--check-old', type=Path, default=None, help='team-format gt3d.npz of the same sequence (1024 camera)')
    a = ap.parse_args()
    from projectaria_tools.core import calibration, data_provider, sophus
    import imageio_ffmpeg
    seq = a.seq; sid = a.sid_prefix + seq.name; out = a.out
    for d in ('videos', 'gt', 'segments', 'meta'):
        (out / d).mkdir(parents=True, exist_ok=True)
    p = data_provider.create_vrs_data_provider(str(seq / 'recording.vrs'))
    stream = p.get_stream_id_from_label('camera-rgb')
    cm = {c['label']: c for c in json.loads((seq / 'camera_models.json').read_text())}['camera-rgb']
    q, tr = cm['T_Device_Camera']['quaternion_wxyz'], cm['T_Device_Camera']['translation_xyz']
    T = sophus.SE3.from_quat_and_translation(float(q[0]), np.array(q[1:], np.float64), np.array(tr, np.float64))
    src = calibration.CameraCalibration('camera-rgb', calibration.CameraModelType.FISHEYE624,
                                        np.array(cm['projectionParams'], np.float64), T, int(cm['imageWidth']),
                                        int(cm['imageHeight']), None, float(cm['maxSolidAngle']), str(cm.get('serialNumber', '')))
    # linear camera whose cw90 rotation has principal point (239.8, 239.8): before the rotation (cx, cy) = (239.8, W-1-239.8)
    dst = calibration.CameraCalibration('camera-rgb', calibration.CameraModelType.LINEAR,
                                        np.array([F_PX, F_PX, C_PX, SIZE - 1 - C_PX], np.float64), T, SIZE, SIZE, None,
                                        float(cm['maxSolidAngle']), str(cm.get('serialNumber', '')))
    dst_cw = calibration.rotate_camera_calib_cw90deg(dst)
    fx, fy = dst_cw.get_focal_lengths(); cx, cy = dst_cw.get_principal_point()
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    assert np.allclose(K, [[F_PX, 0, C_PX], [0, F_PX, C_PX], [0, 0, 1]], atol=1e-3), K
    T_dev_cam = dst_cw.get_transform_device_camera().to_matrix()
    n = p.get_num_data(stream)
    if a.max_frames:
        n = min(n, a.max_frames)
    hands, betas, head, tc = load_tracks(seq)
    hands_ts, head_ts, tc_keys = [x[0] for x in hands], [x[0] for x in head], sorted(tc)
    AM, models, comps = mano_models()
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    vid = out / 'videos' / f'{sid}.mp4'
    enc = subprocess.Popen([ff, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{SIZE}x{SIZE}',
                            '-r', '30', '-i', '-', '-c:v', 'libx264', '-threads', '4', '-preset', 'medium', '-crf', '18',
                            '-pix_fmt', 'yuv420p', str(vid)], stdin=subprocess.PIPE)
    valid = np.zeros((n, 2), bool)
    rot = np.full((n, 2, 3, 3), np.nan); tw_all = np.full((n, 2, 3), np.nan); hp = np.full((n, 2, 45), np.nan)
    Rcw = np.full((n, 3, 3), np.nan); tcw = np.full((n, 3), np.nan)
    for i in range(n):
        img, rec = p.get_image_data_by_index(stream, i)
        und = calibration.distort_by_calibration(img.to_numpy_array(), dst, src)
        und = np.ascontiguousarray(np.rot90(und, k=3))            # clockwise 90: upright, matches dst_cw
        if und.ndim == 2:
            und = np.repeat(und[..., None], 3, axis=2)
        enc.stdin.write(und.astype(np.uint8).tobytes())
        t_dev = rec.capture_timestamp_ns
        t = tc.get(t_dev)
        if t is None:
            j = bisect.bisect_left(tc_keys, t_dev)
            j = min((x for x in (j - 1, j) if 0 <= x < len(tc_keys)), key=lambda x: abs(tc_keys[x] - t_dev))
            t = tc[tc_keys[j]] + (t_dev - tc_keys[j])
        hrec, hd = nearest(hands, hands_ts, t), nearest(head, head_ts, t)
        if hrec is None or hd is None:
            continue
        T_cam_world = np.linalg.inv(hd[1] @ T_dev_cam)
        Rcw[i], tcw[i] = T_cam_world[:3, :3], T_cam_world[:3, 3]
        for s, hand in enumerate(('left', 'right')):
            if hand not in hrec[1]:
                continue
            pca, R, tw = hrec[1][hand]
            valid[i, s] = True
            rot[i, s] = T_cam_world[:3, :3] @ R
            tw_all[i, s] = tw                           # smplx transl, world frame
            hp[i, s] = pca @ comps[s]                    # relative to hands_mean: smplx adds it (flat_hand_mean=False)
    enc.stdin.close(); enc.wait()
    J = np.full((n, 2, 21, 3), np.nan, np.float32); V = np.full((n, 2, 778, 3), np.nan, np.float32)
    tau = np.full((n, 2, 3), np.nan, np.float32)
    B = np.repeat(np.asarray(betas, np.float64)[None, :10], n, 0)
    for s in (0, 1):
        idx = np.flatnonzero(valid[:, s])
        for k in range(0, len(idx), 512):
            ii = idx[k:k + 512]
            j0, v0 = AM.forward(models, s, rot[ii, s], hp[ii, s], B[ii])
            j00 = j0[:, 0].astype(np.float64)                          # OpenPose joint 0 = wrist = rest root J0
            t0 = np.einsum('nij,nj->ni', Rcw[ii], j00 + tw_all[ii, s]) + tcw[ii] - j00
            J[ii, s] = j0 + t0[:, None]; V[ii, s] = v0 + t0[:, None]; tau[ii, s] = t0
    segs = []
    for k in range(n // SEG):
        a0, a1 = k * SEG, (k + 1) * SEG
        name = f'{sid}__s{k:03d}'
        np.savez_compressed(out / 'gt' / f'{name}.npz', K=K.astype(np.float32), W=np.int32(SIZE), H=np.int32(SIZE),
                            valid=valid[a0:a1], joints_cam=J[a0:a1], transl=tau[a0:a1], rot=rot[a0:a1].astype(np.float32),
                            verts_cam=V[a0:a1])
        segs.append(dict(name=name, start=a0, end=a1))
    (out / 'segments' / f'{sid}.json').write_text(json.dumps(dict(sequence_id=sid, video=str(vid), frames=n, fps=30.0,
                                                                   K=K.tolist(), W=SIZE, H=SIZE, segments=segs), indent=1))
    meta = dict(sequence_id=sid, source=str(seq), frames=n, segments=len(segs), hand_frames_annotated=int(valid.sum()),
                K=K.tolist())
    if a.check_old and a.check_old.is_file():          # same extrinsics as the 1024 conversion: camera-frame joints must agree
        old = np.load(a.check_old); fr = old['frame_numbers']; o3 = old['keypoints_3d_cam'].astype(np.float64)
        fr_ok = fr < n; d = np.linalg.norm(J[fr[fr_ok]] - o3[fr_ok], axis=-1)            # (m, 2, 21)
        meta['check_vs_team_conversion_mm'] = {h: float(np.nanmean(d[:, s]) * 1000) for s, h in enumerate(('left', 'right'))}
    (out / 'meta' / f'{sid}.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
