"""HOT3D recordings WITHOUT public hand GT (the official test set's 62 Aria recordings) -> ACE-protocol layout for
teacher-only training.

Same camera pipeline as data/convert_hot3d.py: Aria RGB undistorted to a
480 x 480 linear camera, rotated 90 deg clockwise, fx = fy = 207.8, cx = cy = 239.8, 30 fps, every frame, non-overlapping
81-frame segments. Their mano_hand_pose_trajectory.jsonl is empty, so every GT file has valid = False and NaN poses: the
evaluator never sees these recordings (train split only), and Loop B's GT term ignores them.
Output (--out): videos/<sid>.mp4, gt/<sid>__s<k>.npz, segments/<sid>.json, meta/<sid>.json with sid = h3__<recording>.

  PYTHONPATH=<hot3d toolkit> $PY data/convert_hot3d_nogt.py --seq $RSI_DATA_RAW/HOT3D/P0004_1984bf99 \
      --out data/hot3d_nogt_ace
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

SIZE, F_PX, C_PX, SEG = 480, 207.8, 239.8, 81


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seq', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    from projectaria_tools.core import calibration, data_provider, sophus
    import imageio_ffmpeg
    seq, out = a.seq, a.out
    sid = 'h3__' + seq.name
    assert (seq / 'mano_hand_pose_trajectory.jsonl').stat().st_size == 0, f'{sid} has hand GT: use convert_ace.py'
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
    dst = calibration.CameraCalibration('camera-rgb', calibration.CameraModelType.LINEAR,
                                        np.array([F_PX, F_PX, C_PX, SIZE - 1 - C_PX], np.float64), T, SIZE, SIZE, None,
                                        float(cm['maxSolidAngle']), str(cm.get('serialNumber', '')))
    dst_cw = calibration.rotate_camera_calib_cw90deg(dst)
    fx, fy = dst_cw.get_focal_lengths(); cx, cy = dst_cw.get_principal_point()
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    assert np.allclose(K, [[F_PX, 0, C_PX], [0, F_PX, C_PX], [0, 0, 1]], atol=1e-3), K
    n = p.get_num_data(stream)
    vid = out / 'videos' / f'{sid}.mp4'
    enc = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                            '-s', f'{SIZE}x{SIZE}', '-r', '30', '-i', '-', '-c:v', 'libx264', '-threads', '4', '-preset', 'medium',
                            '-crf', '18', '-pix_fmt', 'yuv420p', str(vid)], stdin=subprocess.PIPE)
    for i in range(n):
        img, _ = p.get_image_data_by_index(stream, i)
        und = calibration.distort_by_calibration(img.to_numpy_array(), dst, src)
        und = np.ascontiguousarray(np.rot90(und, k=3))
        if und.ndim == 2:
            und = np.repeat(und[..., None], 3, axis=2)
        enc.stdin.write(und.astype(np.uint8).tobytes())
    enc.stdin.close()
    assert enc.wait() == 0, f'{sid}: ffmpeg failed'
    segs = []
    for k in range(n // SEG):
        name = f'{sid}__s{k:03d}'
        np.savez_compressed(out / 'gt' / f'{name}.npz', K=K.astype(np.float32), W=np.int32(SIZE), H=np.int32(SIZE),
                            valid=np.zeros((SEG, 2), bool), joints_cam=np.full((SEG, 2, 21, 3), np.nan, np.float32),
                            transl=np.full((SEG, 2, 3), np.nan, np.float32), rot=np.full((SEG, 2, 3, 3), np.nan, np.float32),
                            verts_cam=np.full((SEG, 2, 778, 3), np.nan, np.float32))
        segs.append(dict(name=name, start=k * SEG, end=(k + 1) * SEG))
    (out / 'segments' / f'{sid}.json').write_text(json.dumps(dict(sequence_id=sid, video=str(vid), frames=n, fps=30.0,
                                                                   K=K.tolist(), W=SIZE, H=SIZE, segments=segs), indent=1))
    meta = dict(sequence_id=sid, source=str(seq), frames=n, segments=len(segs), hand_frames_annotated=0, gt='none (teacher only)',
                K=K.tolist())
    (out / 'meta' / f'{sid}.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
