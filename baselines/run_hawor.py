"""HaWoR (ACE A: released hawor.ckpt; scored on its camera-space front end with SLAM and infilling disabled; GT
intrinsics) on our test views, one run per 81-frame segment -> preds/hawor/<view>/<segment>.npz.
Steps as scripts/scripts_test_video (detect_track_video + hawor_motion_estimation), minus the mask rendering that only
feeds SLAM / infilling: the segment's frames as JPEGs (the demo extracts JPEGs with ffmpeg), WiLoR's YOLO hand detector
with tracking (conf 0.2, autocast, one right and one left per frame), tracks merged into left / right by majority
handedness, contiguous detection runs (parse_chunks, min_len 1), model.inference with img_focal = GT fx and img_center =
GT principal point, left hands un-flipped as the demo does. Joints / vertices: HaWoR's own run_mano / run_mano_left
(OpenPose order with the smplx fingertip vertices = ACE's convention; left MANO with the shapedirs fix), camera frame.
  python run_hawor.py --view arctic_test [--shard i/n] [--seqs a,b]"""
import argparse
import os
import shutil
import sys
import tempfile
import time

import cv2
import numpy as np
import torch

from common import THIRD_PARTY, OUT, direct_to_pred, read_frames, sequences

HW = THIRD_PARTY / 'official/HaWoR'
LW = THIRD_PARTY / 'weights/HaWoR'         # HaWoR's weights + hawor/model_config.yaml (README)


def detect_track(det_path, imgfiles, thresh=0.2):
    """lib/pipeline/tools.py detect_track, a fresh tracker per segment"""
    from ultralytics import YOLO
    from torch.cuda.amp import autocast
    model = YOLO(det_path)
    tracks = {}
    for t, p in enumerate(imgfiles):
        img = cv2.imread(p)
        with torch.no_grad(), autocast():
            r = model.track(img, conf=thresh, persist=True, verbose=False)[0]
            boxes = r.boxes.xyxy.cpu().numpy(); confs = r.boxes.conf.cpu().numpy(); hand = r.boxes.cls.cpu().numpy()
            tid = r.boxes.id.cpu().numpy() if r.boxes.id is not None else [-1] * len(boxes)
            boxes = np.hstack([boxes, confs[:, None]])
            fr = fl = False
            for i, b in enumerate(boxes):
                k = (10000 if hand[[i]] > 0 else 5000) if tid[i] == -1 else tid[i]
                subj = dict(frame=t, det=True, det_box=boxes[[i]], det_handedness=hand[[i]])
                if (not fr and hand[[i]] > 0) or (not fl and hand[[i]] == 0):
                    tracks.setdefault(k, []).append(subj)
                    if hand[[i]] > 0:
                        fr = True
                    elif hand[[i]] == 0:
                        fl = True
    return tracks


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--view', required=True); ap.add_argument('--shard', default='0/1')
    ap.add_argument('--seqs', default=''); a = ap.parse_args()
    torch.set_num_threads(8)
    os.chdir(HW); sys.path[:0] = [str(HW)]
    from scripts.scripts_test_video.hawor_video import load_hawor
    from lib.pipeline.tools import parse_chunks
    from lib.eval_utils.custom_utils import interpolate_bboxes
    from hawor.utils.process import run_mano, run_mano_left
    from hawor.utils.rotation import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis
    model, _ = load_hawor(str(LW / 'hawor/checkpoints/hawor.ckpt')); model = model.cuda().eval()
    od = OUT / 'preds/hawor' / a.view; od.mkdir(parents=True, exist_ok=True)
    tmp_root = OUT / 'tmp/hawor'; tmp_root.mkdir(parents=True, exist_ok=True)
    checked = False
    for m in sequences(a.view, a.shard, a.seqs):
        todo = [s for s in m['segments'] if not (od / f"{s['name']}.npz").is_file()]
        if not todo:
            continue
        t0 = time.time(); frames = read_frames(m['video_path'], max(s['end'] for s in todo)); K = m['Kmat']
        for s in todo:
            F = s['end'] - s['start']
            td = tempfile.mkdtemp(dir=tmp_root)
            try:
                imgfiles = []
                for i in range(F):
                    p = os.path.join(td, f'{i:04d}.jpg'); cv2.imwrite(p, frames[s['start'] + i]); imgfiles.append(p)
                imgfiles = np.array(imgfiles)
                tracks = detect_track(str(LW / 'external/detector.pt'), imgfiles, thresh=0.2)
                lt, rt = [], []
                for k in tracks:
                    trk = tracks[k]
                    valid = np.array([t['det'] for t in trk]); isr = np.concatenate([t['det_handedness'] for t in trk])[valid]
                    (lt if isr.sum() / len(isr) < 0.5 else rt).extend(trk)
                entries = []
                for side, trk in ((0, sorted(lt, key=lambda x: x['frame'])), (1, sorted(rt, key=lambda x: x['frame']))):
                    valid = np.array([t['det'] for t in trk])
                    if valid.sum() < 2:
                        continue
                    boxes = np.concatenate([t['det_box'] for t in trk])
                    nz = np.where(np.any(boxes != 0, axis=1))[0]
                    boxes[nz[0]:nz[-1] + 1] = interpolate_bboxes(boxes[nz[0]:nz[-1] + 1]); valid[nz[0]:nz[-1] + 1] = True
                    boxes = boxes[nz[0]:nz[-1] + 1]
                    isr = np.concatenate([t['det_handedness'] for t in trk])[valid]
                    right = isr.sum() / len(isr) >= 0.5
                    frame = np.array([t['frame'] for t in trk])[valid]
                    for fck, bck in zip(*parse_chunks(frame, boxes, min_len=1)):
                        res = model.inference(imgfiles[fck], bck, img_focal=float(K[0, 0]), img_center=[float(K[0, 2]), float(K[1, 2])],
                                              do_flip=not right)
                        root = rotation_matrix_to_angle_axis(res['pred_rotmat'][None, :, 0]); pose = rotation_matrix_to_angle_axis(res['pred_rotmat'][None, :, 1:])
                        if not right:
                            root[..., 1:] *= -1; pose[..., 1:] *= -1
                        trans = res['pred_trans'][None, :, 0]; betas = res['pred_shape'][None, :]
                        out = (run_mano if right else run_mano_left)(trans, root, pose, betas=betas)
                        J = out['joints'][0].cpu().numpy(); V = out['vertices'][0].cpu().numpy()
                        if right and not checked:                # HaWoR's joints == ace_eval's MANO joints (ACE order)
                            from common import mano
                            from ace_eval.mano import forward
                            Jo, _ = forward(mano(True), 1, root[0].cpu().numpy(), pose[0].cpu().numpy().reshape(-1, 15, 3), betas[0].cpu().numpy())
                            err = float(np.abs(Jo + trans[0].cpu().numpy()[:, None] - J).max())
                            assert err < 1e-4, f'joint convention mismatch {err} m'
                            print('joint check', err, flush=True); checked = True
                        R = angle_axis_to_rotation_matrix(root)[0].cpu().numpy(); tr = trans[0].cpu().numpy()
                        for n, f in enumerate(fck):
                            entries.append(dict(f=int(f), side=1 if right else 0, J=J[n], V=V[n], tau=tr[n], R=R[n], score=1.0))
                P = direct_to_pred(F, entries)
                np.savez_compressed(od / f"{s['name']}.npz", **P)
            finally:
                shutil.rmtree(td, ignore_errors=True)
        print(f"{a.view} {m['name']}: {len(todo)} segments, {time.time() - t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
