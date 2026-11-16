"""Shared detector front end of ACE A for HaMeR, Hamba, WildHands, OmniHands and InterWild: Detectron2 ViTDet-H person
boxes (score > 0.5, cascade test threshold 0.25, as hamer/demo.py) + ViTPose+-Huge whole-body keypoints. Stored per
person: its box, score and both hands' 21 keypoints (x, y, conf); each method derives its hand boxes with its own demo
rule (hamer/demo.py: the extent of the keypoints with conf > 0.5 when more than 3 are; hamba/demo.py: > 0.1). Use
boxes_from() below.
-> $BASELINES_OUT/frontend/<view>/<seq>.npz
  python frontend.py --view arctic_test [--shard i/n] [--seqs a,b]
Any number of workers may share a view: a sequence is claimed with <seq>.lock (O_EXCL, holds the pid; a lock whose
pid is gone is taken over)."""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from common import THIRD_PARTY, OUT, needed_frames, read_frames, sequences

sys.path[:0] = [str(THIRD_PARTY / 'official/HaMeR')]
VITCFG = THIRD_PARTY / 'dependencies/ViTPose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_huge_wholebody_256x192.py'
VITCKPT = THIRD_PARTY / 'weights/HaMeR/_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth'


def load():
    from detectron2.config import LazyConfig
    import hamer
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    from mmpose.apis import init_pose_model
    cfg = LazyConfig.load(str(Path(hamer.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'))
    cfg.train.init_checkpoint = str(THIRD_PARTY / 'weights/vitdet/model_final_f05665.pkl')   # Detectron2's ViTDet-H release
    for i in range(3):
        cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    return DefaultPredictor_Lazy(cfg), init_pose_model(str(VITCFG), str(VITCKPT), device='cuda:0')


@torch.no_grad()
def detect(det, imgs):
    """DefaultPredictor_Lazy.__call__ on a batch of BGR frames (same resize and format handling, one model call)"""
    import detectron2.data.transforms as T
    inputs = []
    for img in imgs:
        x = img[:, :, ::-1] if det.input_format == 'RGB' else img
        im = det.aug(T.AugInput(x)).apply_image(x)
        inputs.append({'image': torch.as_tensor(im.astype('float32').transpose(2, 0, 1)), 'height': x.shape[0], 'width': x.shape[1]})
    return [o['instances'] for o in det.model(inputs)]


@torch.no_grad()
def run(det, pose, imgs):
    from mmpose.apis import inference_top_down_pose_model, process_mmdet_results
    out = []
    for img, inst in zip(imgs, detect(det, imgs)):
        ok = (inst.pred_classes == 0) & (inst.scores > 0.5)
        pb = inst.pred_boxes.tensor[ok].cpu().numpy(); ps = inst.scores[ok].cpu().numpy()
        kps = np.zeros((0, 2, 21, 3), np.float32)
        if len(pb):
            res, _ = inference_top_down_pose_model(pose, img, person_results=process_mmdet_results([np.concatenate([pb, ps[:, None]], 1)], 1),
                                                   bbox_thr=0.5, format='xyxy')
            pb = np.array([r['bbox'][:4] for r in res], np.float32).reshape(-1, 4); ps = np.array([r['bbox'][4] for r in res], np.float32)
            kps = np.array([[r['keypoints'][-42:-21], r['keypoints'][-21:]] for r in res], np.float32).reshape(-1, 2, 21, 3)
        out.append((pb, ps, kps))
    return out


def boxes_from(fe, thr=0.5):
    """front-end npz -> list of (frame, side, box xyxy, keypoints (21,3), score = mean conf of the used keypoints)"""
    out = []
    for f, kp2 in zip(fe['person_frame'], fe['person_hands']):
        for side in (0, 1):
            kp = kp2[side]; v = kp[:, 2] > thr
            if v.sum() > 3:
                out.append((int(f), side, np.array([kp[v, 0].min(), kp[v, 1].min(), kp[v, 0].max(), kp[v, 1].max()], np.float32),
                            kp, float(kp[v, 2].mean())))
    return out


def claim(lock):
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd)
            return True
        except FileExistsError:
            try:
                pid = int(Path(lock).read_text() or 0)
            except (OSError, ValueError):
                pid = 0
            if pid and Path(f'/proc/{pid}').exists():
                return False
            Path(lock).unlink(missing_ok=True)          # stale: its worker is gone
    return False


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--view', required=True); ap.add_argument('--shard', default='0/1')
    ap.add_argument('--seqs', default=''); ap.add_argument('--batch', type=int, default=2); a = ap.parse_args()
    torch.set_num_threads(8)
    # TF32 matmuls (ViT-H in full fp32 is ~2 s/frame on a shared GPU)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    od = OUT / 'frontend' / a.view; od.mkdir(parents=True, exist_ok=True)
    det = pose = None
    for m in sequences(a.view, a.shard, a.seqs):
        lock = od / f"{m['name']}.lock"
        if (od / f"{m['name']}.npz").is_file() or not claim(lock):
            continue
        if det is None:
            det, pose = load()
        t0 = time.time(); need = needed_frames(m); frames = read_frames(m['video_path'], need[-1] + 1)
        P = dict(frame=[], box=[], score=[], hands=[])
        for i in range(0, len(need), a.batch):
            fs = need[i:i + a.batch]
            for f, (pb, ps, kps) in zip(fs, run(det, pose, [frames[f] for f in fs])):
                P['frame'] += [f] * len(pb); P['box'] += list(pb); P['score'] += list(ps); P['hands'] += list(kps)
        np.savez_compressed(od / f"{m['name']}.part.npz", person_frame=np.array(P['frame'], np.int32),
                            person_box=np.array(P['box'], np.float32).reshape(-1, 4), person_score=np.array(P['score'], np.float32),
                            person_hands=np.array(P['hands'], np.float32).reshape(-1, 2, 21, 3), frames=np.array(need, np.int32))
        os.replace(od / f"{m['name']}.part.npz", od / f"{m['name']}.npz")      # readers never see a partial file
        nh = len(boxes_from(np.load(od / f"{m['name']}.npz")))
        print(f"{a.view} {m['name']}: {len(need)} frames, {len(P['frame'])} persons, {nh} hands (conf 0.5), "
              f"{time.time() - t0:.0f}s ({len(need) / (time.time() - t0):.1f} fps)", flush=True)
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
