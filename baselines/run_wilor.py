"""WiLoR (ACE A: released wilor_final.ckpt with its own YOLO detector), official repository pipeline (WiLoR/demo.py:
detector conf 0.3, every detection, ViTDetDataset rescale_factor 2.0) on our test views -> preds/wilor/<view>/<segment>.npz.
Camera: GT intrinsics (common.crop_cam_to_K) instead of the demo's 5000/256*max(W,H) focal and image centre. Joints:
WiLoR's MANO (rotation matrices, no mean pose) through ace_eval's MANO forward (flat_hand_mean=True), checked against
WiLoR's own pred_vertices once per process. One hand per side and frame: the most confident detection.
  python run_wilor.py --view arctic_test [--shard i/n] [--seqs a,b]"""
import argparse
import os
import sys
import time

import numpy as np
import torch

from common import THIRD_PARTY, OUT, crop_cam_to_K, done, hands_to_pred, mano, needed_frames, read_frames, rotvec, sequences, write_segments

WL = THIRD_PARTY / 'official/WiLoR'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--view', required=True); ap.add_argument('--shard', default='0/1')
    ap.add_argument('--seqs', default=''); ap.add_argument('--batch', type=int, default=48); a = ap.parse_args()
    torch.set_num_threads(8)
    os.chdir(WL); sys.path[:0] = [str(WL)]
    from wilor.models import load_wilor
    from wilor.datasets.vitdet_dataset import ViTDetDataset
    from wilor.utils import recursive_to
    from ultralytics import YOLO
    from ace_eval.mano import forward
    model = det = None; checked = False
    for m in sequences(a.view, a.shard, a.seqs):
        if done('wilor', a.view, m):
            continue
        if model is None:
            model, cfg = load_wilor(checkpoint_path='./pretrained_models/wilor_final.ckpt', cfg_path='./pretrained_models/model_config.yaml')
            model = model.cuda().eval(); det = YOLO('./pretrained_models/detector.pt')
        t0 = time.time(); need = needed_frames(m); F = need[-1] + 1
        frames = read_frames(m['video_path'], F)
        hb = []
        for i in range(0, len(need), 16):
            fs = need[i:i + 16]
            for f, r in zip(fs, det([frames[f] for f in fs], conf=0.3, verbose=False)):
                for b, c, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                    hb.append((f, int(c), b.astype(np.float32), float(cf)))
        hands = []
        for i in range(0, len(hb), a.batch):
            chunk = hb[i:i + a.batch]
            items = [ViTDetDataset(cfg, frames[f], box[None], np.array([side], np.float32), rescale_factor=2.0)[0] for f, side, box, _ in chunk]
            batch = recursive_to(torch.utils.data.default_collate(items), 'cuda')
            with torch.no_grad():
                out = model(batch)
            cam = out['pred_cam'].float().cpu().numpy().copy(); r = batch['right'].cpu().numpy(); cam[:, 1] *= 2 * r - 1
            tau = crop_cam_to_K(cam, batch['box_center'].float().cpu().numpy(), batch['box_size'].float().cpu().numpy(), m['Kmat'])
            mp = {k: v.float().cpu().numpy() for k, v in out['pred_mano_params'].items()}
            if not checked:
                _, V = forward(mano(True), 1, rotvec(mp['global_orient'][:, 0]), rotvec(mp['hand_pose']), mp['betas'])
                err = float(np.abs(V - out['pred_vertices'].float().cpu().numpy()).max())
                assert err < 1e-4, f'MANO mismatch {err} m'
                print('mesh check', err, flush=True); checked = True
            for n, (f, side, box, cf) in enumerate(chunk):
                hands.append(dict(f=f, side=side, go=mp['global_orient'][n, 0], hp=mp['hand_pose'][n], betas=mp['betas'][n], tau=tau[n], score=cf))
        write_segments('wilor', a.view, m, hands_to_pred(F, hands))
        print(f"{a.view} {m['name']}: {len(need)} frames, {len(hb)} detections, {time.time() - t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
