"""HaMeR (ACE A: released hamer.ckpt behind the ViTDet-H + ViTPose+-Huge front end, hamer/demo.py rules: hand keypoint
conf > 0.5, rescale factor 2.0) on our test views -> evaluator predictions (preds/hamer/<view>/<segment>.npz).
Camera: pred_cam -> translation under the clip's GT intrinsics (common.crop_cam_to_K) instead of the demo's
5000/256*max(W,H) focal and image centre. Joints: HaMeR's MANO (rotation matrices, no mean pose) through ace_eval's
MANO forward with flat_hand_mean=True; checked against HaMeR's own pred_vertices on the first batch of every sequence.
Runs in work/hamer: the model_config.yaml we used names its MANO files '_DATA/data/...' and HaMeR prefixes './_DATA'.
  python run_hamer.py --view arctic_test [--shard i/n] [--seqs a,b]"""
import argparse
import contextlib
import io
import os
import sys
import time

import numpy as np
import torch

from common import THIRD_PARTY, OUT, crop_cam_to_K, done, fwd, hands_to_pred, mano, needed_frames, read_frames, rotvec, sequences, workdir, write_segments
from frontend import boxes_from

sys.path[:0] = [str(THIRD_PARTY / 'official/HaMeR')]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--view', required=True); ap.add_argument('--shard', default='0/1')
    ap.add_argument('--seqs', default=''); ap.add_argument('--batch', type=int, default=48); a = ap.parse_args()
    os.chdir(workdir('hamer', {'_DATA/_DATA': THIRD_PARTY / 'weights/HaMeR/_DATA'}))
    from hamer.models import load_hamer
    from hamer.datasets.vitdet_dataset import ViTDetDataset
    from hamer.utils import recursive_to
    model = None
    for m in sequences(a.view, a.shard, a.seqs):
        fe_p = OUT / 'frontend' / a.view / f"{m['name']}.npz"
        if done('hamer', a.view, m) or not fe_p.is_file():
            continue
        if model is None:
            model, cfg = load_hamer(str(THIRD_PARTY / 'weights/HaMeR/_DATA/hamer_ckpts/checkpoints/hamer.ckpt'))
            model = model.cuda().eval()
        t0 = time.time(); need = needed_frames(m); F = need[-1] + 1
        frames = read_frames(m['video_path'], F)
        hb = boxes_from(np.load(fe_p), 0.5)
        hands, checked = [], False
        for i in range(0, len(hb), a.batch):
            chunk = hb[i:i + a.batch]
            items = []
            with contextlib.redirect_stdout(io.StringIO()):          # vitdet_dataset prints every crop's downsampling factor
                for f, side, box, kp, sc in chunk:
                    items.append(ViTDetDataset(cfg, frames[f], box[None], np.array([side]), rescale_factor=2.0)[0])
            batch = recursive_to(torch.utils.data.default_collate(items), 'cuda')
            with torch.no_grad():
                out = model(batch)
            cam = out['pred_cam'].float().cpu().numpy().copy()
            r = batch['right'].cpu().numpy()
            cam[:, 1] *= 2 * r - 1
            tau = crop_cam_to_K(cam, batch['box_center'].float().cpu().numpy(), batch['box_size'].float().cpu().numpy(), m['Kmat'])
            mp = {k: v.float().cpu().numpy() for k, v in out['pred_mano_params'].items()}
            if not checked:                                           # our MANO forward == HaMeR's own mesh
                _, V = fwd(mano(True), 1, rotvec(mp['global_orient'][:, 0]), rotvec(mp['hand_pose']), mp['betas'])
                err = float(np.abs(V - out['pred_vertices'].float().cpu().numpy()).max())
                assert err < 1e-4, f'MANO mismatch {err} m'
                checked = True
            for n, (f, side, box, kp, sc) in enumerate(chunk):
                hands.append(dict(f=f, side=side, go=mp['global_orient'][n, 0], hp=mp['hand_pose'][n], betas=mp['betas'][n],
                                  tau=tau[n], score=sc))
        write_segments('hamer', a.view, m, hands_to_pred(F, hands))
        print(f"{a.view} {m['name']}: {len(need)} frames, {len(hb)} hand boxes, {time.time() - t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
