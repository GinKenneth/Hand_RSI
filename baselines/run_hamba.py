"""Hamba (ACE A: released hamba.ckpt behind the ViTDet-H + ViTPose+-Huge front end), hamba/demo.py pipeline: hand boxes
from the hand keypoints with conf > 0.1 (more than 3), Hamba's ViTDetDataset (rescale_factor 2.0) which re-detects the
crop's 2D keypoints with MediaPipe Hands (a fresh tracker per frame, as the demo builds one dataset per image) and drops
the hand when MediaPipe finds none -> preds/hamba/<view>/<segment>.npz. One MediaPipe Hands object serves every frame
and is reset() at each frame's dataset (graph closed and restarted = the demo's fresh tracker per image; building and
closing one costs ~16 s here); the first frame is checked against a really fresh tracker. Camera: GT intrinsics (common.crop_cam_to_K).
Joints: Hamba's MANO (HaMeR's wrapper: rotation matrices, no mean pose) through ace_eval's MANO forward, checked
against Hamba's own pred_vertices. Runs in work/hamba (Hamba resolves ./_DATA/data/mano relative to the cwd).
  python run_hamba.py --view arctic_test [--shard i/n] [--seqs a,b]"""
import argparse
import contextlib
import io
import os
import sys
import time
import types

import numpy as np
import torch

from common import THIRD_PARTY, OUT, crop_cam_to_K, done, fwd, hands_to_pred, mano, needed_frames, read_frames, rotvec, sequences, workdir, write_segments
from frontend import boxes_from

HB = THIRD_PARTY / 'official/Hamba'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--view', required=True); ap.add_argument('--shard', default='0/1')
    ap.add_argument('--seqs', default=''); ap.add_argument('--batch', type=int, default=32); a = ap.parse_args()
    torch.set_num_threads(8)
    os.chdir(workdir('hamba', {'_DATA': THIRD_PARTY / 'weights/HaMeR/_DATA'})); sys.path[:0] = [str(HB)]   # Hamba reads ./_DATA/data/mano
    from hamba.models import load_hamba
    import hamba.datasets.vitdet_dataset as hvd
    from hamba.datasets.vitdet_dataset import ViTDetDataset
    KW = dict(min_detection_confidence=0.2, min_tracking_confidence=0.2)      # the dataset's own arguments
    real_hands = hvd.mp_hands.Hands; shared = real_hands(**KW)

    def fresh_hands(**kw):
        assert kw == KW, kw
        shared.reset(); return shared
    hvd.mp_hands = types.SimpleNamespace(Hands=fresh_hands)
    mp_checked = False
    from hamba.utils import recursive_to
    model = None; checked = False
    for m in sequences(a.view, a.shard, a.seqs):
        fe_p = OUT / 'frontend' / a.view / f"{m['name']}.npz"
        if done('hamba', a.view, m) or not fe_p.is_file():
            continue
        if model is None:
            model, cfg = load_hamba(str(THIRD_PARTY / 'weights/Hamba/checkpoints/hamba.ckpt')); model = model.cuda().eval()
        t0 = time.time(); need = needed_frames(m); F = need[-1] + 1
        frames = read_frames(m['video_path'], F)
        hb = boxes_from(np.load(fe_p), 0.1)
        byf = {}
        for h in hb:
            byf.setdefault(h[0], []).append(h)
        items, meta = [], []
        n_drop = 0
        for f in sorted(byf):
            hs = byf[f]
            with contextlib.redirect_stdout(io.StringIO()):          # the dataset prints shapes / MediaPipe handedness
                ds = ViTDetDataset(cfg, frames[f], np.stack([h[2] for h in hs]), np.array([h[1] for h in hs]), rescale_factor=2.0,
                                   keypoints_2d_arr=np.stack([h[3] for h in hs]))
                cur = [ds[i] for i in range(len(hs))]           # in order, from the reset() state
                for it, h in zip(cur, hs):
                    if it['is_valid'] == -1:
                        n_drop += 1; continue
                    items.append(it); meta.append(h)
            if not mp_checked:          # reset() == a new tracker: redo this frame with a really fresh one
                hvd.mp_hands = types.SimpleNamespace(Hands=real_hands)
                with contextlib.redirect_stdout(io.StringIO()):
                    ds2 = ViTDetDataset(cfg, frames[f], np.stack([h[2] for h in hs]), np.array([h[1] for h in hs]), rescale_factor=2.0,
                                        keypoints_2d_arr=np.stack([h[3] for h in hs]))
                    it2 = [ds2[i] for i in range(len(hs))]
                ds2.mp_hand.close(); hvd.mp_hands = types.SimpleNamespace(Hands=fresh_hands)
                for x, y in zip(cur, it2):
                    assert x['is_valid'] == y['is_valid'] and np.array_equal(x['keypoints_2d'], y['keypoints_2d']), 'reset() != fresh tracker'
                print('mediapipe reset check passed on', len(hs), 'hands', flush=True); mp_checked = True
        hands = []
        for i in range(0, len(items), a.batch):
            batch = recursive_to(torch.utils.data.default_collate(items[i:i + a.batch]), 'cuda')
            with torch.no_grad():
                out = model(batch)
            cam = out['pred_cam'].float().cpu().numpy().copy(); r = batch['right'].cpu().numpy(); cam[:, 1] *= 2 * r - 1
            tau = crop_cam_to_K(cam, batch['box_center'].float().cpu().numpy(), batch['box_size'].float().cpu().numpy(), m['Kmat'])
            mp = {k: v.float().cpu().numpy() for k, v in out['pred_mano_params'].items()}
            if not checked:
                _, V = fwd(mano(True), 1, rotvec(mp['global_orient'][:, 0]), rotvec(mp['hand_pose']), mp['betas'])
                err = float(np.abs(V - out['pred_vertices'].float().cpu().numpy()).max())
                assert err < 1e-4, f'MANO mismatch {err} m'
                print('mesh check', err, flush=True); checked = True
            for n, (f, side, box, kp, sc) in enumerate(meta[i:i + a.batch]):
                hands.append(dict(f=f, side=side, go=mp['global_orient'][n, 0], hp=mp['hand_pose'][n], betas=mp['betas'][n], tau=tau[n], score=sc))
        write_segments('hamba', a.view, m, hands_to_pred(F, hands))
        print(f"{a.view} {m['name']}: {len(need)} frames, {len(hb)} boxes, {n_drop} dropped by MediaPipe, {time.time() - t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
