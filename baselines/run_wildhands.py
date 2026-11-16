"""WildHands (ACE A: released wildhands.ckpt behind the ViTDet-H + ViTPose+-Huge front end; GT intrinsics, rescaled with
the image) on our test views -> preds/wildhands/<view>/<segment>.npz. As WildHands/demo.py: the frame is padded to a
square around its centre and resized to 840 px (render_res), the intrinsics follow, hand boxes are the demo's rule
(keypoint conf > 0.5, more than 3) mapped into that image, WildHandsDataset with rescale_factor 1.75. Output: the hand's
own MANO (16 rotation matrices, betas; left hands with the left MANO) + cam_t, posed through ace_eval's MANO forward;
the flat_hand_mean that reproduces WildHands' own v3d.cam is picked on the first batch and asserted.
  python run_wildhands.py --view arctic_test [--shard i/n] [--seqs a,b]"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

from common import THIRD_PARTY, OUT, done, fwd, hands_to_pred, mano, needed_frames, read_frames, rotvec, sequences, write_segments
from frontend import boxes_from

WH = THIRD_PARTY / 'official/WildHands'
sys.path[:0] = [str(WH), str(WH / 'third-party')]
RES = 840


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--view', required=True); ap.add_argument('--shard', default='0/1')
    ap.add_argument('--seqs', default=''); ap.add_argument('--batch', type=int, default=32); a = ap.parse_args()
    torch.set_num_threads(8)
    from ace_eval.mano import MANO_DIR
    os.environ['WILDHANDS_MANO_DIR'] = os.path.join(MANO_DIR, 'mano')   # WildHands loads MANO_{RIGHT,LEFT}.pkl from here
    from wildhands.configs.parser import construct_args
    from wildhands.models.wrapper import WildHandsWrapper as Wrapper
    from wildhands.datasets.dataset import WildHandsDataset
    import wildhands.common.data_utils as data_utils
    from hamer.utils import recursive_to
    model, flat, left_fix = None, None, False
    for m in sequences(a.view, a.shard, a.seqs):
        fe_p = OUT / 'frontend' / a.view / f"{m['name']}.npz"
        if done('wildhands', a.view, m) or not fe_p.is_file():
            continue
        if model is None:
            cfg = construct_args(); model = Wrapper(cfg)
            ck = torch.load(os.environ.get('WILDHANDS_CKPT', str(THIRD_PARTY / 'weights/WildHands/wildhands.ckpt')), map_location='cpu')
            model.load_state_dict({k: v for k, v in ck['state_dict'].items() if not any(r in k for r in ('head_o', 'arti_head', 'grasp_classifier'))})
            model = model.cuda().eval()
        t0 = time.time(); need = needed_frames(m); F = need[-1] + 1
        frames = read_frames(m['video_path'], F)
        H, W = frames[0].shape[:2]; R0 = max(W, H); sc = RES / R0
        K = m['Kmat']
        intrx = np.array([[K[0, 0] * sc, 0, (K[0, 2] - (W - R0) / 2) * sc], [0, K[1, 1] * sc, (K[1, 2] - (H - R0) / 2) * sc], [0, 0, 1]])
        hb = boxes_from(np.load(fe_p), 0.5)
        sq = {}
        hands = []
        for i in range(0, len(hb), a.batch):
            chunk = hb[i:i + a.batch]; items = []
            for f, side, box, kp, s_ in chunk:
                if f not in sq:
                    img = data_utils.generate_patch_image_clean(frames[f][:, :, ::-1], [W / 2, H / 2, R0, R0], 1.0, 0.0, [RES, RES], cv2.INTER_CUBIC)[0]
                    sq = {f: img.clip(0, 255).astype(np.uint8)}          # RGB, as demo.py's `img`
                b = np.array([(box[0] - (W - R0) / 2) * sc, (box[1] - (H - R0) / 2) * sc, (box[2] - (W - R0) / 2) * sc, (box[3] - (H - R0) / 2) * sc], np.float32)
                ds = WildHandsDataset(cfg, sq[f], b[None], np.array([side]), focal_length=float(K[0, 0] * sc), rescale_factor=1.75, intrx=intrx)
                items.append(ds[0])
            batch = recursive_to(torch.utils.data.default_collate(items), 'cuda')
            with torch.no_grad():
                out = model(batch)
            o = {k: v.float().cpu().numpy() for k, v in out.items() if torch.is_tensor(v)}
            if flat is None:                                    # which MANO mean reproduces WildHands' own mesh
                errs = {}
                for fl in (False, True):
                    _, V = fwd(mano(fl), 1, rotvec(o['pred.pose.r'][:, 0]), rotvec(o['pred.pose.r'][:, 1:]), o['pred.beta.r'])
                    errs[fl] = float(np.abs(V + o['pred.cam_t.r'][:, None] - o['pred.v3d.cam.r']).max())
                flat = min(errs, key=errs.get)
                for lf in (False, True):
                    _, VL = fwd(mano(flat, lf), 0, rotvec(o['pred.pose.l'][:, 0]), rotvec(o['pred.pose.l'][:, 1:]), o['pred.beta.l'])
                    errs[f'left_fix={lf}'] = float(np.abs(VL + o['pred.cam_t.l'][:, None] - o['pred.v3d.cam.l']).max())
                left_fix = errs['left_fix=True'] < errs['left_fix=False']
                assert errs[flat] < 1e-4 and errs[f'left_fix={left_fix}'] < 1e-4, f'MANO mismatch {errs}'
                print('flat_hand_mean', flat, 'left_fix', left_fix, errs, flush=True)
            for n, (f, side, box, kp, s_) in enumerate(chunk):
                h = 'r' if side else 'l'
                hands.append(dict(f=f, side=side, go=o[f'pred.pose.{h}'][n, 0], hp=o[f'pred.pose.{h}'][n, 1:], betas=o[f'pred.beta.{h}'][n],
                                  tau=o[f'pred.cam_t.{h}'][n], score=s_, native=True))
        write_segments('wildhands', a.view, m, hands_to_pred(F, hands, flat, left_fix))
        print(f"{a.view} {m['name']}: {len(need)} frames, {len(hb)} hand boxes, {time.time() - t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
