"""InterWild (ACE A: the egocentric release behind the ViTDet-H + ViTPose+-Huge front end, its internal BoxNet bypassed
at run time so that the external hand boxes are injected) on our test views -> preds/interwild/<view>/<segment>.npz.
Checkpoints: --ckpt ego = IH26M(H+M)+MSCOCO+ReInterHand(Ego_cameras) (the ACE row; weights/InterWild/ego/snapshot_6.pth),
--ckpt base = IH26M(H+M)+MSCOCO (pipeline check only) -> preds/interwild_base.
As demo/demo.py: the whole frame, process_bbox (aspect 384:512, x1.25), generate_patch_image to 512x384, /255.
Box injection: per frame and side the front end's best hand box (hamer/demo.py rule: keypoint conf > 0.5 on more than 3;
highest mean conf) replaces BoxNet's centre and size (same units: output_body_hm space), so restore_bbox applies
InterWild's own square aspect and x2.0 extension; a side without a front-end box keeps BoxNet's guess as the second crop
the model needs, and its output is discarded. Two-hand mode (rroot_cam + rel_trans for the left) when both sides are
injected and their boxes overlap (the demo's IoU > 0), else each hand's own root_cam.
Camera: each hand crop has InterWild's virtual camera (focal 5000, centre 128 px); with hand2orig and bb2img it is a
camera of focal 5000*s at the crop centre in the original image. The root keeps its image position and apparent size
under the clip's GT K: Z' = Z * fx / (5000 s), (X', Y') from the projected root (common: GT K for every method).
Joints: InterWild's MANO parameters (native left hand) through ace_eval's MANO forward; flat_hand_mean and the left
shapedirs fix are picked by matching InterWild's own root-relative meshes, asserted.
  python run_interwild.py --view arctic_test [--shard i/n] [--seqs a,b] [--ckpt ego|base]"""
import argparse
import sys
import time

import cv2
import numpy as np
import torch

from common import THIRD_PARTY, OUT, done, fwd, hands_to_pred, mano, needed_frames, read_frames, sequences, write_segments
from frontend import boxes_from

IW = THIRD_PARTY / 'official/InterWild'
CKPTS = {'ego': THIRD_PARTY / 'weights/InterWild/ego/snapshot_6.pth', 'base': THIRD_PARTY / 'weights/InterWild/base/snapshot_6.pth'}   # sha256 in README


def aa2R(x):
    return np.stack([cv2.Rodrigues(np.asarray(v, np.float64))[0] for v in np.asarray(x).reshape(-1, 3)])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--view', required=True); ap.add_argument('--shard', default='0/1')
    ap.add_argument('--seqs', default=''); ap.add_argument('--ckpt', default='ego', choices=list(CKPTS))
    ap.add_argument('--batch', type=int, default=4); a = ap.parse_args()
    torch.set_num_threads(8)
    meth = 'interwild' if a.ckpt == 'ego' else 'interwild_base'
    sys.path[:0] = [str(IW / 'main'), str(IW / 'data'), str(IW / 'common')]
    from config import cfg
    from model import get_model
    from utils.preprocessing import process_bbox, generate_patch_image, get_iou
    from torch.nn.parallel.data_parallel import DataParallel
    hm = cfg.output_body_hm_shape[2] / cfg.input_img_shape[1]           # patch px -> BoxNet heatmap units (48 / 384)
    assert abs(hm - cfg.output_body_hm_shape[1] / cfg.input_img_shape[0]) < 1e-9
    model, inj, stash, fit = None, {}, {}, None
    for m in sequences(a.view, a.shard, a.seqs):
        fe_p = OUT / 'frontend' / a.view / f"{m['name']}.npz"
        if done(meth, a.view, m) or not fe_p.is_file():
            continue
        if model is None:
            model = DataParallel(get_model('test')).cuda()
            r = model.load_state_dict(torch.load(str(CKPTS[a.ckpt]), map_location='cpu')['network'], strict=False)
            print('ckpt', CKPTS[a.ckpt], 'missing', len(r.missing_keys), r.missing_keys[:6], 'unexpected', len(r.unexpected_keys), flush=True)
            model.eval(); net = model.module
            box_fwd, roi_fwd = net.body_box_net.forward, net.hand_roi_net.forward

            def box_forward(feat):                       # BoxNet bypass: external centres / sizes where we have them
                rc, rs, lc, ls, rcf, lcf = box_fwd(feat)
                rc, rs = torch.where(inj['rm'][:, None], inj['rc'], rc), torch.where(inj['rm'][:, None], inj['rs'], rs)
                lc, ls = torch.where(inj['lm'][:, None], inj['lc'], lc), torch.where(inj['lm'][:, None], inj['ls'], ls)
                return rc, rs, lc, ls, rcf, lcf

            def roi_forward(img, rb, lb):                # keep hand2orig (crop px -> patch px) for the camera
                out = roi_fwd(img, rb, lb); stash['h2o'] = out[2].detach().float().cpu().numpy(); return out
            net.body_box_net.forward = box_forward; net.hand_roi_net.forward = roi_forward
        t0 = time.time(); need = needed_frames(m); F = need[-1] + 1; K = m['Kmat']
        frames = read_frames(m['video_path'], F)
        H, W = frames[0].shape[:2]
        bbox = process_bbox([0, 0, W, H], W, H)
        _, img2bb, bb2img = generate_patch_image(frames[0][:, :, ::-1].astype(np.float32), bbox, 1.0, 0.0, False, cfg.input_img_shape)
        B2I = np.vstack([bb2img, [0, 0, 1]])
        best = {}
        for f, side, box, kp, sc in boxes_from(np.load(fe_p), 0.5):
            if (f, side) not in best or sc > best[(f, side)][1]:
                best[(f, side)] = (box, sc)
        fl = sorted({f for f, _ in best})
        hands = []
        for i in range(0, len(fl), a.batch):
            fs = fl[i:i + a.batch]; n = len(fs)
            imgs = []
            for f in fs:
                p = generate_patch_image(frames[f][:, :, ::-1].astype(np.float32), bbox, 1.0, 0.0, False, cfg.input_img_shape)[0]
                imgs.append(torch.from_numpy(p.astype(np.float32).transpose(2, 0, 1).copy()) / 255)   # = ToTensor()(img) / 255
            for s, k in ((1, 'r'), (0, 'l')):
                c = np.zeros((n, 2), np.float32); z = np.ones((n, 2), np.float32); msk = np.zeros(n, bool)
                for j, f in enumerate(fs):
                    if (f, s) in best:
                        b = best[(f, s)][0]
                        q = (img2bb @ np.array([[b[0], b[1], 1], [b[2], b[3], 1]], np.float64).T).T * hm
                        c[j] = q.mean(0); z[j] = q[1] - q[0]; msk[j] = True
                inj[k + 'c'] = torch.from_numpy(c).cuda(); inj[k + 's'] = torch.from_numpy(z).cuda(); inj[k + 'm'] = torch.from_numpy(msk).cuda()
            with torch.no_grad():
                out = model({'img': torch.stack(imgs).cuda()}, {}, {}, 'test')
            o = {k: v.detach().float().cpu().numpy() for k, v in out.items() if torch.is_tensor(v)}
            h2o = stash['h2o'][:, :2, :]
            A = {'r': [B2I @ np.vstack([h2o[j], [0, 0, 1]]) for j in range(n)], 'l': [B2I @ np.vstack([h2o[n + j], [0, 0, 1]]) for j in range(n)]}
            if fit is None:                              # which MANO reproduces InterWild's own root-relative meshes
                errs = {}
                for flat in (False, True):
                    _, V = fwd(mano(flat), 1, o['rmano_root_pose'], o['rmano_hand_pose'], o['rmano_shape'])
                    d = V - o['rmano_mesh_cam']; errs[flat] = float(np.abs(d - d.mean(1, keepdims=True)).max())
                flat = min(errs, key=errs.get)
                for lf in (False, True):
                    _, V = fwd(mano(flat, lf), 0, o['lmano_root_pose'], o['lmano_hand_pose'], o['lmano_shape'])
                    d = V - o['lmano_mesh_cam']; errs[f'left_fix={lf}'] = float(np.abs(d - d.mean(1, keepdims=True)).max())
                left_fix = errs['left_fix=True'] < errs['left_fix=False']
                assert errs[flat] < 1e-4 and errs[f'left_fix={left_fix}'] < 1e-4, f'MANO mismatch {errs}'
                fit = (flat, left_fix); print('flat_hand_mean', flat, 'left_fix', left_fix, errs, flush=True)
            Vs = {k: fwd(mano(fit[0], fit[1] if s == 0 else False), s, o[f'{k}mano_root_pose'], o[f'{k}mano_hand_pose'], o[f'{k}mano_shape'])[1]
                  for s, k in ((1, 'r'), (0, 'l'))}
            for j, f in enumerate(fs):
                th = (f, 0) in best and (f, 1) in best and get_iou(o['rhand_bbox'][j], o['lhand_bbox'][j], 'xyxy') > 0
                for s, k in ((1, 'r'), (0, 'l')):
                    if (f, s) not in best:
                        continue
                    if k == 'l' and th:
                        P, Ah = o['rroot_cam'][j] + o['rel_trans'][j], A['r'][j]
                    else:
                        P, Ah = o[f'{k}root_cam'][j], A[k][j]
                    sc = np.sqrt(abs(np.linalg.det(Ah[:2, :2])))
                    u, v, _ = Ah @ np.array([cfg.focal[0] * P[0] / P[2] + cfg.princpt[0], cfg.focal[1] * P[1] / P[2] + cfg.princpt[1], 1.0])
                    Z = P[2] * K[0, 0] / (cfg.focal[0] * sc)
                    root = np.array([(u - K[0, 2]) * Z / K[0, 0], (v - K[1, 2]) * Z / K[1, 1], Z])
                    root_sh = (Vs[k][j] - o[f'{k}mano_mesh_cam'][j]).mean(0)      # InterWild's wrist in the tau = 0 mesh
                    hands.append(dict(f=f, side=s, go=aa2R(o[f'{k}mano_root_pose'][j])[0], hp=aa2R(o[f'{k}mano_hand_pose'][j]),
                                      betas=o[f'{k}mano_shape'][j], tau=root - root_sh, score=best[(f, s)][1], native=True))
        write_segments(meth, a.view, m, hands_to_pred(F, hands, fit[0] if fit else False, fit[1] if fit else False))
        print(f"{a.view} {m['name']}: {len(need)} frames, {len(best)} hand boxes, {time.time() - t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
