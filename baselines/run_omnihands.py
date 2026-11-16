"""OmniHands (ACE A: nine-frame video model Demo_Video.pth with config_video.yaml behind the ViTDet-H + ViTPose+-Huge
front end) on our test views, one run per 81-frame segment (as HaWoR) -> preds/omnihands/<view>/<segment>.npz.
Steps as run_demo.py --mode video (detect_video + run_model_video(smooth=True)), on the segment's frames:
  - boxes from the FIRST person's hand keypoints only (vitposes_out[0:1]), conf > 0.5 on more than 3; a frame without
    a box takes the previous frame's, then the demo's forward / backward fill (its backward loop stops before frame 0);
  - the demo's 30-frame box smoothing (it averages box[2:], i.e. x2 y2), ViTDetInterDataset_Batch / _Sequence with
    rescale_factor 2.0, sequences gen_sequences(F, SEQ_LEN 9, gap 10), token pass in batches of 32, temporal pass in 8.
    The time head reads only the centre frame's bbox_right / bbox_left / bbox_full from the sequence batch, so those are
    taken from the single-frame dataset (same formula) instead of re-cropping all nine frames per window as the demo
    does (~9x the Gaussian-blurred crops, unused);
  - seq_smooth on the root-relative ("world") vertices, applied here to vertices and joints together (it is a linear
    per-coordinate recursion, so the joints follow the vertices exactly).
Camera: the time head places each hand by cv2.solvePnPRansac of its 3D joints on its full-image 2D joints under
cam_k = (5000/256*max(W,H), image centre); we redo that PnP with the clip's GT intrinsics (common: every method gets GT
K where its camera meets the full image), same RANSAC call and the same align rule (one joint solve unless the boxes'
IoU < 0.01 or the roots are > 0.2 m apart). The replication is checked against the model's own cam_aligned under cam_k.
Existence: the demo draws both hands on every frame once a side has a box anywhere in the clip, so a side is predicted
on every frame of a segment where it is detected at least once (filled frames included). A side never detected in the
segment gets the other side's box as a stand-in input (the model always takes two crops), its output is discarded and
the detected side is solved alone (no joint align).
Joints: OmniHands' right MANO (smplx, both sides through layer['right'], left crops flipped) re-posed by ace_eval's MANO
forward (flat_hand_mean picked by matching OmniHands' own vertices, asserted) -> ACE's 21 joints; left hands mirrored
as OmniHands does. Global orientation for GO-p: the MANO global_orient (mirrored for the left), before seq_smooth.
  python run_omnihands.py --view arctic_test [--shard i/n] [--seqs a,b]"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

from common import THIRD_PARTY, OUT, direct_to_pred, fwd, mano, read_frames, sequences, workdir
from frontend import boxes_from  # noqa: F401  (same front-end files; OmniHands applies its own first-person rule)

OH = THIRD_PARTY / 'official/OmniHands'
M = np.diag([-1.0, 1.0, 1.0])


def first_person_boxes(fe, frames, thr=0.5):
    """detect_video on the segment's frames: {'right': {i: box}, 'left': {i: box}} (i = index within the segment)"""
    first = {}
    for n, f in enumerate(fe['person_frame']):
        first.setdefault(int(f), n)                     # persons are stored in detector score order: first = vitposes_out[0]
    b = {'right': {}, 'left': {}}
    for i, f in enumerate(frames):
        for side, key in ((0, 'left'), (1, 'right')):
            ok = False
            if f in first:
                kp = fe['person_hands'][first[f]][side]; v = kp[:, 2] > thr
                if v.sum() > 3:
                    b[key][i] = [float(kp[v, 0].min()), float(kp[v, 1].min()), float(kp[v, 0].max()), float(kp[v, 1].max())]
                    ok = True
            if not ok and i > 0 and (i - 1) in b[key]:
                b[key][i] = b[key][i - 1]
    F = len(frames)
    for i in range(F):                                  # the demo's fill, verbatim (backward loop excludes index 0)
        for key in ('right', 'left'):
            if i not in b[key]:
                for j in range(i + 1, F):
                    if j in b[key]:
                        b[key][i] = b[key][j]; break
            if i not in b[key]:
                for j in range(i - 1, 0, -1):
                    if j in b[key]:
                        b[key][i] = b[key][j]; break
    return b


def gen_sequences(f_num, seq_len, gap):
    out = []
    for fid in range(f_num):
        start = fid - (seq_len // 2) * gap
        out.append([min(max(start + k * gap, 0), f_num - 1) for k in range(seq_len)])
    return np.array(out)


def smooth_boxes(b, F):
    seqs = gen_sequences(F, 30, 1); out = {'right': {}, 'left': {}}
    for i in range(F):
        for key in ('right', 'left'):
            sc = np.zeros(2)
            for s in seqs[i]:
                sc = sc + np.array(b[key][s], np.float64)[2:]
            out[key][f'{i}'] = np.concatenate([np.array(b[key][i], np.float64)[:2], sc / 30])
    return out


def seq_smooth(x):
    """run_demo.seq_smooth for one side, x (T, N, 3), in place"""
    last = v = a = last_v = None
    for i in range(len(x)):
        if last is not None and v is not None and a is not None:
            x[i] = x[i] * 0.7 + (last + v + 0.5 * a) * 0.3
        if last is not None:
            v = x[i] - last
        if last_v is not None and v is not None:
            a = v - last_v
        last = x[i]; last_v = v
    return x


def pnp(j3d, j2d, K):
    ok, r, t, _ = cv2.solvePnPRansac(j3d, j2d, K, np.zeros(4))
    assert ok
    return t.squeeze(1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--view', required=True); ap.add_argument('--shard', default='0/1')
    ap.add_argument('--seqs', default='')
    ap.add_argument('--tag', default='omnihands', help='output method name (diagnostics write elsewhere)')
    ap.add_argument('--both', action='store_true', help='also write the official cam_k placement to <tag>_demoK (diagnostic)')
    a = ap.parse_args()
    torch.set_num_threads(8)
    os.chdir(workdir('omnihands', {'_DATA': OH / '_DATA'})); sys.path[:0] = [str(OH)]   # OmniHands' _DATA (its README)
    import einops
    from hands_4d.configs import get_config
    from hands_4d.models.Hands4D import Hands4D
    from hands_4d.utils import recursive_to
    from hands_4d.datasets.vitdet_dataset import ViTDetInterDataset_Batch, HandToken_Sequence
    from hands_4d.models.heads.time_head import calculate_iou
    model, flat, checked = None, None, False      # checked: PnP replication verified on a segment with both sides
    od = OUT / 'preds' / a.tag / a.view; od2 = OUT / 'preds' / f'{a.tag}_demoK' / a.view
    for m in sequences(a.view, a.shard, a.seqs):
        todo = [s for s in m['segments'] if not (od / f"{s['name']}.npz").is_file()]
        fe_p = OUT / 'frontend' / a.view / f"{m['name']}.npz"
        if not todo or not fe_p.is_file():
            continue
        if model is None:
            cfg = get_config(str(OH / 'checkpoints/config_video.yaml'), update_cachedir=True)   # = load_from_ckpt
            model = Hands4D(cfg)
            r = model.load_state_dict(torch.load(str(THIRD_PARTY / 'weights/OmniHands/Demo_Video.pth'), map_location='cpu'), strict=False)
            print('state dict: missing', len(r.missing_keys), r.missing_keys[:8], 'unexpected', len(r.unexpected_keys), r.unexpected_keys[:8], flush=True)
            model = model.cuda().eval()
        t0 = time.time(); fe = np.load(fe_p); K = m['Kmat']
        frames = read_frames(m['video_path'], max(s['end'] for s in todo))
        H, W = frames[0].shape[:2]
        cam_k = np.array([[cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * max(W, H), 0, W / 2],
                          [0, cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * max(W, H), H / 2], [0, 0, 1]], np.float32)
        for s in todo:
            fids = list(range(s['start'], s['end'])); T = len(fids)
            b = first_person_boxes(fe, fids)
            have = {k: len(b[k]) > 0 for k in ('right', 'left')}
            entries = []
            if any(have.values()):
                for k, o in (('right', 'left'), ('left', 'right')):
                    if not have[k]:
                        b[k] = dict(b[o])                     # stand-in crop for the undetected side
                bx = smooth_boxes(b, T)
                imgs = [frames[f] for f in fids]
                seqs = gen_sequences(T, cfg.MODEL.SEQ_LEN, 10)
                ds1 = ViTDetInterDataset_Batch(cfg, imgs, bx, rescale_factor=2.0)
                items = [ds1[j] for j in range(T)]
                BK = ('bbox_right', 'bbox_left', 'bbox_full')
                toks = []
                with torch.no_grad():
                    for i in range(0, T, 32):
                        batch = recursive_to(torch.utils.data.default_collate(items[i:i + 32]), 'cuda')
                        toks.append(einops.rearrange(model.inference_token_forward(batch), '(b s) c -> b s c', s=2).cpu())
                    toks = torch.cat(toks); tds = HandToken_Sequence(toks, seqs)
                    outs = []
                    for i in range(0, T, 8):
                        idx = range(i, min(i + 8, T))
                        tk = einops.rearrange(torch.stack([tds[j] for j in idx]), 'b t s c -> (b t s) c').cuda()
                        batch = {k: torch.stack([torch.stack([items[q][k] for q in seqs[j]]) for j in idx]).cuda() for k in BK}
                        assert all(seqs[j][cfg.MODEL.SEQ_LEN // 2] == j for j in idx)
                        o = model.inference_temp_forward(tk, batch)
                        bb = {k: batch[k][:, cfg.MODEL.SEQ_LEN // 2] for k in ('bbox_right', 'bbox_left')}
                        o['iou'] = calculate_iou(bb['bbox_right'], bb['bbox_left'])
                        outs.append({k: v.detach().float().cpu().numpy() for k, v in o.items() if torch.is_tensor(v)})
                o = {k: np.concatenate([x[k] for x in outs]) for k in outs[0]}
                # camera: the time head's PnP with GT K (checked against its own result under cam_k on the first segment)
                both = have['right'] and have['left']
                align = both & (o['iou'] >= 0.01) & (np.linalg.norm(o['joints3d_world_left'][:, 9] - o['joints3d_world_right'][:, 9], axis=-1) <= 0.2)
                cams = {}
                for KK, tag in ((cam_k, 'demo'), (K.astype(np.float32), 'gt')):
                    if tag == 'demo' and (checked or not both) and not a.both:
                        continue
                    c = {k: np.zeros((T, 3)) for k in ('right', 'left')}
                    for t in range(T):
                        j3 = {k: o[f'joints3d_world_{k}'][t].astype(np.float32) for k in ('right', 'left')}
                        j2 = {k: o[f'joints2d_world_{k}'][t].astype(np.float32) for k in ('right', 'left')}
                        if align[t]:
                            c['right'][t] = c['left'][t] = pnp(np.concatenate([j3['right'], j3['left']]), np.concatenate([j2['right'], j2['left']]), KK)
                        else:
                            for k in ('right', 'left'):
                                c[k][t] = pnp(j3[k], j2[k], KK)
                    cams[tag] = c
                if not checked and both:      # our PnP and align rule == the model's own, under its cam_k
                    err = max(float(np.abs(cams['demo'][k] - o[f'cam_aligned_{k}']).max()) for k in ('right', 'left'))
                    assert err < 1e-3, f'PnP replication mismatch {err}'
                    print('pnp check', err, flush=True); checked = True
                if flat is None:
                    errs = {}
                    for fl in (False, True):
                        _, V = fwd(mano(fl), 1, o['mano_pose_right'][:, :3], o['mano_pose_right'][:, 3:], o['mano_shape_right'])
                        errs[fl] = float(np.abs(V - o['verts3d_right']).max())
                    flat = min(errs, key=errs.get)
                    assert errs[flat] < 1e-4, f'MANO mismatch {errs}'
                    print('flat_hand_mean', flat, errs, flush=True)
                ent2 = []
                for k, side in (('right', 1), ('left', 0)):
                    if not have[k]:
                        continue
                    J, V = fwd(mano(flat), 1, o[f'mano_pose_{k}'][:, :3], o[f'mano_pose_{k}'][:, 3:], o[f'mano_shape_{k}'])
                    r9 = o[f'joints3d_{k}'][:, 9:10]
                    Jw, Vw = J - r9, V - r9
                    R = np.stack([cv2.Rodrigues(np.asarray(g, np.float64))[0] for g in o[f'mano_pose_{k}'][:, :3]])
                    if side == 0:                             # OmniHands' left: mirrored right-hand mesh + root offset
                        Jw[..., 0] *= -1; Vw[..., 0] *= -1; rel = o['root_rel'][:, None]
                        Jw, Vw = Jw + rel, Vw + rel; R = M @ R @ M
                    assert np.abs(Vw - o[f'verts3d_world_{k}']).max() < 1e-4
                    X = seq_smooth(np.concatenate([Vw, Jw], 1).astype(np.float64))
                    Vw, Jw = X[:, :778], X[:, 778:]
                    c = cams['gt'][k]
                    tau = c - (r9[:, 0] if side == 1 else (r9[:, 0] @ M) - o['root_rel'])
                    for t in range(T):
                        entries.append(dict(f=t, side=side, J=Jw[t] + c[t], V=Vw[t] + c[t], tau=tau[t], R=R[t], score=1.0))
                    if a.both and 'demo' in cams:
                        c2 = cams['demo'][k]; tau2 = c2 - (r9[:, 0] if side == 1 else (r9[:, 0] @ M) - o['root_rel'])
                        for t in range(T):
                            ent2.append(dict(f=t, side=side, J=Jw[t] + c2[t], V=Vw[t] + c2[t], tau=tau2[t], R=R[t], score=1.0))
            P = direct_to_pred(T, entries)
            od.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(od / f"{s['name']}.npz", **P)
            if a.both:
                od2.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(od2 / f"{s['name']}.npz", **direct_to_pred(T, ent2 if any(have.values()) else []))
        print(f"{a.view} {m['name']}: {len(todo)} segments, {time.time() - t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
