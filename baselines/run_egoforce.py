"""EgoForce (ACE A: released model_weights.pth; the ACE row is marked with a double dagger because the release
post-filters the camera-space translation with its causal Kalman filter, on by default) on our test views, one run per
81-frame segment (as HaWoR / OmniHands) -> preds/egoforce/<view>/<segment>.npz.
Pipeline = demo/inference.py Inference.run_outputs on every frame, as demo/run_app.py's video loop does it:
reset_runtime_state at the segment start, set_camera_model(PinholeCameraModel(GT fx fy, GT cx cy, W, H)), Kalman
frequency = the video's fps, EgoForce's own detectors (RTMDet hand / forearm epoch_460.pth + the YOLO-pose hand
detector with tracking), crops, HALO, limb model, ray-space solver, Kalman filter.
Changes needed to run it here, deployment only:
  - torch_tensorrt is not installed: a stub module satisfies the import and compile_to_tensorrt returns the plain fp32
    model; infer() runs it under fp16 autocast (the released engine is fp16);
  - the detector's torch.compile(mode='max-autotune') is skipped (speed only; the detector stays half precision);
  - mmcv 2.1.0 (the release's pin; CUDA ops built here for this torch) / mmengine 0.10.7 / ultralytics 8.4.23 (the release pin; 8.1.x ignores the detector's 512 px metadata) / shapely / terminaltables / lmdb
    from envs/egoforce_site (build_egoforce_env.sh) and the repo's own thirdparty/mmdetection fork and thirdparty/datapipes, first on sys.path
    for this script only;
  - settings._DATA_DIR points inside the repo, the weights are in weights/EgoForce/_DATA: those config paths re-pointed.
Joints: EgoForce's MANO parameters (captured at its MANO forward kinematics, i.e. after its 6D -> axis-angle step) through ace_eval's MANO forward; the hand is
placed where EgoForce's camera-space mesh is (its vertices minus the tau = 0 mesh: a pure translation, asserted).
Existence: visible_hand. Global orientation: EgoForce's MANO global_orient.
  python run_egoforce.py --view arctic_test [--shard i/n] [--seqs a,b]"""
import argparse
import contextlib
import io
import os
import sys
import time
import types

import cv2
import numpy as np
import torch

from common import THIRD_PARTY, OUT, direct_to_pred, fwd, mano, read_frames, sequences

EF = THIRD_PARTY / 'official/EgoForce'
DATA = THIRD_PARTY / 'weights/EgoForce/_DATA'


def as_R(x):
    x = np.asarray(x, np.float64)
    if x.shape[-2:] == (3, 3):
        return x.reshape(3, 3)
    return cv2.Rodrigues(x.reshape(3))[0]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--view', required=True); ap.add_argument('--shard', default='0/1')
    ap.add_argument('--seqs', default=''); a = ap.parse_args()
    torch.set_num_threads(8)
    trt = types.ModuleType('torch_tensorrt')
    trt.runtime = types.SimpleNamespace(set_multi_device_safe_mode=lambda *x, **k: None, set_cudagraphs_mode=lambda *x, **k: None)
    sys.modules['torch_tensorrt'] = trt
    sys.path.insert(0, os.environ.get('EGOFORCE_SITE', str(OUT / 'envs/egoforce_site')))   # build_egoforce_env.sh; ahead of ViTPose's mmcv 1.x
    sys.path.insert(0, str(EF / 'thirdparty/mmdetection'))  # EgoForce's own mmdet 3.3.0 fork (arm datasets, keypoint RTMDet)
    sys.path.insert(0, str(EF / 'thirdparty/datapipes'))    # the repo's datapipes package (install.sh pip-installs it)
    sys.path[:0] = [str(EF), str(EF / 'demo')]
    from settings import config as cfg
    if hasattr(cfg, 'defrost'):
        cfg.defrost()
    cfg.POSE_3D.CHECKPOINT_PATH = str(DATA / 'model_weights.pth'); cfg.DETECTION.HAND_ARM_PATH = str(DATA / 'epoch_460.pth')
    cfg.DETECTION.HAND_PATH = str(DATA / 'detector.torchscript'); cfg.MANO_PATH = str(DATA / 'mano')
    import inference as EI
    from camera_models import PinholeCameraModel
    EI.compile_to_tensorrt = lambda model, device: model.to(device).eval()
    cap = {}
    inf, fit, nprof = None, [None, None], 0
    od = OUT / 'preds/egoforce' / a.view
    for m in sequences(a.view, a.shard, a.seqs):
        todo = [s for s in m['segments'] if not (od / f"{s['name']}.npz").is_file()]
        if not todo:
            continue
        if inf is None:
            tc = torch.compile
            torch.compile = lambda model, *x, **k: model             # the detector's max-autotune compile, skipped
            try:
                inf = EI.Inference()
            finally:
                torch.compile = tc
            hmod = inf.limb_model.hand_model; fk = hmod.forward_kinematics

            def fk_cap(shape_params, joint_angles, global_xfrom, is_right_hand):   # the MANO inputs (after 6D -> axis-angle)
                cap.update(go=global_xfrom[:, :3].detach().float().cpu().numpy(), hp=joint_angles.detach().float().cpu().numpy(),
                           betas=shape_params.detach().float().cpu().numpy(), ht=is_right_hand.detach().cpu().numpy().reshape(-1).astype(int))
                return fk(shape_params=shape_params, joint_angles=joint_angles, global_xfrom=global_xfrom, is_right_hand=is_right_hand)
            hmod.forward_kinematics = fk_cap
        t0 = time.time(); K = m['Kmat']; dev = []
        frames = read_frames(m['video_path'], max(s['end'] for s in todo))
        H, W = frames[0].shape[:2]
        c = cv2.VideoCapture(m['video_path']); fps = c.get(cv2.CAP_PROP_FPS) or 30.0; c.release()
        for s in todo:
            inf.reset_runtime_state()
            inf.set_camera_model(PinholeCameraModel([K[0, 0], K[1, 1]], [K[0, 2], K[1, 2]], W, H))
            inf.set_kalman_filter_frequency(fps)
            entries = []
            for t, f in enumerate(range(s['start'], s['end'])):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):               # per-frame fps prints
                    outs = inf.run_outputs(cv2.cvtColor(frames[f], cv2.COLOR_BGR2RGB), inf.device)
                if os.environ.get('EF_PROFILE') and nprof < 30:
                    nprof += 1; print(' | '.join(l.strip() for l in buf.getvalue().splitlines() if 'time taken' in l), flush=True)
                vis = np.asarray(outs['visible_hand']).reshape(-1)
                Vc = np.asarray(outs['pred_vertices']).reshape(len(vis), 778, 3)
                assert list(cap['ht']) == [0, 1], cap['ht']             # infer() stacks left then right
                n = len(vis); go = cap['go'].reshape(n, -1); hp = cap['hp'].reshape(n, -1); be = cap['betas'].reshape(n, -1)
                go = go.reshape(n, 3, 3) if go.shape[1] == 9 else go.reshape(n, 3)          # rotation matrices or axis-angle
                hp = hp.reshape(n, 15, 3, 3) if hp.shape[1] == 135 else hp.reshape(n, 45)
                for h in np.flatnonzero(vis):                           # which MANO reproduces EgoForce's own meshes:
                    if fit[0] is None or (h == 0 and fit[1] is None):  # flat_hand_mean on the first hand, the left
                        errs = {}                                        # shapedirs fix on the first left hand
                        for flat in ((False, True) if fit[0] is None else (fit[0],)):
                            for lf in ((False, True) if h == 0 else (False,)):
                                _, V = fwd(mano(flat, lf), int(h), go[h:h + 1], hp[h:h + 1], be[h:h + 1])
                                d = Vc[h] - V[0]; errs[(flat, lf)] = float(np.abs(d - d.mean(0)).max())
                        best = min(errs, key=errs.get)
                        assert errs[best] < 1e-3, f'MANO mismatch {errs}'
                        fit = [best[0], best[1] if h == 0 else fit[1]]
                        print('side', h, 'flat_hand_mean, left_fix', fit, errs, flush=True)
                for h in np.flatnonzero(vis):
                    J, V = fwd(mano(fit[0], bool(fit[1])), int(h), go[h:h + 1], hp[h:h + 1], be[h:h + 1])
                    d = Vc[h] - V[0]; off = d.mean(0); dev.append(float(np.abs(d - off).max()))
                    assert dev[-1] < 1e-2, f'EgoForce camera-space mesh is not a translated MANO mesh ({dev[-1]} m)'
                    entries.append(dict(f=t, side=int(h), J=J[0] + off, V=Vc[h], tau=off, R=as_R(go[h]), score=1.0))
            P = direct_to_pred(s['end'] - s['start'], entries)
            od.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(od / f"{s['name']}.npz", **P)
        q = np.percentile(dev, [50, 99, 100]) if dev else [np.nan] * 3
        print(f"{a.view} {m['name']}: {len(todo)} segments, {time.time() - t0:.0f}s; mesh vs MANO+t residual m: median {q[0]:.1e} p99 {q[1]:.1e} max {q[2]:.1e}", flush=True)


if __name__ == '__main__':
    main()
