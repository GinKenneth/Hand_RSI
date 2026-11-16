"""Shared pieces of the baseline runs on the test views (ACE-Ego-Hand Appendix A protocol, reporting only).

Paths (environment variables, defaults in brackets):
  BASELINES_VIEWS  [rsi/data/views]        <view>/{segments/<seq>.json, videos/, gt/} made by rsi/data/build_views.py
  BASELINES_ROOT   [third_party/baselines]  official/<Method>/ (each repository at the commit in baselines/README.md),
                                            weights/<Method>/ and dependencies/ViTPose
  BASELINES_OUT    [baselines_out]          frontend/, preds/<method>/<view>/<segment>.npz, work/
Every method writes one evaluator prediction npz per 81-frame segment.
Camera: every crop-based method gets the clip's ground-truth intrinsics (fx, fy, cx, cy) when its crop camera is
converted to the full image (ACE A gives GT intrinsics to the methods that take camera geometry; we extend the same
favour to all, instead of the image-centre principal point of the methods' demo code)."""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
VIEWS = Path(os.environ.get('BASELINES_VIEWS', REPO / 'rsi' / 'data' / 'views'))
OUT = Path(os.environ.get('BASELINES_OUT', REPO / 'baselines_out'))
THIRD_PARTY = Path(os.environ.get('BASELINES_ROOT', REPO / 'third_party' / 'baselines'))
ACE_EVAL = os.environ.get('ACE_EVAL_ROOT', str(REPO / 'ace_eval'))
if ACE_EVAL not in sys.path:
    sys.path.insert(0, ACE_EVAL)                    # ace_eval.mano (the evaluator's MANO and joint convention)


def sequences(view, shard='0/1', seqs=''):
    js = sorted((VIEWS / view / 'segments').glob('*.json'))
    if seqs:
        keep = set(seqs.split(','))
        js = [p for p in js if p.stem in keep]
    i, n = map(int, shard.split('/'))
    for p in js[i::n]:
        m = json.loads(p.read_text())
        v = VIEWS / view / 'videos' / Path(m['video']).name
        m['video_path'] = str(v if v.exists() else m['video'])
        m['name'] = p.stem
        m['Kmat'] = np.asarray(m['K'], np.float64)
        yield m


def needed_frames(m):
    return sorted({f for s in m['segments'] for f in range(s['start'], s['end'])})


def read_frames(path, last):
    """BGR frames 0 .. last-1, decoded sequentially"""
    cap = cv2.VideoCapture(path)
    out = []
    while len(out) < last:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    if len(out) < last:
        raise RuntimeError(f'{path}: {len(out)} frames decoded, {last} needed')
    return out


def crop_cam_to_K(pred_cam, box_center, box_size, K):
    """weak-perspective crop camera (s, tx, ty) -> camera-frame translation under intrinsics K
    (HaMeR / WiLoR cam_crop_to_full with the true focal length and principal point)"""
    pred_cam = np.asarray(pred_cam, np.float64); c = np.asarray(box_center, np.float64)
    b = np.asarray(box_size, np.float64).reshape(-1)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    bs = b * pred_cam[:, 0] + 1e-9
    tz = 2 * fx / bs
    tx = pred_cam[:, 1] + 2 * (c[:, 0] - cx) / bs
    ty = pred_cam[:, 2] + 2 * (c[:, 1] - cy) / bs * fx / fy
    return np.stack([tx, ty, tz], -1)


def workdir(name, links):
    """OUT/work/<name> with the given symlinks {relative path: target}; for official code that resolves its data
    relative to the working directory (HaMeR / Hamba './_DATA/...', OmniHands './_DATA/data/...')"""
    wd = OUT / 'work' / name
    for rel, target in links.items():
        p = wd / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.is_symlink():
            p.symlink_to(Path(target).resolve())
    return wd


_MODELS = {}


def mano(flat, left_fix=False):
    """ace_eval's MANO pair {0: left, 1: right}. left_fix: the left model with smplx issue 48's shapedirs fix
    (shapedirs[:, 0, :] *= -1), as HaWoR / ARCTIC-derived code applies it"""
    key = (flat, left_fix)
    if key not in _MODELS:
        if ACE_EVAL not in sys.path:
            sys.path.insert(0, ACE_EVAL)
        from ace_eval.mano import load_models
        m = load_models(flat_hand_mean=flat)
        if left_fix:
            import torch
            with torch.no_grad():
                m[0].shapedirs[:, 0, :] *= -1
        _MODELS[key] = m
    return _MODELS[key]


def rotvec(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(np.asarray(R, np.float64).reshape(-1, 3, 3)).as_rotvec().reshape(R.shape[:-2] + (3,))


M = np.diag([-1.0, 1.0, 1.0])


def fwd(models, side, go, hp, betas):
    """ace_eval.mano.forward with rotation matrices: its shape test reads an axis-angle batch of exactly 3 hands, (3, 3),
    as one rotation matrix. go (n,3)|(n,3,3), hp (n,45)|(n,15,3)|(n,15,3,3)"""
    from ace_eval.mano import forward
    go = np.asarray(go, np.float64); hp = np.asarray(hp, np.float64); n = len(go)
    if go.shape[1:] != (3, 3):
        go = rotmat(go.reshape(n, 3))
    if hp.shape[1:] != (15, 3, 3):
        hp = rotmat(hp.reshape(n * 15, 3)).reshape(n, 15, 3, 3)
    return forward(models, side, go, hp, betas)


def rotmat(aa):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(np.asarray(aa, np.float64).reshape(-1, 3)).as_matrix()


def hands_to_pred(F, hands, flat=True, left_fix=False):
    """hands: list of dicts (frame f, side 0 left / 1 right, go (3,3), hp (15,3,3), betas (10,), tau (3,), score[, native]).
    Default (HaMeR / WiLoR / Hamba): MANO parameters in the RIGHT-hand frame of the (for a left hand: mirrored) crop; the
    left hand is that right-hand mesh mirrored in x. native=True (WildHands): a left hand comes with LEFT-hand MANO
    parameters. flat: flat_hand_mean of the method's MANO; left_fix: its left MANO has the shapedirs fix. Per frame and side the highest score is kept (the evaluator
    has one slot per side). -> evaluator npz dict"""
    best = {}
    for h in hands:
        k = (int(h['f']), int(h['side']))
        if k not in best or h['score'] > best[k]['score']:
            best[k] = h
    P = dict(exists=np.zeros((F, 2), np.float32), joints_cam=np.full((F, 2, 21, 3), np.nan, np.float32),
             verts_cam=np.full((F, 2, 778, 3), np.nan, np.float32), transl=np.full((F, 2, 3), np.nan, np.float32),
             rot=np.full((F, 2, 3, 3), np.nan, np.float32), score=np.zeros((F, 2), np.float32))
    groups = {}
    for k, h in best.items():
        groups.setdefault((bool(h.get('native')) and k[1] == 0), []).append(k)
    for native_left, ks in groups.items():
        go = np.stack([best[k]['go'] for k in ks])
        hp = np.stack([best[k]['hp'] for k in ks])
        be = np.stack([best[k]['betas'] for k in ks])
        J, V = fwd(mano(flat, left_fix), 0 if native_left else 1, go, hp, be)
        for n, (f, s) in enumerate(ks):
            j, v, R = J[n].copy(), V[n].copy(), np.asarray(best[(f, s)]['go'], np.float64)
            if s == 0 and not native_left:
                j[:, 0] *= -1; v[:, 0] *= -1; R = M @ R @ M
            t = np.asarray(best[(f, s)]['tau'], np.float64)
            P['exists'][f, s] = 1; P['joints_cam'][f, s] = j + t; P['verts_cam'][f, s] = v + t; P['transl'][f, s] = t
            P['rot'][f, s] = R; P['score'][f, s] = best[(f, s)]['score']
    return P


def write_segments(method, view, m, P):
    """P: full-sequence prediction arrays indexed by video frame -> one npz per segment"""
    od = OUT / 'preds' / method / view
    od.mkdir(parents=True, exist_ok=True)
    for s in m['segments']:
        np.savez_compressed(od / f"{s['name']}.npz", **{k: v[s['start']:s['end']] for k, v in P.items()})


def done(method, view, m):
    od = OUT / 'preds' / method / view
    return all((od / f"{s['name']}.npz").is_file() for s in m['segments'])


def direct_to_pred(F, entries):
    """entries: dicts (f, side, J (21,3) camera frame in ACE's joint order, V (778,3), tau (3,), R (3,3), score); the
    highest score per frame and side is kept -> evaluator npz dict"""
    best = {}
    for e in entries:
        k = (int(e['f']), int(e['side']))
        if k not in best or e['score'] > best[k]['score']:
            best[k] = e
    P = dict(exists=np.zeros((F, 2), np.float32), joints_cam=np.full((F, 2, 21, 3), np.nan, np.float32),
             verts_cam=np.full((F, 2, 778, 3), np.nan, np.float32), transl=np.full((F, 2, 3), np.nan, np.float32),
             rot=np.full((F, 2, 3, 3), np.nan, np.float32), score=np.zeros((F, 2), np.float32))
    for (f, s), e in best.items():
        P['exists'][f, s] = 1; P['joints_cam'][f, s] = e['J']; P['verts_cam'][f, s] = e['V']; P['transl'][f, s] = e['tau']
        P['rot'][f, s] = e['R']; P['score'][f, s] = e['score']
    return P
