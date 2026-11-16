"""ACE-Ego-Hand K-free (released ace_ego_hand_kfree.pt + options/ace_ego_hand_kfree.yml: self_ray_decode, the model never
reads K) on the test views, through the same official-inference capture as the theta0 row (rsi/capture/run_ace.py,
native resolution, per-segment context) and the empty-harness conversion. As ACE A.1 says for K-free, the GT intrinsics
enter only the scoring: the direct 2D head is dropped, so EPE2D re-projects the 3D joints under the GT K.
Existence: --exists 2d (default, the in-frame visibility head, infer_video's deployment rule) -> preds/ace_kfree_e2;
--exists 3d (the rule of the theta0 row) -> preds/ace_kfree.
  ACE_KFREE_CKPT=<dir with ace_ego_hand_k.pt + ace_ego_hand_k.yml> python run_kfree.py [--exists 2d|3d] [views...]"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from common import OUT, REPO, VIEWS

sys.path[:0] = [str(REPO / 'rsi'), str(REPO / 'rsi/rounds'), str(REPO / 'rsi/loopA')]
import run_round as rr

ap = argparse.ArgumentParser(); ap.add_argument('views', nargs='*', default=['arctic_test', 'hoi4d_test', 'hot3d_test'])
ap.add_argument('--exists', choices=('2d', '3d'), default='2d'); ap.add_argument('--gpus', default='0,1,2,3'); a = ap.parse_args()
CK = Path(os.environ['ACE_KFREE_CKPT']); log = OUT / 'logs/kfree.log'; log.parent.mkdir(parents=True, exist_ok=True)
for v in a.views:
    vd = VIEWS / v; cap = OUT / 'captures/ace_kfree' / v; cap.mkdir(parents=True, exist_ok=True)
    n_seg = len(list((vd / 'gt').glob('*.npz')))
    use = a.gpus.split(',')
    if len(list(cap.glob('*.npz'))) < n_seg:
        ps = [subprocess.Popen(['env', f'CUDA_VISIBLE_DEVICES={g}', rr.PY, str(REPO / 'rsi/capture/run_ace.py'), '--data', str(vd), '--video-dir',
                                str(vd / 'videos'), '--out', str(cap), '--ckpt', str(CK / 'ace_ego_hand_k.pt'), '--opt', str(CK / 'ace_ego_hand_k.yml'),
                                '--gpu', '0', '--shard', f'{i}/{len(use)}'], stdout=open(log, 'a'), stderr=subprocess.STDOUT)
              for i, g in enumerate(use)]
        print(time.strftime('%H:%M'), v, 'capture rc', [p.wait() for p in ps], flush=True)
    assert len(list(cap.glob('*.npz'))) >= n_seg, f'{v}: capture incomplete'
    tmp = rr.harness_preds(rr.EMPTY, cap, v, OUT / 'preds/_ace_kfree_raw' / v, log)
    od = OUT / 'preds' / ('ace_kfree_e2' if a.exists == '2d' else 'ace_kfree') / v; od.mkdir(parents=True, exist_ok=True)
    for p in sorted(tmp.glob('*.npz')):
        d = dict(np.load(p)); d.pop('joints_2d', None)
        if a.exists == '2d':
            d['exists'] = np.asarray(np.load(cap / p.name)['exists_2d'], np.float32)[:len(d['exists'])].reshape(-1, 2)
        np.savez_compressed(od / p.name, **d)
    print(time.strftime('%H:%M'), v, 'K-free predictions', len(list(od.glob('*.npz'))), flush=True)
