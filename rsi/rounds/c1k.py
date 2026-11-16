"""C1_k for the teacher-on-C1 Loop B (LOOPB_PARAMS teacher_on_c1 = true). Started by run_round in the
background when Loop A starts (the cards are mostly idle then):
  1. C1_k = θ_{k-1} + GT-only LoRA (P['c1_steps'] steps, the Loop B data lists, no teacher)   -> models/<R>_C1k
  2. its raw outputs on the dev views (to prune the teacher on it) and on every training view incl. the extra no-GT
     views (P['extra_train_views'], e.g. arctic_s03) (to label them)                        -> captures/<R>_C1k/<view>
Loop B (run_round) then prunes H_k on C1_k's dev outputs, labels the training views with C1_k + H_k', and trains the
student from C1_k. Idempotent: a finished step is skipped on a rerun.
  python rounds/c1k.py --round R2 --base-ckpt models/R1_theta --gpus 0,1,2,3,4,5,6,7 --datasets hot3d,arctic
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

S = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(S), str(S / 'rounds'), str(S / 'loopA')]
import run_round as rr                                   # noqa: E402
from loopC.run_loopC import loopb_params                 # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--round', required=True)
    ap.add_argument('--base-ckpt', default='release', help="θ_{k-1}'s model dir, or 'release'")
    ap.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    ap.add_argument('--datasets', default='hot3d,arctic')
    ap.add_argument('--train-split', default='train')
    a = ap.parse_args()
    R, gpus, dsets = a.round, a.gpus.split(','), a.datasets.split(',')
    P = loopb_params()
    log = S / 'rounds' / R / 'c1k.log'; log.parent.mkdir(parents=True, exist_ok=True)
    node = f'{R}_C1k'; c1m, c1run = S / 'models' / node, S / 'loopB/runs' / node
    steps = int(P.get('c1_steps', P['steps']))
    extra = list(P.get('extra_train_views', []))
    if not (c1m / 'MERGED_OK').is_file():
        lists = ','.join([str(S / 'data/splits' / f'{ds}_{a.train_split}.txt') for ds in dsets] +
                         [str(S / 'data/splits' / f'{v}.txt') for v in extra])
        mix = ','.join(f'{d}={x}' for d, x in P['mix'].items() if d in dsets)
        ck = [] if a.base_ckpt == 'release' else ['--base-ckpt', f'{a.base_ckpt}/ace_ego_hand_k.pt', '--base-opt', f'{a.base_ckpt}/ace_ego_hand_k.yml']
        rr.say(f'[{R}] C1_k started (background, during Loop A): θ_(k-1) {a.base_ckpt} + GT-only LoRA, {steps} steps')
        if not (c1run / f'step{steps}.pt').is_file():
            rr.run_torch(['--out', str(c1run), '--steps', str(steps), '--w_teacher', '0', '--w_gt', str(P['w_gt']),
                          '--w_exists', str(P['w_exists']), '--w_rot', str(P['w_rot']), '--lr', str(P['lr']), '--rank',
                          str(P['rank']), '--train-list', lists, '--mix', mix, *ck], gpus, log, 29643)
        rr.merge_retry(c1run / f'step{steps}.pt', c1m, gpus, log)
        (c1m / 'MERGED_OK').touch()
    # raw outputs: capture/run_ace.py is CPU-bound, so the views run concurrently with 10 threads per process
    for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[k] = '10'
    vs = [f'{ds}_dev' for ds in dsets] + rr.views(dsets, a.train_split) + extra
    # 2 views at a time (five at once, next to the merge agent's replays and other users' jobs, ran every card out of
    # memory; finished segments are kept, so retries get there, slowly)
    with ThreadPoolExecutor(min(2, len(vs))) as ex:
        list(ex.map(lambda v: rr.capture(node, str(c1m), v, gpus, log), vs))
    (c1m / 'CAPTURED_OK').touch()
    rr.say(f'[{R}] C1_k ready: merged, raw outputs on {vs}')


if __name__ == '__main__':
    main()
