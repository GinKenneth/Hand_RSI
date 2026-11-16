"""Run rounds back to back on this host until a stop rule fires (evaluated after a possible
Loop B redo of a rejected round, which its Loop C may ask for once):
  - two consecutive rounds fail gate GB, or
  - --max-rounds rounds were run, or the file rounds/STOP exists, or --until has passed (no new round starts then).
Round k = rounds/run_round.py --round R<k> (every stage but loopC). Loop C of R<k> then runs in the background while
R<k+1>'s capture runs (the two are independent: the capture only needs GB's verdict); R<k+1> continues once Loop C has
written its plan (run_round applies the plan when its base / Loop A stages start). Control C1 runs with the rounds in
--c1-rounds (default: R1 only; the full C1 at the end is started by hand).
  python rounds/rsi_loop.py --gpus 0,1,2,3,4,5,6,7 [--start 1] [--max-rounds 20] [--until 'YYYY-MM-DD HH:MM'] \
      [-- extra run_round args, e.g. --loopa-hours 4]
Run as root (sandbox), with RSI_KEYS_FILE and TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD set.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

S = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(S), str(S / 'rounds')]
from run_round import PY, STATE, say  # noqa: E402

STAGES = 'capture,base,loopA,teacher,integrate,loopB,revalidate,gate,sealed,controls'
RETRY_STAGES = 'loopB,revalidate,gate,sealed,controls'      # a Loop B redo keeps the round's Loop A and teacher


def loopc_action(R: str) -> tuple[str, str]:
    """Loop C's validated decision for round R (loopC/run_loopC.py check_action): ('next_round' | 'retry_loopB', why)"""
    p = S / 'rounds' / R / 'loopC' / 'plan_validated' / 'action.json'
    try:
        d = json.loads(p.read_text())
        return str(d.get('action', 'next_round')), str(d.get('why', ''))
    except (OSError, ValueError):
        return 'next_round', ''


def verdict(R: str):
    p = S / 'rounds' / R / 'round.json'
    v = json.loads(p.read_text()).get('verdict') if p.is_file() else None
    return None if v is None else bool(v.get('accepted'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gpus', required=True)
    ap.add_argument('--start', type=int, default=1)
    ap.add_argument('--max-rounds', type=int, default=30)
    ap.add_argument('--until', default='', help="'YYYY-MM-DD HH:MM': no round starts after this time")
    ap.add_argument('--c1-rounds', default='1')
    ap.add_argument('--prefix', default='R', help='round names <prefix><k> (smoke tests: SMOKE)')
    ap.add_argument('rest', nargs=argparse.REMAINDER, help='extra run_round arguments after --')
    a = ap.parse_args()
    extra = [x for x in a.rest if x != '--']
    until = time.mktime(time.strptime(a.until, '%Y-%m-%d %H:%M')) if a.until else None
    c1 = {int(x) for x in a.c1_rounds.split(',') if x}
    rr = [PY, str(S / 'rounds/run_round.py'), '--gpus', a.gpus, *extra]
    lc, prev_lc_round = None, None
    for k in range(a.start, a.start + a.max_rounds):
        R = f'{a.prefix}{k}'
        if (STATE / 'STOP').exists() or (S / 'rounds' / 'STOP').exists():
            say(f'[loop] STOP file found; not starting {R}'); break
        if until and time.time() > until:
            say(f'[loop] past --until {a.until}; not starting {R}'); break
        if lc is not None:                          # R<k> capture overlaps Loop C of R<k-1>
            subprocess.run(rr + ['--round', R, '--stages', 'capture'], cwd=str(S))
            if lc.wait():
                say(f'[loop] Loop C of {prev_lc_round} failed (rc {lc.returncode}); {R} continues without a new plan')
        say(f'[loop] {R} starts')
        r = subprocess.run(rr + ['--round', R, '--stages', STAGES] + (['--c1'] if k in c1 else []), cwd=str(S))
        if r.returncode:
            say(f'[loop] {R} FAILED (rc {r.returncode}); the loop stops for a human look'); break
        lc = subprocess.Popen(rr + ['--round', R, '--stages', 'loopC'], cwd=str(S)); prev_lc_round = R
        if verdict(R) is False:
            # GB rejected R: wait for its Loop C, which may ask (once, check_action) to redo R's Loop B with changed
            # Loop B parameters, keeping R's Loop A; the next round's capture waits for that decision
            if lc.wait():
                say(f'[loop] Loop C of {R} failed (rc {lc.returncode}); no Loop B redo')
            else:
                act, why = loopc_action(R)
                if act == 'retry_loopB':
                    say(f'[loop] Loop C of {R} asks to redo its Loop B (Loop A kept): {why[:300]}')
                    r = subprocess.run(rr + ['--round', R, '--stages', RETRY_STAGES, '--retry-loopB'] + (['--c1'] if k in c1 else []),
                                       cwd=str(S))
                    if r.returncode:
                        say(f'[loop] {R} Loop B redo FAILED (rc {r.returncode}); the loop stops for a human look'); lc = None; break
                    lc = subprocess.Popen(rr + ['--round', R, '--stages', 'loopC'], cwd=str(S))   # Loop C again: may only go on
        v_now, v_prev = verdict(R), verdict(f'{a.prefix}{k - 1}') if k > 1 else None
        if v_now is False and v_prev is False:
            say(f'[loop] stop rule: {R} and {a.prefix}{k - 1} both failed GB'); break
    if lc is not None:
        lc.wait()
    say('[loop] finished')


if __name__ == '__main__':
    main()
