"""Score baseline predictions with the release evaluator (holes-only rule, 10 columns) -> scores/<method>_<view>.json
  python score.py METHOD VIEW [VIEW ...]"""
import json
import sys
from pathlib import Path

from common import ACE_EVAL, OUT, VIEWS
sys.path.insert(0, ACE_EVAL)
from ace_eval.evaluate import evaluate
from ace_eval.metrics import Config

B, V = OUT, VIEWS
COLS = ['FAcc', 'Recall', 'F1', 'MPJPE-p', 'PA-p', 'MPJPE+OOS', 'EPE2D-p', 'GO-p', 'CT-p', 'Jitter']
meth = sys.argv[1]
for v in sys.argv[2:]:
    r = evaluate(V / v / 'gt', B / 'preds' / meth / v, Config())
    (B / 'scores').mkdir(exist_ok=True)
    (B / 'scores' / f'{meth}_{v}.json').write_text(json.dumps(r, indent=1, default=float))
    print(f'| {meth} {v} | ' + ' | '.join(f"{r[c]:.3f}" if c in ('FAcc', 'Recall', 'F1', 'CT-p') else f"{r[c]:.2f}" for c in COLS) + ' |',
          r.get('counts'), flush=True)
