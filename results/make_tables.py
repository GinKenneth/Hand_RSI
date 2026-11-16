"""Re-score every table of results/README.md from prediction files, with this repository's evaluator.

  python results/make_tables.py --preds PRED_ROOT --views rsi/data/views [--out results/test_scores.json]

PRED_ROOT/<row>/<view>/<segment>.npz, one directory per table row (e.g. 00, F1, N1, ..., N5, C1_1, C2_1) and view
(hot3d_test, arctic_test, hoi4d_test). The ground truth comes from the dataset views (rsi/data/build_views.py). Prints
one 10-column table per view.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rsi'))
import rsi_paths  # noqa: E402,F401  (puts ace_eval on sys.path)
from eval.score import TABLE, per_sequence, pooled  # noqa: E402


def score(job):
    row, view, pred, gt = job
    m = pooled(list(per_sequence(Path(gt), Path(pred)).values()))
    return row, view, {k: float(m[k]) for k in TABLE}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--preds', type=Path, required=True)
    ap.add_argument('--views', type=Path, required=True)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--jobs', type=int, default=8)
    a = ap.parse_args()
    jobs = [(r.name, v.name, str(v), str(a.views / v.name / 'gt')) for r in sorted(a.preds.iterdir()) if r.is_dir()
            for v in sorted(r.iterdir()) if v.is_dir() and (a.views / v.name / 'gt').is_dir()]
    res = {}
    with ProcessPoolExecutor(a.jobs) as ex:
        for row, view, m in ex.map(score, jobs):
            res.setdefault(row, {})[view] = m
    fmt = lambda k, x: f'{x:.3f}' if k in ('FAcc', 'Recall', 'F1', 'CT-p') else f'{x:.2f}'   # noqa: E731
    for view in sorted({v for r in res.values() for v in r}):
        print(f'\n{view}\n| | ' + ' | '.join(TABLE) + ' |\n|---|' + '---|' * len(TABLE))
        for row, r in res.items():
            if view in r:
                print(f'| {row} | ' + ' | '.join(fmt(k, r[view][k]) for k in TABLE) + ' |')
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
