"""Per-split dataset views in the ACE-protocol layout (segments/ gt/ videos/ meta/ as HARD links: a sandbox may bind one
view without seeing any other split), one per dataset/split:
  data/views/<dataset>_<split>/   e.g. hot3d_val, hot3d_dev, arctic_train, hoi4d_test
Every tool of the pipeline (capture, harness.run, eval.score, loopA, loopB) reads a view, so a split is defined once,
here, from data/splits/<dataset>_<split>.txt.
  python data/build_views.py [--only arctic]
Sources are searched in order, then the view itself and the other views of the dataset (a host that received only
the views builds new views from them).
"""
import argparse
import json
import os
from pathlib import Path

S = Path(__file__).resolve().parents[1]; V = S / 'data' / 'views'
SOURCES = {'hot3d': [S / 'data' / 'hot3d_ace',                    # data/convert_hot3d.py: every sequence with public GT
                     S / 'data' / 'hot3d_extra_ace', S / 'data' / 'hot3d_nogt_ace'],   # nogt: 62 recordings, teacher only
           'arctic': [S / 'data' / 'arctic_ace'],
           'hoi4d': [S / 'data' / 'hoi4d_ace']}                    # hoi4d_test = official annotations only (main)
SPLIT_SOURCES = {'hoi4d_testpseudo': [S / 'data' / 'hoi4d_pgt']}   # + WiLoR pseudo-GT for unannotated hands (appendix)
PREFIX = {'hot3d': 'h3__', 'arctic': 'ar__', 'hoi4d': 'hd__'}

ap = argparse.ArgumentParser()
ap.add_argument('--only', default='')
a = ap.parse_args()
for lst in sorted((S / 'data' / 'splits').glob('*.txt')):
    ds, split = lst.stem.split('_', 1)
    if a.only and ds != a.only:
        continue
    view = V / f'{ds}_{split}'
    for d in ('segments', 'gt', 'videos', 'meta'):
        (view / d).mkdir(parents=True, exist_ok=True)
    nseq = nseg = 0
    for s in lst.read_text().split():
        sid = s if s.startswith(PREFIX[ds]) else PREFIX[ds] + s
        others = sorted(x for x in V.glob(f'{ds}_*') if x != view)              # e.g. a host that only has the views
        srcs = SPLIT_SOURCES.get(f'{ds}_{split}', SOURCES[ds] + [view] + others)
        src = next(r for r in srcs if (r / 'segments' / f'{sid}.json').is_file())
        for d, name in (('segments', f'{sid}.json'), ('videos', f'{sid}.mp4'), ('meta', f'{sid}.json')):
            t = view / d / name
            if not t.exists() and (src / d / name).exists():
                os.link(src / d / name, t)
        for sg in json.loads((src / 'segments' / f'{sid}.json').read_text())['segments']:
            t = view / 'gt' / f"{sg['name']}.npz"
            if not t.exists():
                os.link(src / 'gt' / f"{sg['name']}.npz", t)
            nseg += 1
        nseq += 1
    print(f'{view.name}: {nseq} sequences, {nseg} segments')
