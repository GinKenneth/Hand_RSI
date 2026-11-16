"""ARCTIC splits (rule fixed before any training or scoring):
  test = s05 (the ACE protocol's test subject, sealed); of the other subjects, one is dev and one is val, drawn with
  numpy default_rng(20261028) from the sorted list; the rest are train.
Writes data/splits/arctic_{train,dev,val,test}.txt (sequence ids ar__<sub>_<seq>) from the converted sequences.
  python data/make_arctic_splits.py [--src data/arctic_ace]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

S = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument('--src', type=Path, default=S / 'data' / 'arctic_ace')
a = ap.parse_args()
by_sub = defaultdict(list)
for f in sorted((a.src / 'segments').glob('ar__*.json')):
    by_sub[json.loads((a.src / 'meta' / f.name).read_text())['subject']].append(f.stem)
others = sorted(s for s in by_sub if s != 's05')
perm = list(np.random.default_rng(20261028).permutation(others))
split = dict(test=['s05'], dev=[perm[0]], val=[perm[1]], train=sorted(perm[2:]))
for name, subs in split.items():
    ids = sorted(sid for s in subs for sid in by_sub[s])
    (S / 'data' / 'splits' / f'arctic_{name}.txt').write_text('\n'.join(ids) + '\n')
    segs = sum(len(json.loads((a.src / 'segments' / f'{i}.json').read_text())['segments']) for i in ids)
    print(f'arctic_{name}: subjects {subs}, {len(ids)} sequences, {segs} segments')
