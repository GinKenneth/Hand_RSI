"""HOI4D pseudo ground truth for the hands HOI4D does not annotate, fixed before re-scoring.

Why: most recordings annotate one hand only (146 of 166 right only), so a correctly detected unannotated hand counts as
a false positive (00 scored precision ~0.69 vs ACE's ~0.95). ACE says part of its HOI4D GT is per-frame-estimator
pseudo-GT; we use WiLoR (wilor_final.ckpt + its YOLO detector, default confidence 0.3, ace_eval.baselines.run_wilor).
Rule, decided before looking at any re-score and applied once:
  - only hand-frames WITHOUT an official HOI4D annotation are filled; every official annotation stays untouched;
  - a hand-frame is filled when WiLoR detects that side (exists > 0.5) with finite joints; otherwise it stays without GT;
  - every filled hand-frame is flagged in gt[...]['pseudo'] (F, 2); results are also reported on the official-only view.
Writes data/hoi4d_pgt/{gt,segments,videos,meta} (segments / videos / meta hard-linked from data/hoi4d_ace) and
data/hoi4d_pgt/pseudo_stats.json.
  python data/hoi4d_pseudo_gt.py
"""
import json
import os
from pathlib import Path

import numpy as np

S = Path(__file__).resolve().parents[1]
SRC, WIL, OUT = S / 'data/hoi4d_ace', S / 'data/hoi4d_wilor_pred', S / 'data/hoi4d_pgt'
for d in ('gt', 'segments', 'videos', 'meta'):
    (OUT / d).mkdir(parents=True, exist_ok=True)
for d in ('segments', 'videos', 'meta'):
    for f in (SRC / d).iterdir():
        if not (OUT / d / f.name).exists():
            os.link(f, OUT / d / f.name)
stats = dict(segments=0, official=[0, 0], filled=[0, 0], wilor_detected_on_official=[0, 0], no_gt=[0, 0])
for g in sorted((SRC / 'gt').glob('*.npz')):
    gt = dict(np.load(g))
    w = dict(np.load(WIL / g.name))
    valid = gt['valid'].copy()
    det = (w['exists'] > 0.5) & np.isfinite(w['joints_cam']).all((-1, -2)) & np.isfinite(w['transl']).all(-1)
    fill = ~valid & det
    pseudo = np.zeros_like(valid)
    for k in ('joints_cam', 'verts_cam', 'transl', 'rot'):
        gt[k] = gt[k].copy()
        gt[k][fill] = w[k][fill]
    pseudo[fill] = True
    gt['valid'] = valid | fill
    gt['pseudo'] = pseudo
    np.savez_compressed(OUT / 'gt' / g.name, **gt)
    stats['segments'] += 1
    for s in (0, 1):
        stats['official'][s] += int(valid[:, s].sum()); stats['filled'][s] += int(fill[:, s].sum())
        stats['wilor_detected_on_official'][s] += int((valid & det)[:, s].sum())
        stats['no_gt'][s] += int((~valid & ~det)[:, s].sum())
stats['note'] = 'per side [left, right], hand-frames over the 498 segments'
(OUT / 'pseudo_stats.json').write_text(json.dumps(stats, indent=1))
print(json.dumps(stats))
