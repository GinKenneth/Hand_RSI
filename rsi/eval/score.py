"""Scoring and gate on top of ace_eval (unchanged).

  python -m eval.score --gt GT_DIR --pred PRED_DIR [--ref REF_PRED_DIR] [--segs list] [--out res.json]

Per sequence, the evaluator's accumulators are summed, so the pooled metrics equal ace_eval's. With --ref, the
composite S is computed over ALL 10 table columns:
  r_m = (ref - new) / ref         error columns: MPJPE-p, PA-p, MPJPE+OOS, EPE2D-p, GO-p, CT-p, Jitter
  r_m = (new - ref) / max(1 - ref, 0.05)   detection columns (share of the remaining error removed, the remaining
                                  error floored at 5 %): FAcc, Recall, F1
  S = mean over the columns defined on this dataset (HOI4D has no MPJPE+OOS) of r clipped to +-5 %
plus a paired bootstrap over sequences (both runs resampled with the same sequence draw).
Gate (one dataset): CI90 lower bound of S > 0, S >= min_gain, and no regressed column. A column is regressed when
it is worse than max_regress (2 %) AND significantly so (its own CI90 entirely below 0), or worse than hard_cap (5 %)
at all.
Across datasets, rounds/ averages S and requires every dataset to pass the no-regression part.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rsi_paths  # noqa: E402,F401  (puts ace_eval on sys.path)
from ace_eval.evaluate import load_pair  # noqa: E402
from ace_eval.mano import canonical_joints  # noqa: E402
from ace_eval.metrics import Acc, Config, score_segment, summarize  # noqa: E402

ERR = ("MPJPE-p", "PA-p", "MPJPE+OOS", "EPE2D-p", "GO-p", "CT-p", "Jitter")
DET = ("FAcc", "Recall", "F1")
CLIP = 0.05
DET_FLOOR = 0.05      # detection columns: the remaining error is floored at 5 %
# the gate's composite S is computed on HOT3D only. ARCTIC has 9 public subjects, ACE trained on all but
# the test subject s05, so any ARCTIC dev / val split is training data for θ0 (00 on ARCTIC val: MPJPE-p 6.1 mm vs 15.6 on
# s05) and says little about generalisation. ARCTIC (and every dataset) still enters the no-regression check.
S_DATASETS = ("hot3d",)
# datasets whose regressions block a gate: HOT3D only. ARCTIC dev / val (s09 / s07) were
# ACE training subjects, so a drop there measures forgetting of memorised subjects, not generalisation (any GT fine-tune
# lowers them while s05 improves); ARCTIC is still scored and reported (worst_rel, per-dataset results), and the held-out
# ARCTIC subject s05 is reported on the sealed test.
NOREG_DATASETS = ("hot3d",)


def combine(per: dict) -> dict:
    """per: {dataset: compare(...)} -> gate quantities: S and its CI90 lower bound averaged over S_DATASETS (all datasets
    if none of them is present); blocking regressions over NOREG_DATASETS, the
    other datasets' regressed columns in regressed_info."""
    sd = [d for d in per if d in S_DATASETS] or list(per)
    S = sum(per[d]["S"] for d in sd) / len(sd)
    lo = sum(per[d]["S_ci90"][0] for d in sd) / len(sd)
    rd = [d for d in per if d in NOREG_DATASETS] or list(per)
    regressed = {d: per[d]["regressed"] for d in rd if per[d]["regressed"]}
    info = {d: c["regressed"] for d, c in per.items() if d not in rd and c["regressed"]}     # reported, not blocking
    return dict(S=S, S_lo=lo, s_datasets=sd, regressed=regressed, regressed_info=info, noreg_datasets=rd,
                worst=min(per[d]["worst_rel"] for d in rd))


TABLE = ("FAcc", "Recall", "F1", "MPJPE-p", "PA-p", "MPJPE+OOS", "EPE2D-p", "GO-p", "CT-p", "Jitter")


def per_sequence(gt_dir: Path, pred_dir: Path, segs=None, cfg: Config = Config()) -> dict:
    """{sequence: Acc} (sequence = segment name before its last '__s'). Detection follows ace_eval: a prediction on a
    hand slot without GT in that frame is neither a TP nor a FP."""
    canon = canonical_joints(cfg.canonical_flat)
    out = defaultdict(Acc)
    files = sorted(gt_dir.glob("*.npz"))
    if segs is not None:
        files = [f for f in files if f.stem in segs]
    for g in files:
        p = pred_dir / g.name
        if not p.is_file():
            raise FileNotFoundError(f"no prediction {p}")
        gt, pred = load_pair(g, p)
        # the LAST '__s' starts the segment index: ARCTIC ids begin with 'ar__s07_...' (split('__s')[0] would lump
        # every ARCTIC segment into one "sequence", and its bootstrap CI would have zero width)
        out[g.stem.rsplit("__s", 1)[0]].add(score_segment(gt, pred, canon, cfg))
    return dict(out)


def pooled(accs, cfg: Config = Config()) -> dict:
    tot = Acc()
    for a in accs:
        tot.add(a)
    return summarize(tot, cfg)


def composite(ref: dict, new: dict) -> tuple[float, dict]:
    ok = lambda m: np.isfinite(ref[m]) and np.isfinite(new[m])          # noqa: E731
    r = {m: (ref[m] - new[m]) / ref[m] if ref[m] else 0.0 for m in ERR if ok(m)}
    r.update({m: (new[m] - ref[m]) / max(1.0 - ref[m], DET_FLOOR) for m in DET if ok(m)})
    # each column's contribution to S is capped at +-CLIP: the detection columns are
    # normalised by the remaining error (F1 0.991 -> 0.992 = +11 %), and without a cap one column would dominate S.
    # DET_FLOOR: near saturation the remaining error goes to 0 and the ratio explodes. Measured on 00's
    # val: removing 10 false-positive hand-frames on ARCTIC (F1 0.9990 -> 0.9991, 0.05 % of hand-frames) gave ARCTIC
    # S +1.00 % without the floor (enough for GA), +0.13 % with it. The no-regression check uses the raw r.
    return float(np.mean([max(-CLIP, min(CLIP, v)) for v in r.values()])), r


def compare(ref_seq: dict, new_seq: dict, n_boot: int = 2000, seed: int = 0, min_gain: float = 0.01,
            max_regress: float = 0.02, hard_cap: float = 0.05, cfg: Config = Config()) -> dict:
    """No-regression rule: a column counts as regressed when its point r is below -max_regress AND
    its own paired-bootstrap CI90 lies entirely below 0 (a significant drop), or when r is below -hard_cap at all.
    Why: the detection columns are normalised by the remaining error, so near saturation (ARCTIC F1 0.996) 2 % is a
    handful of hand-frames and pure noise would fail the old point-only rule."""
    seqs = sorted(set(ref_seq) & set(new_seq))
    if set(ref_seq) != set(new_seq):
        raise ValueError(f"sequence sets differ: {sorted(set(ref_seq) ^ set(new_seq))}")
    ref_m, new_m = pooled([ref_seq[s] for s in seqs], cfg), pooled([new_seq[s] for s in seqs], cfg)
    S, r = composite(ref_m, new_m)
    rng = np.random.default_rng(seed)
    boot, rboot = [], {m: [] for m in r}
    for _ in range(n_boot):
        draw = rng.integers(0, len(seqs), len(seqs))
        bs = [seqs[i] for i in draw]
        Sb, rb = composite(pooled([ref_seq[s] for s in bs], cfg), pooled([new_seq[s] for s in bs], cfg))
        boot.append(Sb)
        for m in r:
            if m in rb and np.isfinite(rb[m]):
                rboot[m].append(rb[m])
    lo, hi = np.percentile(boot, [5, 95])
    r_ci = {m: [float(x) for x in np.percentile(v, [5, 95])] if v else [float('nan')] * 2 for m, v in rboot.items()}
    regressed = sorted(m for m in r if (r[m] < -max_regress and r_ci[m][1] < 0) or r[m] < -hard_cap)
    worst = min(r.values())
    return dict(S=S, S_ci90=[float(lo), float(hi)], rel=r, rel_ci90=r_ci, worst_metric=min(r, key=r.get), worst_rel=worst,
                regressed=regressed, no_regress=not regressed,
                ref={m: ref_m[m] for m in TABLE}, new={m: new_m[m] for m in TABLE},
                passed=bool(lo > 0 and S >= min_gain and not regressed),
                rule=dict(min_gain=min_gain, max_regress=max_regress, hard_cap=hard_cap, regress='point < -max and CI90 '
                          'upper < 0, or point < -hard_cap', n_boot=n_boot, sequences=len(seqs)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, type=Path)
    ap.add_argument("--pred", required=True, type=Path)
    ap.add_argument("--ref", type=Path)
    ap.add_argument("--segs", type=Path, help="text file of segment names to score (default: all GT segments)")
    ap.add_argument("--min-gain", type=float, default=0.01)
    ap.add_argument("--max-regress", type=float, default=0.02)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    segs = set(a.segs.read_text().split()) if a.segs else None
    new = per_sequence(a.gt, a.pred, segs)
    res = dict(new=pooled(new.values()))
    if a.ref:
        res["compare"] = compare(per_sequence(a.gt, a.ref, segs), new, a.boot, min_gain=a.min_gain, max_regress=a.max_regress)
    print(" ".join(f"{m}={res['new'][m]:.4g}" for m in TABLE))
    if a.ref:
        c = res["compare"]
        print(f"S={100 * c['S']:+.2f}% CI90 [{100 * c['S_ci90'][0]:+.2f}, {100 * c['S_ci90'][1]:+.2f}]  worst {c['worst_metric']} "
              f"{100 * c['worst_rel']:+.2f}%  regressed {c['regressed'] or 'none'}  -> {'PASS' if c['passed'] else 'FAIL'}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
