"""Score a directory of per-segment predictions against per-segment ground truth.

  python -m ace_eval.evaluate --gt GT_DIR --pred PRED_DIR --out result.json
         [--set box_dilate=1.1 --set epe_pool=hand ...] [--per-seg] [--missing-as-empty]

GT_DIR and PRED_DIR hold one <segment>.npz each (format in README.md); every GT
segment needs a prediction with the same name unless --missing-as-empty, which
scores a missing file as a method that emitted nothing (every on-screen hand a FN).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

from .mano import canonical_joints
from .metrics import Acc, Config, score_segment, summarize

GT_REQ = {"K": (3, 3), "W": (), "H": (), "valid": ("F", 2), "joints_cam": ("F", 2, 21, 3),
          "transl": ("F", 2, 3)}
GT_OPT = {"rot": ("F", 2, 3, 3), "verts_cam": ("F", 2, 778, 3)}
PRED_REQ = {"exists": ("F", 2), "joints_cam": ("F", 2, 21, 3), "transl": ("F", 2, 3)}
PRED_OPT = {"rot": ("F", 2, 3, 3), "verts_cam": ("F", 2, 778, 3), "joints_2d": ("F", 2, 21, 2)}


def _check(d: dict, req: dict, opt: dict, F: int, name: str) -> None:
    for k, shp in list(req.items()) + [(k, s) for k, s in opt.items() if k in d]:
        if k not in d:
            raise ValueError(f"{name}: missing key '{k}'")
        want = tuple(F if x == "F" else x for x in shp)
        if tuple(np.shape(d[k])) != want:
            raise ValueError(f"{name}: '{k}' has shape {np.shape(d[k])}, expected {want}")


def load_pair(gt_path: Path, pred_path: Path | None) -> tuple[dict, dict]:
    gt = dict(np.load(gt_path))
    F = gt["valid"].shape[0]
    _check(gt, GT_REQ, GT_OPT, F, str(gt_path))
    if pred_path is None:
        pred = {"exists": np.zeros((F, 2)), "joints_cam": np.full((F, 2, 21, 3), np.nan),
                "transl": np.full((F, 2, 3), np.nan)}
    else:
        pred = dict(np.load(pred_path))
        _check(pred, PRED_REQ, PRED_OPT, F, str(pred_path))
    return gt, pred


def evaluate(gt_dir: Path, pred_dir: Path, cfg: Config, missing_as_empty: bool = False,
             per_seg: bool = False) -> dict:
    canon = canonical_joints(cfg.canonical_flat)
    total = Acc()
    segs = {}
    gt_files = sorted(gt_dir.glob("*.npz"))
    if not gt_files:
        raise SystemExit(f"no *.npz in {gt_dir}")
    missing = []
    for g in gt_files:
        p = pred_dir / g.name
        if not p.exists():
            if not missing_as_empty:
                raise SystemExit(f"no prediction for {g.name} in {pred_dir} (use --missing-as-empty to score it as empty)")
            missing.append(g.stem)
            p = None
        gt, pred = load_pair(g, p)
        a = score_segment(gt, pred, canon, cfg)
        total.add(a)
        if per_seg:
            segs[g.stem] = summarize(a, cfg)
    out = summarize(total, cfg)
    out["segments"] = len(gt_files)
    out["missing_predictions"] = missing
    out["config"] = dataclasses.asdict(cfg)
    if per_seg:
        out["per_segment"] = segs
    return out


def _parse_set(items: list[str]) -> Config:
    cfg = Config()
    for it in items:
        k, v = it.split("=", 1)
        cur = getattr(cfg, k)          # AttributeError on unknown keys, deliberately
        setattr(cfg, k, type(cur)(v) if not isinstance(cur, bool) else v.lower() in ("1", "true", "yes"))
    return cfg


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, type=Path)
    ap.add_argument("--pred", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--per-seg", action="store_true")
    ap.add_argument("--missing-as-empty", action="store_true")
    a = ap.parse_args(argv)
    res = evaluate(a.gt, a.pred, _parse_set(a.set), a.missing_as_empty, a.per_seg)
    main_keys = ["FAcc", "Recall", "F1", "MPJPE-p", "PA-p", "MPJPE+OOS", "EPE2D-p", "GO-p", "CT-p", "Jitter"]
    print(" ".join(f"{k}={res[k]:.4g}" for k in main_keys))
    print("counts:", json.dumps(res["counts"]))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))
        print(f"-> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
