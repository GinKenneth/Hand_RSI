"""3D diagnosis for Loop A agents (numbers only), on a split the agent may see (dev).

  python -m eval.diagnose --gt GT_DIR --pred PRED_DIR [--vs PARENT_PRED_DIR] [--n 15] [--json out.json]

Re-runs the evaluator's frame matching (ace_eval.metrics, same rules and config) and keeps every hand instance:
  - the 10 table metrics + matched-only means (ace_eval)
  - wrist-aligned error per joint and per finger, on matched hands
  - misses (FN) by how many GT joints are in frame (edge hands 1-3 / 4-10 / 11-20 / 21), and false positives
  - as in the evaluator, a prediction on a hand slot without ground truth in that frame is neither TP nor FP; such
    predictions are counted apart (FP_holes_not_scored)
  - the worst matched hand-frames by MPJPE, CT (root translation) and GO (global orientation)
  - with --vs: the same per metric for the parent, and which hand-frames changed most
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rsi_paths  # noqa: E402,F401  (puts ace_eval on sys.path)
from ace_eval.evaluate import load_pair  # noqa: E402
from ace_eval.metrics import (Config, _match, dilate, geodesic_deg, joints_in_frame, proj_box,  # noqa: E402
                              wrist_mpjpe)

FINGERS = dict(thumb=[1, 2, 3, 4], index=[5, 6, 7, 8], middle=[9, 10, 11, 12], ring=[13, 14, 15, 16], pinky=[17, 18, 19, 20])
CFG = Config()


def instances(gt_dir: Path, pred_dir: Path) -> dict:
    """Per hand instance records over all segments."""
    tp, fn, fp, fp_holes = [], [], [], []
    for g in sorted(gt_dir.glob("*.npz")):
        gt, pred = load_pair(g, pred_dir / g.name)
        K = np.asarray(gt["K"], np.float64); W, H = int(gt["W"]), int(gt["H"])
        for f in range(gt["valid"].shape[0]):
            gts, preds = [], []
            for s in (0, 1):
                Jg = gt["joints_cam"][f, s]
                if gt["valid"][f, s] and np.isfinite(Jg).all():
                    vis = joints_in_frame(Jg, K, W, H, CFG.z_min)
                    if vis.any():
                        box = proj_box(gt["verts_cam"][f, s] if "verts_cam" in gt else Jg, K, W, H, CFG)
                        if box is not None:
                            gts.append(dict(side=s, box=dilate(box, CFG.box_dilate), vis=vis))
                Jp = pred["joints_cam"][f, s]
                if float(pred["exists"][f, s]) > CFG.exist_thr and np.isfinite(Jp).all() and \
                        joints_in_frame(Jp, K, W, H, CFG.z_min).any():
                    box = proj_box(pred["verts_cam"][f, s] if "verts_cam" in pred else Jp, K, W, H, CFG)
                    if box is not None:
                        if not bool(gt["valid"][f, s]):              # no ground truth on this slot: not scored
                            fp_holes.append(dict(seg=g.stem, frame=f, side=s))
                        else:
                            preds.append(dict(side=s, box=box))
            pairs, n_fp, fns = _match(gts, preds)
            matched_pred = {pi for _, pi in pairs}
            for pi, p in enumerate(preds):
                if pi not in matched_pred:
                    fp.append(dict(seg=g.stem, frame=f, side=p["side"]))
            for gi, pi in pairs:
                s = gts[gi]["side"]
                Jg, Jp = gt["joints_cam"][f, s], pred["joints_cam"][f, s]
                per_joint = np.linalg.norm((Jp - Jp[0]) - (Jg - Jg[0]), axis=-1) * 1e3
                rec = dict(seg=g.stem, frame=f, side=s, nvis=int(gts[gi]["vis"].sum()), mpjpe=float(per_joint.mean()),
                           per_joint=per_joint, ct=float(np.linalg.norm(pred["transl"][f, s] - gt["transl"][f, s])),
                           go=geodesic_deg(pred["rot"][f, s], gt["rot"][f, s]) if "rot" in pred else float("nan"))
                tp.append(rec)
            for gi in fns:
                fn.append(dict(seg=g.stem, frame=f, side=gts[gi]["side"], nvis=int(gts[gi]["vis"].sum())))
    return dict(tp=tp, fn=fn, fp=fp, fp_holes=fp_holes)


def summary(inst: dict, n: int = 15) -> dict:
    tp = inst["tp"]
    pj = np.stack([r["per_joint"] for r in tp]) if tp else np.zeros((0, 21))
    bins = ((1, 3), (4, 10), (11, 20), (21, 21))
    fn_nv = np.array([r["nvis"] for r in inst["fn"]], int)
    tp_nv = np.array([r["nvis"] for r in tp], int)
    out = dict(
        counts=dict(TP=len(tp), FN=len(inst["fn"]), FP=len(inst["fp"]), FP_holes_not_scored=len(inst.get("fp_holes", []))),
        per_joint_mm=[round(float(x), 2) for x in pj.mean(0)] if len(pj) else [],
        per_finger_mm={k: round(float(pj[:, v].mean()), 2) for k, v in FINGERS.items()} if len(pj) else {},
        wrist_aligned_mm=round(float(pj.mean()), 2) if len(pj) else None,
        miss_by_visible_joints={f"{a}-{b}": dict(missed=int(((fn_nv >= a) & (fn_nv <= b)).sum()),
                                                 recall=round(float(((tp_nv >= a) & (tp_nv <= b)).sum() /
                                                                    max(((tp_nv >= a) & (tp_nv <= b)).sum() +
                                                                        ((fn_nv >= a) & (fn_nv <= b)).sum(), 1)), 4))
                                for a, b in bins},
        fp_by_side=dict(left=sum(r["side"] == 0 for r in inst["fp"]), right=sum(r["side"] == 1 for r in inst["fp"])),
    )
    key = lambda k: sorted(tp, key=lambda r: -r[k] if np.isfinite(r[k]) else 0)[:n]   # noqa: E731
    for k in ("mpjpe", "ct", "go"):
        out[f"worst_{k}"] = [dict(seg=r["seg"], frame=r["frame"], side="LR"[r["side"]], value=round(r[k], 3), nvis=r["nvis"])
                             for r in key(k)]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, type=Path)
    ap.add_argument("--pred", required=True, type=Path)
    ap.add_argument("--vs", type=Path)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    from eval.score import per_sequence, pooled, compare, TABLE
    res = dict(metrics=pooled(per_sequence(a.gt, a.pred).values()))
    inst = instances(a.gt, a.pred)
    res["detail"] = summary(inst, a.n)
    if a.vs:
        res["vs_parent"] = compare(per_sequence(a.gt, a.vs), per_sequence(a.gt, a.pred), n_boot=500)
        par = {(r["seg"], r["frame"], r["side"]): r for r in instances(a.gt, a.vs)["tp"]}
        d = [(r["mpjpe"] - par[k]["mpjpe"], r) for r in inst["tp"] if (k := (r["seg"], r["frame"], r["side"])) in par]
        d.sort(key=lambda x: x[0])
        res["changed_most"] = dict(
            better=[dict(seg=r["seg"], frame=r["frame"], side="LR"[r["side"]], delta_mm=round(x, 2)) for x, r in d[:a.n]],
            worse=[dict(seg=r["seg"], frame=r["frame"], side="LR"[r["side"]], delta_mm=round(x, 2)) for x, r in d[::-1][:a.n]])
    m = res["metrics"]
    print(" ".join(f"{k}={m[k]:.4g}" for k in TABLE))
    dt = res["detail"]
    print("counts", dt["counts"], "| wrist-aligned matched mm", dt["wrist_aligned_mm"])
    print("per finger mm", dt["per_finger_mm"])
    print("misses by visible joints", dt["miss_by_visible_joints"], "| FP", dt["fp_by_side"])
    for k in ("mpjpe", "ct", "go"):
        print(f"worst {k}:", ", ".join(f"{w['seg']}#{w['frame']}{w['side']}={w['value']}" for w in dt[f"worst_{k}"][:8]))
    if a.vs:
        c = res["vs_parent"]
        print(f"vs parent: S={100 * c['S']:+.2f}% CI90 [{100 * c['S_ci90'][0]:+.2f}, {100 * c['S_ci90'][1]:+.2f}] "
              + " ".join(f"{k}{100 * v:+.1f}%" for k, v in c["rel"].items()))
    if a.json:
        a.json.write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))


if __name__ == "__main__":
    main()
