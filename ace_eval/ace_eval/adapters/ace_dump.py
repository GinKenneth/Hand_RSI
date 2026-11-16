"""Convert ACE-Ego-Hand segment dumps (infer_video.py --split_segments, or raw
predict_video() arrays) into the evaluator's prediction npz.

  python -m ace_eval.adapters.ace_dump --inp DUMP_DIR --out PRED_DIR [--kfree] [--joints mano|direct]

Accepted pickle/npz keys (either naming):
  packed (infer_video): go (F,2,1,3,3)  hp (F,2,15,3,3)  betas (F,2,10)  cam_trans (F,2,3)
                        exists_2d (F,2)  joints_2d (F,2,21,2)  joints_cam_direct  intrinsics
  raw (predict_video):  global_orient  hand_pose  betas  cam_trans  exists_2d  direct_joints2d
                        direct_joints_cam
Existence score: ACE emits an existence and a visibility score per hand ("per-frame existence and
visibility flags", Sec. 3; trained by L_pres). exists_3d is the existence head; it stays on for
occluded and out-of-sight hands, which ACE supervises explicitly. exists_2d is the in-frame
visibility head, and infer_video.py gates its *deployment* claim on it. Appendix A.1 defines a
detection as "existence score exceeds 0.5" and applies the on-screen gate to predictions so
that an out-of-sight claim is not a FP; both make sense only for the existence head. Default:
--exists 3d: the existence head (exists_3d), which A.1's detection rule refers to; exists_2d is the in-frame
visibility head.
infer_video NaNs every pose row whose claim is False; the GT-gated MPJPE+OOS
pass then has nothing to score there, so dump UNGATED arrays (raw predict_video
output) when MPJPE+OOS / MPJPE_OOS matter.

2D anchors: ACE's direct_joints2d is a sigmoid output, i.e. normalised to the
image; it is scaled by (W, H). With --kfree no anchors are written and the
evaluator re-projects the camera joints under the GT intrinsics (Appendix A.1).
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from ..mano import forward, load_models

ALIASES = {"go": "global_orient", "hp": "hand_pose", "joints_2d": "direct_joints2d",
           "joints_cam_direct": "direct_joints_cam"}


def _load(p: Path) -> dict:
    if p.suffix == ".npz":
        d = dict(np.load(p, allow_pickle=True))
    else:
        with open(p, "rb") as f:
            d = pickle.load(f)
    return {ALIASES.get(k, k): v for k, v in d.items()}


def convert(d: dict, models: dict, W: int | None, H: int | None, kfree: bool, joints: str,
            exists: str = "3d") -> dict:
    intr = d.get("intrinsics")
    if isinstance(intr, np.ndarray) and intr.dtype == object:
        intr = intr.item()
    if isinstance(intr, dict):
        W = W or int(round(float(intr["image_width"])))
        H = H or int(round(float(intr["image_height"])))
    if W is None or H is None:
        raise ValueError("image size unknown: pass --W/--H")
    go = np.asarray(d["global_orient"], np.float64)
    F = go.shape[0]
    go = go.reshape(F, 2, 3, 3) if go.size == F * 2 * 9 else go.reshape(F, 2, 3)
    hp = np.asarray(d["hand_pose"], np.float64)
    betas = np.asarray(d["betas"], np.float64).reshape(F, 2, -1)
    tau = np.asarray(d["cam_trans"], np.float32).reshape(F, 2, 3)
    J = np.full((F, 2, 21, 3), np.nan, np.float32)
    V = np.full((F, 2, 778, 3), np.nan, np.float32)
    for s in (0, 1):
        J[:, s], V[:, s] = forward(models, s, go[:, s], hp[:, s], betas[:, s])
    J += tau[:, :, None]
    V += tau[:, :, None]
    if joints == "direct":
        J = np.asarray(d["direct_joints_cam"], np.float32).reshape(F, 2, 21, 3)
    out = {"exists": np.asarray(d[f"exists_{exists}"], np.float32).reshape(F, 2),
           "joints_cam": J, "transl": tau, "verts_cam": V}
    if go.shape[-2:] == (3, 3):
        out["rot"] = go.astype(np.float32)
    else:
        from scipy.spatial.transform import Rotation
        r = np.full((F, 2, 3, 3), np.nan, np.float32)
        ok = np.isfinite(go).all(-1)
        r[ok] = Rotation.from_rotvec(go[ok]).as_matrix()
        out["rot"] = r
    if not kfree and "direct_joints2d" in d:
        j2 = np.asarray(d["direct_joints2d"], np.float32).reshape(F, 2, 21, 2)
        out["joints_2d"] = j2 * np.array([W, H], np.float32)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inp", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--W", type=int)
    ap.add_argument("--H", type=int)
    ap.add_argument("--kfree", action="store_true")
    ap.add_argument("--joints", choices=("mano", "direct"), default="mano")
    ap.add_argument("--exists", choices=("3d", "2d"), default="3d", help="existence head (3d) or in-frame visibility head (2d)")
    a = ap.parse_args(argv)
    models = load_models(flat_hand_mean=False)
    a.out.mkdir(parents=True, exist_ok=True)
    files = sorted(list(a.inp.glob("*.pkl")) + list(a.inp.glob("*.npz")))
    for p in files:
        np.savez_compressed(a.out / f"{p.stem}.npz", **convert(_load(p), models, a.W, a.H, a.kfree, a.joints, a.exists))
    print(f"{len(files)} segment(s) -> {a.out}")


if __name__ == "__main__":
    main()
