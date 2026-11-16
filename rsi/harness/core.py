"""Harness framework: post-processing of the official ACE outputs, one 81-frame
protocol segment at a time.

Input per segment (a capture, see capture/run_ace.py): the raw, UNGATED predict_video() arrays (DUMP_KEYS) of the
network θ, the segment's frames (read on demand) and its intrinsics. Output: the evaluator's prediction npz
(ace_eval README). The harness is an ordered chain of modules (harness/<tree>/chain.json + modules/*.py). The empty
chain is 00 itself: MANO forward of ACE's own parameters, existence = exists_3d, 2D anchors = direct_joints2d.

Protocol rule: a module sees ONE segment only (its 81 frames, its K, the raw network outputs on those frames).
No other segment, no sequence id, no frame number, no GT. That keeps the comparison with ACE fair (ACE decodes each
segment on its own) and makes special-casing impossible.

Module interface (modules/<name>.py):
    def apply(state: dict, seg: Segment, params: dict) -> dict
  state keys (all numpy, slot 0 = left, 1 = right, camera frame, metres, px in the evaluated image):
    go (F,2,3,3)  hp (F,2,15,3,3)  betas (F,2,10)  transl (F,2,3)      MANO parameters, flat_hand_mean=False (ACE)
    joints_cam (F,2,21,3)  verts_cam (F,2,778,3)  joints_2d (F,2,21,2)  exists (F,2)
  A module that changes MANO parameters calls seg.mano(state) to recompute joints_cam/verts_cam; one that changes
  joints_cam directly must keep transl/rot consistent itself (the evaluator scores joints_cam, transl, rot, verts).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

_MODELS = None


def mano_models():
    global _MODELS
    if _MODELS is None:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import rsi_paths  # noqa: F401  (puts ace_eval on sys.path)
        from ace_eval.mano import load_models
        _MODELS = load_models(flat_hand_mean=False)
    return _MODELS


class Segment:
    """One protocol segment: raw network outputs, intrinsics, and lazily its RGB frames."""

    def __init__(self, raw: dict, K, W: int, H: int, video: str | None = None, start: int = 0, n: int | None = None):
        self.raw, self.K, self.W, self.H = raw, np.asarray(K, np.float64), int(W), int(H)
        self.n = int(n if n is not None else len(raw["exists_3d"]))
        self._video, self._start, self._frames = video, int(start), None

    def frames(self) -> np.ndarray:
        """(F, H, W, 3) uint8 RGB of this segment only."""
        if self._frames is None:
            import cv2
            cap = cv2.VideoCapture(self._video)
            cap.set(cv2.CAP_PROP_POS_FRAMES, self._start)
            out = []
            for _ in range(self.n):
                ok, f = cap.read()
                if not ok:
                    break
                out.append(f[:, :, ::-1])
            cap.release()
            self._frames = np.stack(out)
        return self._frames

    def project(self, X: np.ndarray) -> np.ndarray:
        z = np.maximum(X[..., 2:3], 1e-6)
        return np.concatenate([self.K[0, 0] * X[..., :1] / z + self.K[0, 2], self.K[1, 1] * X[..., 1:2] / z + self.K[1, 2]], -1)

    def mano(self, state: dict) -> dict:
        """Recompute joints_cam / verts_cam from go, hp, betas, transl (ACE's MANO convention)."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import rsi_paths  # noqa: F401
        from ace_eval.mano import forward
        F = self.n
        J = np.full((F, 2, 21, 3), np.nan, np.float32)
        V = np.full((F, 2, 778, 3), np.nan, np.float32)
        for s in (0, 1):
            J[:, s], V[:, s] = forward(mano_models(), s, state["go"][:, s], state["hp"][:, s], state["betas"][:, s])
        t = state["transl"].astype(np.float32)[:, :, None]
        state["joints_cam"], state["verts_cam"] = J + t, V + t
        return state


def initial_state(seg: Segment) -> dict:
    """The empty chain = 00: ACE's own parameters, existence head, direct 2D anchors."""
    r, F = seg.raw, seg.n
    st = {"go": np.asarray(r["global_orient"], np.float64)[:F].reshape(F, 2, 3, 3),
          "hp": np.asarray(r["hand_pose"], np.float64)[:F].reshape(F, 2, 15, 3, 3),
          "betas": np.asarray(r["betas"], np.float64)[:F].reshape(F, 2, -1),
          "transl": np.asarray(r["cam_trans"], np.float32)[:F].reshape(F, 2, 3),
          "exists": np.asarray(r["exists_3d"], np.float32)[:F].reshape(F, 2),
          "joints_2d": np.asarray(r["direct_joints2d"], np.float32)[:F].reshape(F, 2, 21, 2) * np.array([seg.W, seg.H], np.float32)}
    return seg.mano(st)


def load_chain(tree: Path):
    """[(name, apply, params)] from <tree>/chain.json: {"modules": [{"name": "x", "params": {...}, "enabled": true}]}."""
    tree = Path(tree)
    spec = json.loads((tree / "chain.json").read_text()) if (tree / "chain.json").is_file() else {"modules": []}
    chain = []
    for m in spec.get("modules", []):
        if not m.get("enabled", True):
            continue
        p = tree / "modules" / f"{m['name']}.py"
        sp = importlib.util.spec_from_file_location(f"harness_mod_{m['name']}", p)
        mod = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(mod)
        chain.append((m["name"], mod.apply, dict(m.get("params", {}))))
    return chain


def to_pred(state: dict) -> dict:
    """Evaluator prediction npz (ace_eval README)."""
    return {"exists": state["exists"].astype(np.float32), "joints_cam": state["joints_cam"].astype(np.float32),
            "transl": state["transl"].astype(np.float32), "rot": state["go"].astype(np.float32),
            "verts_cam": state["verts_cam"].astype(np.float32), "joints_2d": state["joints_2d"].astype(np.float32),
            # extra keys (ignored by the evaluator): the MANO parameters, Loop B's teacher targets
            "hp": state["hp"].astype(np.float32), "betas": state["betas"].astype(np.float32)}


def run_segment(chain, seg: Segment) -> dict:
    st = initial_state(seg)
    for name, apply, params in chain:
        st = apply(st, seg, params)
        if st is None:
            raise RuntimeError(f"module {name} returned None")
    return to_pred(st)
