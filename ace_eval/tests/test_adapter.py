"""ACE dump adapter: joint convention identical to ACE's own MANO forward, and an
end-to-end ACE-format segment that equals the ground truth scores zero error.

  cd ace_eval && python -m unittest -v tests.test_adapter
"""
from __future__ import annotations

import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from ace_eval.adapters.ace_dump import _load, convert
from ace_eval.evaluate import evaluate
from ace_eval.mano import MANO_DIR, forward, load_models
from ace_eval.metrics import Config, project

ACE_REPO = os.environ.get("ACE_OFFICIAL", "../rsi/capture/ace_official")   # clone of ggxxii/ACE-Ego-Hand
K = np.array([[207.8, 0, 239.8], [0, 207.8, 239.8], [0, 0, 1.0]])
W = H = 480


def random_params(F, rng):
    go = Rotation.from_rotvec(rng.normal(0, 0.4, (F * 2, 3))).as_matrix().reshape(F, 2, 3, 3)
    hp = Rotation.from_rotvec(rng.normal(0, 0.25, (F * 2 * 15, 3))).as_matrix().reshape(F, 2, 15, 3, 3)
    betas = rng.normal(0, 0.5, (F, 2, 10))
    tau = np.zeros((F, 2, 3))
    tau[:, 0] = [0.12, 0.06, 0.40]
    tau[:, 1] = [-0.12, 0.06, 0.40]
    tau += rng.normal(0, 0.005, (F, 2, 3))
    return go, hp, betas, tau


class Adapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = load_models(False)

    def test_matches_ace_mano_forward(self):
        import os
        os.environ["ACE_EGO_HAND_MANO_DIR"] = MANO_DIR
        sys.path.insert(0, ACE_REPO)
        from ace_ego_hand.mano_utils import mano_forward_batch_full
        go, hp, betas, _ = random_params(3, np.random.default_rng(1))
        for s in (0, 1):
            ours, _ = forward(self.models, s, go[:, s], hp[:, s], betas[:, s])
            ref, _ = mano_forward_batch_full(torch.tensor(go[:, s], dtype=torch.float32),
                                             torch.tensor(hp[:, s], dtype=torch.float32),
                                             torch.tensor(betas[:, s], dtype=torch.float32),
                                             self.models[s])
            self.assertLess(np.abs(ours - ref.detach().numpy()).max(), 1e-5)

    def test_end_to_end_ace_packed_dump(self):
        F = 8
        go, hp, betas, tau = random_params(F, np.random.default_rng(2))
        J = np.zeros((F, 2, 21, 3), np.float32)
        V = np.zeros((F, 2, 778, 3), np.float32)
        for s in (0, 1):
            J[:, s], V[:, s] = forward(self.models, s, go[:, s], hp[:, s], betas[:, s])
        J += tau[:, :, None]
        V += tau[:, :, None]
        gt = {"K": K, "W": np.array(W), "H": np.array(H), "valid": np.ones((F, 2), bool),
              "joints_cam": J, "transl": tau, "rot": go, "verts_cam": V}
        j2 = project(J, K)[0] / np.array([W, H])
        dump = {"go": go.reshape(F, 2, 1, 3, 3).astype(np.float32), "hp": hp.astype(np.float32),
                "betas": betas.astype(np.float32), "cam_trans": tau.astype(np.float32),
                "exists_2d": np.full((F, 2), 0.9, np.float32), "exists_3d": np.full((F, 2), 0.9, np.float32), "joints_2d": j2.astype(np.float32),
                "intrinsics": {"fx": 207.8, "fy": 207.8, "cx": 239.8, "cy": 239.8,
                               "image_width": W, "image_height": H}}
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / "gt").mkdir(); (td / "dump").mkdir(); (td / "pred").mkdir()
            np.savez(td / "gt" / "seg_000.npz", **gt)
            with open(td / "dump" / "seg_000.pkl", "wb") as f:
                pickle.dump(dump, f)
            np.savez(td / "pred" / "seg_000.npz",
                     **convert(_load(td / "dump" / "seg_000.pkl"), self.models, None, None, False, "mano"))
            r = evaluate(td / "gt", td / "pred", Config())
        self.assertEqual((r["F1"], r["FAcc"]), (1.0, 1.0))
        for k in ("MPJPE-p", "PA-p", "EPE2D-p", "GO-p", "MPJPE+OOS"):
            self.assertLess(r[k], 1e-2, msg=k)       # float32 round trip
        self.assertLess(r["CT-p"], 1e-6)


if __name__ == "__main__":
    unittest.main()
