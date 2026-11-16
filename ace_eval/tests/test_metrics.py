"""Synthetic checks: every metric against a value computed by hand from its definition.

  cd ace_eval && python -m unittest -v tests.test_metrics
"""
from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from ace_eval.mano import canonical_joints
from ace_eval.metrics import (Config, joints_in_frame, procrustes, proj_box, score_segment,
                              summarize, wrist_mpjpe)

K = np.array([[207.8, 0, 239.8], [0, 207.8, 239.8], [0, 0, 1.0]])   # HOT3D protocol camera
W = H = 480
CANON = canonical_joints(False)
CFG = Config()
TAU = np.array([[0.10, 0.08, 0.35], [-0.10, 0.08, 0.35]])           # left / right MANO roots


def make_gt(F=10, seed=0):
    """Two hands, slowly rotating; joints = R @ canonical + tau, verts omitted (box from joints)."""
    rng = np.random.default_rng(seed)
    rot = np.zeros((F, 2, 3, 3))
    J = np.zeros((F, 2, 21, 3))
    tr = np.zeros((F, 2, 3))
    for f in range(F):
        for s in (0, 1):
            R = Rotation.from_rotvec(rng.normal(0, 0.2, 3)).as_matrix()
            rot[f, s] = R
            tr[f, s] = TAU[s] + np.array([0.002 * f, 0, 0])
            J[f, s] = CANON[s] @ R.T + tr[f, s]
    return {"K": K, "W": np.array(W), "H": np.array(H), "valid": np.ones((F, 2), bool),
            "joints_cam": J, "transl": tr, "rot": rot}


def perfect(gt):
    F = gt["valid"].shape[0]
    return {"exists": np.ones((F, 2)), "joints_cam": gt["joints_cam"].copy(),
            "transl": gt["transl"].copy(), "rot": gt["rot"].copy()}


def run(gt, pred, cfg=CFG):
    return summarize(score_segment(gt, pred, CANON, cfg), cfg)


class Geometry(unittest.TestCase):
    def test_setup_hands_on_screen_and_separate(self):
        gt = make_gt()
        for s in (0, 1):
            self.assertTrue(joints_in_frame(gt["joints_cam"][0, s], K, W, H, 0.01).all())
        bl = proj_box(gt["joints_cam"][0, 0], K, W, H, CFG)
        br = proj_box(gt["joints_cam"][0, 1], K, W, H, CFG)
        self.assertTrue(bl[0] > br[2] or br[0] > bl[2], "left/right boxes must not overlap")

    def test_procrustes_removes_similarity(self):
        J = make_gt()["joints_cam"][0, 1]
        R = Rotation.from_rotvec([0.3, -0.5, 0.2]).as_matrix()
        X = 1.3 * J @ R.T + np.array([0.1, -0.2, 0.3])
        self.assertLess(np.abs(procrustes(X, J) - J).max(), 1e-9)

    def test_diagonals_match_paper(self):
        self.assertEqual(round(np.hypot(480, 480)), 679)     # HOT3D
        self.assertEqual(round(np.hypot(672, 480)), 826)     # ARCTIC


class Metrics(unittest.TestCase):
    def test_perfect_prediction(self):
        gt = make_gt()
        r = run(gt, perfect(gt))
        self.assertEqual((r["FAcc"], r["Recall"], r["F1"]), (1.0, 1.0, 1.0))
        for k in ("MPJPE-p", "PA-p", "MPJPE+OOS", "EPE2D-p", "GO-p", "CT-p"):
            self.assertAlmostEqual(r[k], 0.0, places=5, msg=k)
        self.assertEqual(r["counts"]["TP"], 20)

    def test_missed_hand_charged_canonical(self):
        gt = make_gt(F=10)
        pred = perfect(gt)
        pred["exists"][:4, 1] = 0.2                                    # right hand missed on 4 frames
        r = run(gt, pred)
        exp_mpjpe = sum(wrist_mpjpe(CANON[1], gt["joints_cam"][f, 1]) * 1e3 for f in range(4)) / 20
        exp_ct = sum(np.linalg.norm(gt["transl"][f, 1]) for f in range(4)) / 20
        exp_go = sum(np.degrees(np.linalg.norm(Rotation.from_matrix(gt["rot"][f, 1]).as_rotvec()))
                     for f in range(4)) / 20
        self.assertEqual((r["counts"]["TP"], r["counts"]["FN"], r["counts"]["FP"]), (16, 4, 0))
        self.assertAlmostEqual(r["Recall"], 0.8)
        self.assertAlmostEqual(r["FAcc"], 0.6)
        self.assertAlmostEqual(r["MPJPE-p"], exp_mpjpe, places=6)
        self.assertAlmostEqual(r["CT-p"], exp_ct, places=6)
        self.assertAlmostEqual(r["GO-p"], exp_go, places=4)
        # EPE2D-p: all 21 joints on screen -> 4*21 joints at the diagonal over 20*21 joints
        self.assertAlmostEqual(r["EPE2D-p"], 4 * 21 * np.hypot(W, H) / (20 * 21), places=6)
        self.assertAlmostEqual(r["extra"]["MPJPE_matched"], 0.0, places=6)

    def test_side_swap_is_fp_and_fn(self):
        gt = make_gt(F=1)
        pred = perfect(gt)
        for k in ("joints_cam", "transl", "rot"):
            pred[k] = pred[k][:, ::-1].copy()
        r = run(gt, pred)
        self.assertEqual((r["counts"]["TP"], r["counts"]["FP"], r["counts"]["FN"]), (0, 2, 2))
        self.assertEqual(r["FAcc"], 0.0)

    def test_offscreen_prediction_not_fp(self):
        gt = make_gt(F=1)
        pred = perfect(gt)
        pred["joints_cam"][0, 1, :, 2] = -0.5                          # behind the camera
        r = run(gt, pred)
        self.assertEqual((r["counts"]["TP"], r["counts"]["FP"], r["counts"]["FN"]), (1, 0, 1))

    def test_spurious_hand_is_fp(self):
        gt = make_gt(F=1)
        pred = perfect(gt)
        gt["joints_cam"][0, 0, :, 2] = -0.5                            # GT hand annotated, out of sight (behind the camera)
        r = run(gt, pred)                                              # the prediction puts it in the image
        self.assertEqual((r["counts"]["TP"], r["counts"]["FP"], r["counts"]["FN"]), (1, 1, 0))

    def test_prediction_on_unannotated_hand_not_scored(self):
        gt = make_gt(F=1)
        gt["valid"][0, 0] = False                                      # no GT annotation for this hand in this frame
        r = run(gt, perfect(gt))
        self.assertEqual((r["counts"]["TP"], r["counts"]["FP"], r["counts"]["FN"]), (1, 0, 0))
        self.assertEqual(r["FAcc"], 1.0)

    def test_rotation_about_wrist(self):
        gt = make_gt(F=1)
        pred = perfect(gt)
        Rd = Rotation.from_rotvec([0, 0, np.radians(30)]).as_matrix()
        J = gt["joints_cam"][0, 1]
        pred["joints_cam"][0, 1] = (J - J[0]) @ Rd.T + J[0]
        pred["rot"][0, 1] = Rd @ gt["rot"][0, 1]
        r = run(gt, pred)
        self.assertAlmostEqual(r["extra"]["GO_matched"], 15.0, places=4)   # 30 deg on 1 of 2 hands
        self.assertGreater(r["MPJPE-p"], 1.0)
        self.assertAlmostEqual(r["PA-p"], 0.0, places=5)
        self.assertAlmostEqual(r["CT-p"], 0.0, places=9)

    def test_translation_error(self):
        gt = make_gt(F=1)
        pred = perfect(gt)
        d = np.array([0.003, 0.004, 0.0])                               # 5 mm sideways
        pred["joints_cam"][0, 0] += d
        pred["transl"][0, 0] += d
        r = run(gt, pred)
        self.assertAlmostEqual(r["CT-p"], 0.005 / 2, places=9)
        self.assertAlmostEqual(r["MPJPE-p"], 0.0, places=6)            # wrist-aligned

    def test_epe2d_uses_own_anchors(self):
        gt = make_gt(F=2)
        pred = perfect(gt)
        from ace_eval.metrics import project
        pred["joints_2d"] = project(gt["joints_cam"], K)[0] + np.array([3.0, 4.0])
        r = run(gt, pred)
        self.assertAlmostEqual(r["EPE2D-p"], 5.0, places=6)
        del pred["joints_2d"]                                           # K-free: re-projection
        self.assertAlmostEqual(run(gt, pred)["EPE2D-p"], 0.0, places=6)

    def test_epe2d_only_in_frame_joints(self):
        gt = make_gt(F=1)
        gt["joints_cam"][0, 1, 15:, 0] = -0.9                           # 6 right-hand joints leave the image
        vis = joints_in_frame(gt["joints_cam"][0, 1], K, W, H, 0.01)
        self.assertEqual(int(vis.sum()), 15)
        pred = perfect(gt)
        pred["exists"][0, 1] = 0                                        # miss it
        r = run(gt, pred)
        self.assertAlmostEqual(r["EPE2D-p"], 15 * np.hypot(W, H) / (21 + 15), places=6)

    def test_jitter_constant_acceleration_and_runs(self):
        F = 12
        gt = make_gt(F=F)
        a = np.array([0.001, 0.0, 0.0])                                 # 1 mm / frame^2
        for f in range(F):
            for s in (0, 1):
                gt["joints_cam"][f, s] = CANON[s] + TAU[s] + 0.5 * a * f * f
                gt["transl"][f, s] = TAU[s] + 0.5 * a * f * f
                gt["rot"][f, s] = np.eye(3)
        pred = perfect(gt)
        self.assertAlmostEqual(run(gt, pred)["Jitter"], 1.0, places=6)
        pred["exists"][5, :] = 0      # runs 0-4 (3 terms) and 6-11 (4 terms) on each hand
        pred["exists"][7, 0] = 0      # left: 0-4 (3), 6 (too short), 8-11 (2)
        r = run(gt, pred)
        self.assertEqual(r["counts"]["jitter_frames"], 3 + 4 + 3 + 2)
        self.assertAlmostEqual(r["Jitter"], 1.0, places=6)

    def test_oos_pass(self):
        gt = make_gt(F=4)
        gt["joints_cam"][2:, 1, :, 0] -= 2.0                            # right hand far off-screen
        pred = perfect(gt)
        pred["exists"][:] = 0.0                                         # pass 2 ignores existence
        r = run(gt, pred)
        self.assertEqual((r["counts"]["hand_frames_IV"], r["counts"]["hand_frames_OOS"]), (6, 2))
        self.assertAlmostEqual(r["MPJPE+OOS"], 0.0, places=6)
        pred["joints_cam"][2:, 1] = np.nan                              # nothing emitted -> canonical
        r = run(gt, pred)
        exp = np.mean([wrist_mpjpe(CANON[1], gt["joints_cam"][f, 1]) * 1e3 for f in (2, 3)])
        self.assertAlmostEqual(r["extra"]["MPJPE_OOS"], exp, places=6)
        self.assertAlmostEqual(r["MPJPE+OOS"], exp * 2 / 8, places=6)
        r = run(gt, pred, Config(oos_missing="skip"))
        self.assertTrue(np.isnan(r["extra"]["MPJPE_OOS"]))

    def test_box_dilation_rescues_near_miss(self):
        gt = make_gt(F=1)
        pred = perfect(gt)
        from ace_eval.metrics import dilate
        b = proj_box(gt["joints_cam"][0, 1], K, W, H, CFG)
        bd = dilate(b, CFG.box_dilate)
        # slide the right hand away from the left one until its box sits just left of the GT box
        lo, hi = 0.0, 0.2
        for _ in range(60):
            m = (lo + hi) / 2
            pb = proj_box(gt["joints_cam"][0, 1] - np.array([m, 0, 0]), K, W, H, CFG)
            lo, hi = (m, hi) if pb[2] > b[0] - 0.02 * (b[2] - b[0]) else (lo, m)
        d = np.array([-hi, 0, 0])
        pred["joints_cam"][0, 1] += d
        pred["transl"][0, 1] += d
        pb = proj_box(pred["joints_cam"][0, 1], K, W, H, CFG)
        self.assertTrue(bd[0] < pb[2] < b[0], "precondition: box between the raw and the dilated edge")
        self.assertEqual(run(gt, pred)["counts"]["TP"], 2)              # dilated GT box still overlaps
        self.assertEqual(run(gt, pred, Config(box_dilate=1.0))["counts"]["TP"], 1)


if __name__ == "__main__":
    unittest.main()
