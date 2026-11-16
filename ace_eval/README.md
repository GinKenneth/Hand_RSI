# ace_eval: the ACE-Ego-Hand evaluation protocol

A re-implementation of the evaluation protocol of ACE-Ego-Hand (arXiv 2608.20308, Appendix A.1; equation numbers
(S1)-(S11) below refer to it). The ACE-Ego-Hand repository does not include an evaluator yet; this package implements
the protocol from the paper's text. All HandRSI numbers are computed with it.

```
cd ace_eval
python -m unittest -v tests.test_metrics tests.test_adapter          # synthetic cases with hand-computed answers
python -m ace_eval.adapters.ace_dump --inp ACE_DUMPS/ --out PRED/     # raw ACE-Ego-Hand outputs -> prediction npz
python -m ace_eval.evaluate --gt GT/ --pred PRED/ --out result.json [--per-seg]
```

MANO: set `ACE_EVAL_MANO_DIR` to the converted MANO models (ACE-Ego-Hand's `scripts/convert_mano_pkls.py`), or put
them in `ace_eval/data/mano/`.

```
 GT/<seg>.npz ────────┐
                      ├─> evaluate.py ─> score_segment() per segment ─> pooled counts ─> summarize() ─> result.json
 PRED/<seg>.npz ──────┘          ^
   ^                             └─ mano.canonical_joints(): the (S8) placeholder hand
   └─ adapters/ace_dump.py  (ACE outputs: MANO parameters -> joints / vertices)
```

## Input format: one `.npz` per segment, with the same file name in GT/ and PRED/

- **Slots:** slot 0 is the left hand, slot 1 the right hand.
- **Units and frame:** camera frame, metres. Pixels refer to the evaluated image (W x H) under intrinsics K.
- **Joints:** 21 joints in OpenPose order (wrist, then thumb, index, middle, ring, pinky). This is ACE-Ego-Hand's
  convention: the 16 smplx MANO joints plus fingertip vertices 744/320/443/554/671, remapped with `MANO_TO_OP`
  (`ace_eval/mano.py`); `tests/test_adapter.py` checks it against ACE-Ego-Hand's own MANO forward.
- **MANO:** smplx with `flat_hand_mean=False`, ACE-Ego-Hand's setting for HOT3D, ARCTIC and HOI4D.

**Ground truth**

| key | shape | meaning |
|---|---|---|
| `K` | (3,3) | intrinsics of the evaluated image (HOT3D: fx = fy = 207.8, cx = cy = 239.8) |
| `W`, `H` | () | image size (HOT3D 480 x 480, ARCTIC 672 x 480, HOI4D 832 x 480) |
| `valid` | (F,2) bool | an annotation exists for this hand in this frame, in view or not |
| `joints_cam` | (F,2,21,3) | camera-frame joints, J + tau |
| `transl` | (F,2,3) | MANO root translation tau (not the wrist position) |
| `rot` | (F,2,3,3) | MANO global orientation in the camera frame (optional; needed for GO-p) |
| `verts_cam` | (F,2,778,3) | camera-frame MANO vertices (optional; the matching boxes are projected-mesh boxes, else joints) |

**Prediction**

| key | shape | meaning |
|---|---|---|
| `exists` | (F,2) | existence score; the hand is active when the score is > 0.5 |
| `joints_cam` | (F,2,21,3) | camera-frame joints, **ungated**: the evaluator applies the 0.5 gate itself, and the MPJPE+OOS pass scores every slot. NaN = nothing emitted |
| `transl` | (F,2,3) | predicted tau |
| `rot` | (F,2,3,3) | optional; needed for GO-p |
| `verts_cam` | (F,2,778,3) | optional; used for the matching boxes |
| `joints_2d` | (F,2,21,2) px | optional: the model's own 2D anchors; without them EPE2D re-projects `joints_cam` under the GT `K` |

## Metrics (the 10 columns of the tables)

| metric | definition | population | penalty for a missed on-screen hand |
|---|---|---|---|
| on-screen gate | (S1): any joint inside [0,W) x [0,H) at z > 1 cm; applied to GT and predictions | - | - |
| FAcc | (S3): frames with FP + FN = 0 | all frames | - |
| Recall, F1 | (S2): pooled counts over both sides | on-screen GT hands | - |
| MPJPE-p | (S4): wrist-aligned, mm | TP + FN hands | canonical hand |
| PA-p | (S5): Procrustes-aligned (with scale), mm | TP + FN hands | canonical hand |
| MPJPE+OOS | (S9)-(S11): wrist-aligned error in a second, GT-gated pass with no existence gate and no matching, mm | all annotated hand-frames | canonical hand |
| EPE2D-p | (S6): own 2D anchors vs GT, over joints whose GT is in frame, px | TP + FN joints | image diagonal per joint |
| GO-p | geodesic angle of the global rotation, deg | TP + FN hands | angle(I, R_gt) |
| CT-p | (S7): norm of tau_hat - tau, m | TP + FN hands | norm of tau_gt |
| Jitter | second difference of (wrist-relative joints + tau_hat), mm/frame^2 | matched frames; maximal runs of >= 3 frames inside a segment | - |

**Matching** (A.1 "Detection"):
- Active, on-screen predictions are matched by the overlap between the projected-mesh box of the prediction and the
  GT box enlarged by 10%. Each prediction takes its highest-overlap GT hand over both sides; zero overlap or a best
  overlap on the opposite side is a FP; of two same-side predictions on one GT hand the higher overlap wins.
- A prediction that fails the on-screen gate is not a FP.
- **Hands without ground truth are not scored.** A prediction on a hand slot whose GT is not annotated in that frame
  (`valid` = False) is neither a TP nor a FP. HOT3D has untracked stretches where the hand is plainly visible; without
  this rule a correct detection there would count as a false positive. A hand that is annotated but out of sight keeps
  its GT, so a hand pushed into the image by a wrong depth is still a FP. Only FAcc and F1 depend on this rule: such a
  prediction can never be a TP, so Recall and all pose columns are unaffected.

**Canonical placeholder** (S8): identity global orientation, `hand_pose` = 0, `betas` = 0, tau = 0
(computed from MANO on first use and cached in `ace_eval/canonical_mano.npz`).

`result.json` also reports `extra` (Precision, MPJPE_OOS, MPJPE_IV, unpenalised matched-only means) and `counts`
(TP / FP / FN, in-view and out-of-sight hand-frames, missing predictions).

## Choices the paper leaves open

Each is a field of `metrics.Config` (`--set key=value`); the defaults were fixed before any run.

| field | default | alternative | why |
|---|---|---|---|
| `box_dilate` | 1.10 (width and height x1.1 about the centre) | 1.2 | "dilated by 10%" |
| `box_clip` | False | clip boxes to the image | clipping is not mentioned |
| `pa_scale` | True | rigid only | standard PA-MPJPE |
| `epe_pool` | `joint` | `hand` | EPE2D-p is averaged over "on-screen joints"; per-hand pooling is also incompatible with the paper's HaWoR row |
| `jitter_norm` | `joint_mean` | `frobenius` | per-joint norms as in MPJPE; with `frobenius` the GT itself would score about 4x the paper's ACE value |
| `jitter_pool` | `frame` | `run` | runs are weighted by their length |
| `oos_missing` | `canonical` | `skip` | MPJPE+OOS averages over all annotated hands, so an empty slot has to cost something |
| `canonical_flat` | False (MANO mean pose) | flat hand | "zero articulation" under `flat_hand_mean=False` |
| aggregation | pooled over all segments | per-segment mean | the populations are hands / joints / frames |

## How close it is to ACE-Ego-Hand's numbers

There is no official evaluator to compare against, so two checks are indirect:
- The released K-given ACE-Ego-Hand checkpoint, run through `rsi/capture` and scored here, gives MPJPE-p 15.58 / 12.97 /
  23.45 mm on our ARCTIC / HOT3D / HOI4D test splits; ACE-Ego-Hand reports 15.26 / 12.89 / 23.03 on its own splits.
- Nine baselines re-run under ACE-Ego-Hand's Appendix A protocol (`baselines/`) give MPJPE-p close to the values
  ACE-Ego-Hand quotes for most of them (ARCTIC: HaMeR 29.7 vs 29.2, InterWild 31.3 vs 30.8).

Our splits are not ACE-Ego-Hand's, so neither check isolates the evaluator; the unit tests above check the formulas.

## Baselines

`baselines/` (repository root) re-runs the baselines of ACE-Ego-Hand's Table 1 and scores them with this package;
`baselines/README.md` has the protocol; the results are in the top-level README. `python -m ace_eval.baselines.run_wilor` is an earlier,
lighter WiLoR runner (`wilor_final.ckpt` behind its own YOLO detector, through `wilor_mini`; weights in
`$WILOR_MINI_ROOT`); the WiLoR row in `baselines/` comes from the official repository instead.
