# Your job: beat the state of the art in 3D egocentric hand pose by writing a better harness

You improve the post-processing ("harness") that runs after a frozen video network (ACE-Ego-Hand). It is scored with
the published ACE-Ego-Hand protocol on two egocentric datasets, HOT3D (480 x 480) and ARCTIC (672 x 480), by a fixed
evaluator you cannot change. Each improvement that passes
the gate is later distilled into the network, so real, general gains matter. Gains that only fit this data do not.

## What the harness is

`./tree` is your harness: `chain.json` (an ordered list of modules) plus `modules/<name>.py`. The starting tree is
EMPTY: it outputs exactly what the network predicts. That is the published ACE result.

    chain.json:  {"modules": [{"name": "my_module", "enabled": true, "params": {"alpha": 0.5}}, ...]}
    modules/my_module.py:
        def apply(state, seg, params):     # one 81-frame segment, returns the (modified) state
            ...
            return state

`state` (numpy arrays, F frames, slot 0 = left hand, 1 = right, camera frame in metres, pixels in the image):

| key | shape | meaning |
|---|---|---|
| `go`, `hp`, `betas` | (F,2,3,3), (F,2,15,3,3), (F,2,10) | MANO global orientation, hand pose (rotation matrices; mean-relative, flat_hand_mean=False), shape |
| `transl` | (F,2,3) | MANO root translation |
| `joints_cam`, `verts_cam` | (F,2,21,3), (F,2,778,3) | 21 joints in OpenPose order (wrist, thumb x4, index, middle, ring, pinky) and the mesh |
| `joints_2d` | (F,2,21,2) | the network's own 2D keypoints in pixels; used by EPE2D-p |
| `exists` | (F,2) | existence score; a hand counts as predicted when it is > 0.5 |

`seg` gives you:
- `seg.raw`: every raw network output of this segment, including `exists_2d` (in-frame visibility), `exists_3d`,
  `direct_joints_cam` and `direct_wrist_cam`;
- `seg.frames()`: the RGB frames, (F,H,W,3) uint8;
- `seg.K`, `seg.W`, `seg.H`: intrinsics and image size;
- `seg.project(X)`: 3D points to pixels;
- `seg.mano(state)`: recomputes `joints_cam` and `verts_cam` after you change `go`, `hp`, `betas` or `transl`.

If you change `joints_cam` directly, keep `transl`, `go` and `verts_cam` consistent, because all of them are scored.

The exact contract is in `{root}/harness/core.py`. Read it once.

## Rules (the gate enforces them; breaking one gets the submission rejected and flagged)

1. **One segment at a time.** A module sees one 81-frame segment: its frames, K and network outputs. It gets no other
   segment, no sequence name, no frame number and no ground truth. Do not try to recover them (e.g. from file paths).
   Do not key on which dataset a segment comes from either (image size, intrinsics): write behaviour that is general.
2. **No special cases.** Write general behaviour that would hold on any egocentric video. Never key on data
   properties that only identify this dataset.
3. **Deterministic and bounded.** At most about 3 s per segment on CPU. The GPU (`cuda`) may be used for an image
   model. Available read-only: torch, numpy, scipy, opencv, smplx (MANO), and the WiLoR hand model
   (`wilor_mini` under `$WILOR_MINI_ROOT`). Measured on HOT3D train recordings: WiLoR's image-space 2D keypoints (its
   crop camera mapped back to the image, as in `{ace_eval}/ace_eval/baselines/diag_wilor2d.py` `crop_2d`)
   are ~5.5 px from the GT, but its 3D joints re-projected through the true K are ~23 px off (weak-perspective depth
   at a wide field of view). Use the former as 2D image evidence. Replays run 16-32 workers, spread over the cards.
   Your session sees GPU cards 0-3 only (`CUDA_VISIBLE_DEVICES`; the gate replays on the other four). A module must
   still work when its model cannot be loaded on a busy card, but say so on stderr: a silent fallback hides errors.
4. Accuracy only. There are no speed submissions this time.
5. **Papers and open-source code are encouraged** (`lt lit`, `lt code`, below). Port ideas or code freely, but say in
   your submit note what you used (arXiv id, or repo + file) and the code's license. Never bring in model weights,
   datasets or anyone's precomputed results (especially on HOT3D / ARCTIC / HOI4D): the tree may contain only .py /
   .json / .md files, at most 1 MB, or the gate rejects it. No `pip install`. If an idea needs a new pretrained model,
   write that in notes.md: the research lead (Loop C) can propose it to the humans.

## The metrics (ACE protocol; you can only see them on the dev split)

| metric | meaning | better |
|---|---|---|
| FAcc | frames with no missed and no spurious hand | higher |
| Recall | on-screen GT hands that were detected | higher |
| F1 | detection F1 | higher |
| MPJPE-p (mm) | wrist-aligned 3D joint error | lower |
| PA-p (mm) | joint error after Procrustes alignment | lower |
| EPE2D-p (px) | 2D keypoint error | lower |
| GO-p (deg) | global orientation error | lower |
| CT-p (m) | root translation error | lower |
| Jitter | 3D acceleration | lower |
| MPJPE+OOS | wrist-aligned error including out-of-sight hands | lower |

Every "-p" metric charges each missed on-screen hand a penalty: a canonical hand at the camera origin, and the image
diagonal (HOT3D 679 px, ARCTIC 826 px) per joint for EPE2D. So a miss is very expensive, and a spurious hand costs FAcc
and F1.

The gate's composite **S** covers all 10 columns:
- each column's relative improvement is computed; for FAcc, Recall and F1 it is the fraction of the remaining error removed,
  with the remaining error counted as at least 5 % (near saturation a handful of frames must not look like a big gain);
- each column's contribution is capped at ±5 %, so no single column can dominate;
- S is the mean over the columns.

S is computed on HOT3D (the network was trained on every ARCTIC subject except the test one, so ARCTIC dev / val are
almost saturated). On BOTH datasets, no column may get significantly more than 2 % worse, nor more than 5 % worse at all.
So a change must help on HOT3D and must not hurt ARCTIC.

## Tools (`lt`, in your shell)

- `lt try`: run ./tree on all dev segments of both datasets, compare with the parent (S per dataset + mean, every
  metric), plus the 3D diagnosis. It takes a few minutes. `lt replay --seqs A,B` or `--ds arctic` runs on a subset.
- `lt diagnose [RUN] --vs parent`: per-joint and per-finger errors; misses by how many joints are in frame (edge
  hands); false positives; the worst hand-frames by MPJPE, CT and GO; what changed most.
- `lt bucket`: the shared record of ALL rounds: every submission of every agent (note, verdict, coarse S, dev metric
  deltas, `--show ID` for its diff), agents' notes, merge steps, round verdicts and the research lead's plans. Read it
  first; do not repeat a rejected idea unchanged.
- `lt ideas`: this round's idea cards from the research lead (Loop C), with evidence and references.
- `lt lit search "..."` / `lt lit read ARXIV_ID`: search arXiv and read papers (saved under refs/papers/).
- `lt code search "..."` / `lt code get OWNER/REPO`: find and download open-source code (source files only, under
  refs/code/). Useful places to start: hand pose estimation and tracking (HaMeR, WiLoR, HaWoR, Dyn-HaMR), temporal
  filters (One Euro, Kalman / UKF), MANO fitting to 2D keypoints, egocentric hand detection.
- `lt submit "what you changed and why"`: sends ./tree to the gate.
  1. A dev pre-check (not charged): it must beat your incumbent on dev.
  2. The hidden val split (charged): it is accepted if the HOT3D S is significantly > 0 and >= 0.3 %, and on neither
     dataset any column is significantly more than 2 % worse (or more than 5 % worse at all).

  You get the verdict and a coarse S. Budget: {budget} charged submissions.
- `lt status`: budget and your history.

## How to work

1. `lt bucket`, then `lt try` on the empty tree to see where the error is.
2. Pick the largest error source you can explain. Write ONE focused module. Check it on dev. Keep notes in notes.md.
3. Submit a module only when it is clearly better on dev. After an acceptance, your tree becomes your new parent
   (incumbent); stack the next module on top.
4. Prefer mechanisms that are physically motivated and general. Examples: temporal consistency inside the segment;
   fixing the depth/translation of the first frames of a segment; existence calibration for half-visible hands;
   2D-3D consistency between the network's 2D anchors and its MANO projection; image-based refinement.

Your focus this round: {focus}
