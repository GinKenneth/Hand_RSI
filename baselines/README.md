# Baselines: published hand estimators under ACE-Ego-Hand's protocol

These scripts re-run the baselines of ACE-Ego-Hand's Table 1 on our test splits and score them with `ace_eval`, so that
every row of `results/baselines.json` comes from the same videos, the same intrinsics and the same evaluator. They follow
ACE-Ego-Hand's Appendix A (A.3 for the baselines). They are a **reproduction**: the numbers are not ACE-Ego-Hand's
(its splits are not ours, and its baseline PA-p values are quoted from ViDiHand's evaluation), and nothing here feeds
any decision of the RSI loop.

```
 test view (81-frame segments, GT intrinsics K)
   |
   +-- ACE-Ego-Hand K-given (θ0) .......... rsi/capture (official inference), empty harness
   +-- ACE-Ego-Hand K-free ................ same capture, K-free weights; K enters only the scoring     run_kfree.py
   +-- WiLoR, HaWoR, EgoForce ............. their own detectors                                         run_wilor.py, run_hawor.py, run_egoforce.py
   +-- HaMeR, Hamba, WildHands,
       OmniHands, InterWild ............... shared front end: Detectron2 ViTDet-H person boxes          frontend.py, then run_<method>.py
                                            + ViTPose+-Huge whole-body hand keypoints;
                                            each method derives its hand boxes with its own demo rule
   |
   v
 camera:  wherever a method maps a crop camera to the full image, it gets the clip's GT K (not the demo's guess)
 joints:  the method's own MANO parameters, re-posed with ace_eval's MANO -> ACE's 21 joints; checked against the
          method's own mesh in every run (the script stops on a mismatch)
 video methods (HaWoR, OmniHands, EgoForce) run once per 81-frame segment; single-frame methods per frame
   |
   v
 preds/<method>/<view>/<segment>.npz  ->  score.py  (ace_eval, default Config)
```

## Methods

| method | code (commit) | weights | detector | notes |
|---|---|---|---|---|
| ACE-Ego-Hand K-free | [ggxxii/ACE-Ego-Hand](https://github.com/ggxxii/ACE-Ego-Hand) `59da83b` | `ace_ego_hand_kfree.pt` | none (whole frame) | existence from `exists_2d` |
| WiLoR | [rolpotamias/WiLoR](https://github.com/rolpotamias/WiLoR) `fcb9113` | `wilor_final.ckpt` | its YOLO, conf 0.3 | most confident detection per side |
| HaWoR | [ThunderVVV/HaWoR](https://github.com/ThunderVVV/HaWoR) `66c7d41` | `hawor.ckpt` | WiLoR's YOLO with tracking, conf 0.2 | camera-space front end; SLAM and infilling off (as ACE) |
| EgoForce | [dfki-av/EgoForce](https://github.com/dfki-av/EgoForce) `480ffb3` | `model_weights.pth` | its RTMDet + YOLO-pose | causal Kalman filter on (ACE's ‡); no TensorRT (see below) |
| HaMeR | [geopavlakos/hamer](https://github.com/geopavlakos/hamer) `3a01849` | `hamer.ckpt` | front end, keypoint conf > 0.5 | |
| Hamba | [humansensinglab/Hamba](https://github.com/humansensinglab/Hamba) `3aa7077` | `hamba.ckpt` | front end, conf > 0.1, then MediaPipe | |
| WildHands | [ap229997/hands](https://github.com/ap229997/hands) `f99dfea` | `wildhands.ckpt` | front end, conf > 0.5 | GT K rescaled with the 840 px image |
| OmniHands | [LinDixuan/OmniHands](https://github.com/LinDixuan/OmniHands) `935e1f5` | `Demo_Video.pth` (9-frame video model) | front end, first person | PnP redone under GT K |
| InterWild | [facebookresearch/InterWild](https://github.com/facebookresearch/InterWild) `7c01e4a` | IH26M(H+M) + MSCOCO + ReInterHand(Ego_cameras) `snapshot_6.pth` | front end, BoxNet bypassed (as ACE) | |

The front end is Detectron2's ViTDet-H (`model_final_f05665.pkl`, as `hamer/demo.py`) and ViTPose+-Huge whole-body
(`wholebody.pth`, from HaMeR's demo data). InterWild's egocentric checkpoint is the one InterWild's README links;
sha256 of the file we used: `43c59d40a0afb04a875752c9b81d02673732fd452aaedb8d95d09d59a5a5f73c`.

Not run: Dyn-HaMR and ViDiHand (code not released).

Classical post-processing (Gaussian, Savitzky-Golay, One-Euro, 2D alignment) and a naive WiLoR ensemble on our own
network, selected on val like the loop's harness: [`classical/README.md`](classical/README.md).

## Changes from the official demos

Each script's docstring states its full pipeline.
- **Camera.** Wherever a method maps a crop camera to the full image, it uses the clip's ground-truth intrinsics
  instead of the demo's default focal (OmniHands: its PnP placement is redone under the same intrinsics).
- **Joints.** Each method's MANO parameters are re-posed with the evaluator's MANO and checked against the method's own
  mesh on every run.
- **Hamba.** One MediaPipe tracker, reset at every frame (equivalent to the demo's fresh tracker per image).
- **InterWild.** The front-end hand boxes replace its BoxNet output, as in ACE-Ego-Hand's protocol.
- **EgoForce.** Runs without TensorRT (fp16 autocast instead); its mmcv 2.x dependencies are installed apart by
  `build_egoforce_env.sh`.
- **ACE-Ego-Hand K-free.** The direct 2D head is dropped, so EPE2D re-projects the 3D joints (A.1).

## Layout and running

```
$BASELINES_ROOT  (default third_party/baselines)
  official/<Method>/              each repository at the commit above, set up as its README says
                                  (WiLoR: pretrained_models/ and mano_data/; HaWoR: its _DATA/)
  dependencies/ViTPose/           HaMeR's ViTPose checkout
  weights/vitdet/model_final_f05665.pkl
  weights/HaMeR/_DATA/            HaMeR's fetch_demo_data.sh (data/mano, hamer_ckpts, vitpose_ckpts); Hamba reads it too
  weights/Hamba/checkpoints/hamba.ckpt
  weights/WildHands/wildhands.ckpt
  weights/OmniHands/Demo_Video.pth
  weights/InterWild/ego/snapshot_6.pth
  weights/HaWoR/hawor/{checkpoints/hawor.ckpt, model_config.yaml}, weights/HaWoR/external/detector.pt
  weights/EgoForce/_DATA/{model_weights.pth, epoch_460.pth, detector.torchscript, mano/}
$BASELINES_VIEWS (default rsi/data/views)   the test views made by rsi/data/build_views.py
$BASELINES_OUT   (default baselines_out)    frontend/, preds/, scores/, work/ (written by the scripts)
```

```bash
cd baselines
python frontend.py --view hot3d_test                      # any number of workers per view (lock files)
python run_hamer.py --view hot3d_test                     # likewise run_hamba / run_wildhands / run_omnihands / run_interwild
python run_wilor.py --view hot3d_test                     # likewise run_hawor; run_egoforce after bash build_egoforce_env.sh
ACE_KFREE_CKPT=<dir> python run_kfree.py hot3d_test       # dir: ace_ego_hand_k.pt + ace_ego_hand_k.yml of the K-free release
python score.py hamer hot3d_test arctic_test hoi4d_test   # -> $BASELINES_OUT/scores/<method>_<view>.json
```

`frontend.py` and every `run_<method>.py` take `--shard i/n` and `--seqs a,b` and skip what they have already written.
The official checkpoints are pickles: with torch >= 2.6 set `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`.

## Results

The tables are in the top-level README (section "Results"); the numbers with counts are in `results/baselines.json`.
