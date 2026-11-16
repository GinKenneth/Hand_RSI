# Classical post-processing controls

Two controls for the question "could standard post-processing, tuned on val, do what the evolved harness does?"
(technical report, Tab. 13). They run through the same harness runtime (`rsi/harness/run.py`), on the same
captures and with the same evaluator as HandRSI, and they are selected the way the loop selects: on HOT3D val with the
gate's composite gain Δ (ARCTIC val as a regression check), then one run per selected config on the sealed tests.

| family | modules | swept on val (configs) |
|---|---|---|
| smoothers | `cl_gauss` (zero-phase Gaussian, σ), `cl_savgol` (Savitzky-Golay, window / order), `cl_oneeuro` (One-Euro, min cutoff / β) on MANO rotations (re-projected to SO(3)), translation and 2D keypoints | 30 |
| 2D alignment | `cl_root2d`: translation re-fitted by least squares to the network's own 2D keypoints (full or x-y only), optionally + Savitzky-Golay | 7 |
| naive WiLoR ensemble | `cl_wilor`: finger articulation slerped towards WiLoR's with one global weight α (demo crop, no outlier gate, no smoothing of WiLoR), optionally + Savitzky-Golay w7p2 | 8 |

`configs.json` lists every chain; `results.json` holds every score (val: Δ with CI90 and per-metric gains against θ2
alone; test: the 10 columns).

## Val selection (HOT3D val, θ2, Δ against θ2 with the empty harness)

| family | val-selected config | Δ | CI90 | passes the harness gate (Δ ≥ 0.3 %, CI90 > 0, no blocking regression) |
|---|---|---|---|---|
| smoothers | `savgol_w7p2` | +0.90 % | [+0.74, +1.06] | yes |
| 2D alignment | `root2dxy_a1.0+savgol_w7p2` | +2.03 % | [+1.21, +2.10] | no: FAcc regresses (HOT3D test 0.982 → 0.862) |
| naive WiLoR ensemble | `ens_a0.25+savgol_w7p2` | +2.49 % | [+1.31, +3.41] | yes |

For reference, the tiers on the same val split: Lite +3.90 %, Medium +4.06 %, Heavy +4.93 %.

## Sealed tests (MPJPE-p mm / Jitter mm/frame²)

| | HOT3D | ARCTIC | HOI4D |
|---|---|---|---|
| θ2 alone | 9.37 / 2.51 | 15.15 / 2.36 | 24.11 / 1.87 |
| θ2 + Savitzky-Golay | 9.34 / 2.18 | 15.10 / 2.21 | 24.08 / 1.85 |
| θ2 + naive WiLoR ensemble | 9.32 / 2.21 | 15.00 / 2.23 | – |
| θ2 + 2D alignment + SG (fails the gate) | 8.68 / 2.14 | 14.76 / 2.15 | 24.04 / 1.76 |
| θ0 alone | 12.97 / 2.82 | 15.58 / 2.58 | 23.45 / 2.10 |
| θ0 + Savitzky-Golay | 12.94 / 2.44 | 15.52 / 2.40 | 23.43 / 2.00 |
| HandRSI-Heavy (θ2 + 29 modules) | 8.41 / 1.35 | 14.00 / 1.46 | – |
| HandRSI (θ2 + H5*, 39 modules) | 8.37 / 1.34 | 13.85 / 1.45 | 22.77 / 1.01 |

## Running it

```
# captures of θ2 (released weights) and θ0 on hot3d_val, arctic_val, hot3d_test, arctic_test, hoi4d_test:
#   rsi/capture/capture.sh <theta2|theta0> <ckpt dir | release> <view> <gpu>   -> $CAPTURES/<net>/<view>
CAPTURES=... VIEWS=... OUT=... bash baselines/classical/run.sh smoothers    # smoothers + 2D alignment, CPU only
CAPTURES=... VIEWS=... OUT=... WILOR_MINI_ROOT=... bash baselines/classical/run.sh ensemble   # needs one GPU
```

The WiLoR ensemble reuses the WiLoR wrapper of the final harness (`rsi/harness/trees/H5/modules/wilor.py`) with its
demo settings; WiLoR's output is cached per segment, so the α sweep runs WiLoR once per view.
