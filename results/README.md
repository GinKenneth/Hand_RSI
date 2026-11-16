# Results

Sealed test splits, scored with `ace_eval` (the 10 columns of ACE-Ego-Hand's Table 1). The loop never read the test
splits: every gate, checkpoint pick and pruning step used the dev / val splits of HOT3D and ARCTIC. Units: MPJPE-p, PA-p,
MPJPE+OOS in mm; EPE2D-p in px; GO-p in degrees; CT-p in m; Jitter in mm/frame². Machine-readable: `test_scores.json`
(HandRSI) and `baselines.json` (published methods, `baselines/README.md`).

## Round by round

Each round: Loop A (8 coding agents, 4 h) writes harness modules, gated on val; the accepted ones are merged into H_k.
Loop B fine-tunes the network (LoRA) on the ground truth of the training splits and on the teacher's outputs where the
training video has no ground truth, giving θ_k; H_k is re-checked module by module on θ_k. The round is accepted only
if N_k = (θ_k, H_k) beats N_(k-1) on val (composite score S on HOT3D ≥ 1 %, its CI90 lower bound > 0, no blocking
regression); otherwise H_k* is kept on θ_(k-1) if it passes alone (S ≥ 0.3 %, lower bound > 0).

| round | network θ_k | harness H_k | submissions (accepted) | val: S of N_k vs N_(k-1) (CI90 lower bound) |
|---|---|---|---|---|
| R1 | θ1: student and teacher built on C1₁ = θ0 + 500 GT steps; 500 steps | 18 modules | 70 (43) | +3.37 % (+2.46 %)¹ |
| R2 | θ2: student and teacher built on C1₂ = θ1 + 2000 GT steps; step 1500 of 2000, picked on val | 24 modules | 36 (6) | +3.54 % (+2.86 %) |
| R3 | θ3 = θ2: none of R3's Loop B checkpoints was ahead of θ2 on val | 32 modules | 26 (12) | +3.17 % (+2.74 %) |
| R4 | θ4 = θ2: none of R4's Loop B checkpoints was ahead of θ2 on val | 37 modules | 17 (9) | +2.06 % (+1.84 %) |
| R5 | θ5 = θ2: none of R5's Loop B checkpoints was ahead of θ2 on val | 39 modules (H5*) | 7 (3) | +0.77 % (+0.47 %); H5* kept by the harness-only rule |

¹ R1 was accepted by the authors (see "Manual decisions").

Every round with its two controls: C1_k fine-tunes the round's starting network on the ground truth alone (no teacher,
no harness), C2_k runs the round's harness on the released network θ0 (harness without distillation).

**HOT3D (test, 27 recordings, 1162 segments)**

| | FAcc ↑ | Recall ↑ | F1 ↑ | MPJPE-p ↓ | PA-p ↓ | MPJPE+OOS ↓ | EPE2D-p ↓ | GO-p ↓ | CT-p ↓ | Jitter ↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| start: ACE-Ego-Hand (θ0), empty harness | 0.970 | 0.986 | 0.990 | 12.97 | 5.90 | 12.84 | 9.66 | 6.90 | 0.024 | 2.82 |
| R1 · C1₁ = θ0 + GT only (control) | 0.982 | 0.994 | 0.994 | 9.91 | 5.00 | 10.76 | 5.56 | 5.71 | 0.014 | 2.77 |
| R1 · F₁ = θ1 alone | 0.981 | 0.994 | 0.994 | 9.89 | 5.00 | 10.75 | 5.77 | 5.80 | 0.014 | 2.63 |
| R1 · N₁ = θ1 + H1 (18 modules) | 0.977 | 0.998 | 0.993 | 9.39 | 4.83 | 10.78 | 5.38 | 5.20 | 0.012 | 1.65 |
| R1 · C2₁ = θ0 + H1 (control) | 0.976 | 0.993 | 0.992 | 11.13 | 5.09 | 11.73 | 7.96 | 5.92 | 0.021 | 1.75 |
| R2 · C1₂ = θ1 + GT only (control) | 0.983 | 0.994 | 0.994 | 9.39 | 4.76 | 10.33 | 5.32 | 5.71 | 0.013 | 2.73 |
| R2 · F₂ = θ2 alone | 0.982 | 0.994 | 0.994 | 9.37 | 4.71 | 10.35 | 5.31 | 5.80 | 0.013 | 2.51 |
| R2 · N₂ = θ2 + H2 (24 modules) | 0.978 | 0.997 | 0.993 | 8.65 | 4.51 | 10.07 | 4.53 | 5.14 | 0.012 | 1.52 |
| R2 · C2₂ = θ0 + H2 (control) | 0.974 | 0.990 | 0.992 | 10.81 | 5.22 | 11.12 | 7.90 | 6.07 | 0.022 | 1.65 |
| R3 · C1₃ = θ2 + GT only (control) | 0.983 | 0.994 | 0.994 | 9.21 | 4.62 | 10.27 | 5.22 | 5.74 | 0.013 | 2.66 |
| R3 · F₃ = θ3 = θ2 alone | 0.982 | 0.994 | 0.994 | 9.37 | 4.71 | 10.35 | 5.31 | 5.80 | 0.013 | 2.51 |
| R3 · N₃ = θ2 + H3 (32 modules) | 0.980 | 0.996 | 0.994 | 8.47 | 4.45 | 9.83 | 4.21 | 5.16 | 0.012 | 1.44 |
| R3 · C2₃ = θ0 + H3 (control) | 0.971 | 0.986 | 0.991 | 10.90 | 5.37 | 10.87 | 7.97 | 6.55 | 0.024 | 1.57 |
| R4 · C1₄ = θ2 + GT only (control) | 0.983 | 0.994 | 0.994 | 9.19 | 4.62 | 10.25 | 5.21 | 5.74 | 0.013 | 2.67 |
| R4 · F₄ = θ4 = θ2 alone | 0.982 | 0.994 | 0.994 | 9.37 | 4.71 | 10.35 | 5.31 | 5.80 | 0.013 | 2.51 |
| R4 · N₄ = θ2 + H4 (37 modules) | 0.980 | 0.997 | 0.994 | 8.36 | 4.39 | 9.72 | 4.06 | 5.13 | 0.012 | 1.37 |
| R4 · C2₄ = θ0 + H4 (control) | 0.971 | 0.986 | 0.991 | 10.36 | 5.22 | 10.35 | 7.55 | 6.53 | 0.024 | 1.48 |
| R5 · C1₅ = θ2 + GT only (control) | 0.983 | 0.994 | 0.994 | 9.23 | 4.64 | 10.23 | 5.24 | 5.75 | 0.013 | 2.66 |
| R5 · F₅ = θ5 = θ2 alone | 0.982 | 0.994 | 0.994 | 9.37 | 4.71 | 10.35 | 5.31 | 5.80 | 0.013 | 2.51 |
| R5 · N₅ = θ2 + H5* (39 modules) | 0.982 | 0.996 | 0.994 | 8.37 | 4.37 | 9.70 | 4.07 | 5.16 | 0.013 | 1.34 |
| R5 · C2₅ = θ0 + H5* (control) | 0.972 | 0.986 | 0.991 | 10.38 | 5.21 | 10.34 | 7.56 | 6.56 | 0.024 | 1.44 |

**ARCTIC (test = subject s05, 291 segments)**

| | FAcc ↑ | Recall ↑ | F1 ↑ | MPJPE-p ↓ | PA-p ↓ | MPJPE+OOS ↓ | EPE2D-p ↓ | GO-p ↓ | CT-p ↓ | Jitter ↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| start: ACE-Ego-Hand (θ0), empty harness | 0.986 | 0.995 | 0.996 | 15.58 | 7.43 | 16.54 | 9.36 | 12.45 | 0.024 | 2.58 |
| R1 · C1₁ = θ0 + GT only (control) | 0.988 | 0.996 | 0.997 | 15.51 | 7.36 | 16.49 | 9.28 | 12.40 | 0.023 | 2.57 |
| R1 · F₁ = θ1 alone | 0.987 | 0.996 | 0.996 | 15.40 | 7.36 | 16.38 | 9.20 | 12.50 | 0.022 | 2.44 |
| R1 · N₁ = θ1 + H1 (18 modules) | 0.987 | 1.000 | 0.997 | 14.84 | 6.69 | 16.18 | 8.31 | 11.92 | 0.019 | 1.72 |
| R1 · C2₁ = θ0 + H1 (control) | 0.989 | 1.000 | 0.997 | 14.95 | 6.67 | 16.29 | 8.26 | 11.84 | 0.020 | 1.79 |
| R2 · C1₂ = θ1 + GT only (control) | 0.988 | 0.996 | 0.997 | 15.26 | 7.32 | 16.25 | 9.10 | 12.43 | 0.022 | 2.51 |
| R2 · F₂ = θ2 alone | 0.987 | 0.995 | 0.997 | 15.15 | 7.28 | 16.11 | 9.12 | 12.34 | 0.021 | 2.36 |
| R2 · N₂ = θ2 + H2 (24 modules) | 0.989 | 0.999 | 0.997 | 13.97 | 6.59 | 15.26 | 7.89 | 11.65 | 0.018 | 1.60 |
| R2 · C2₂ = θ0 + H2 (control) | 0.991 | 0.999 | 0.998 | 14.35 | 6.61 | 15.62 | 8.02 | 11.79 | 0.021 | 1.69 |
| R3 · C1₃ = θ2 + GT only (control) | 0.989 | 0.996 | 0.997 | 15.23 | 7.27 | 16.26 | 9.00 | 12.38 | 0.021 | 2.48 |
| R3 · F₃ = θ3 = θ2 alone | 0.987 | 0.995 | 0.997 | 15.15 | 7.28 | 16.11 | 9.12 | 12.34 | 0.021 | 2.36 |
| R3 · N₃ = θ2 + H3 (32 modules) | 0.990 | 1.000 | 0.997 | 13.91 | 6.48 | 15.20 | 7.72 | 11.59 | 0.019 | 1.53 |
| R3 · C2₃ = θ0 + H3 (control) | 0.991 | 0.999 | 0.998 | 14.26 | 6.50 | 15.54 | 7.84 | 11.76 | 0.021 | 1.61 |
| R4 · C1₄ = θ2 + GT only (control) | 0.989 | 0.996 | 0.997 | 15.17 | 7.26 | 16.18 | 9.01 | 12.42 | 0.021 | 2.47 |
| R4 · F₄ = θ4 = θ2 alone | 0.987 | 0.995 | 0.997 | 15.15 | 7.28 | 16.11 | 9.12 | 12.34 | 0.021 | 2.36 |
| R4 · N₄ = θ2 + H4 (37 modules) | 0.989 | 1.000 | 0.997 | 13.84 | 6.37 | 15.17 | 7.61 | 11.61 | 0.019 | 1.47 |
| R4 · C2₄ = θ0 + H4 (control) | 0.991 | 0.999 | 0.998 | 14.19 | 6.37 | 15.50 | 7.66 | 11.77 | 0.021 | 1.55 |
| R5 · C1₅ = θ2 + GT only (control) | 0.988 | 0.996 | 0.997 | 15.25 | 7.27 | 16.23 | 9.05 | 12.40 | 0.021 | 2.48 |
| R5 · F₅ = θ5 = θ2 alone | 0.987 | 0.995 | 0.997 | 15.15 | 7.28 | 16.11 | 9.12 | 12.34 | 0.021 | 2.36 |
| R5 · N₅ = θ2 + H5* (39 modules) | 0.990 | 1.000 | 0.997 | 13.85 | 6.34 | 15.18 | 7.61 | 11.61 | 0.019 | 1.45 |
| R5 · C2₅ = θ0 + H5* (control) | 0.992 | 0.999 | 0.998 | 14.20 | 6.35 | 15.50 | 7.65 | 11.77 | 0.021 | 1.52 |

**HOI4D (zero-shot: never used for training, gating or selection; 498 segments)**

| | FAcc ↑ | Recall ↑ | F1 ↑ | MPJPE-p ↓ | PA-p ↓ | MPJPE+OOS ↓ | EPE2D-p ↓ | GO-p ↓ | CT-p ↓ | Jitter ↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| start: ACE-Ego-Hand (θ0), empty harness | 0.994 | 0.994 | 0.994 | 23.45 | 11.02 | 22.93 | 15.60 | 19.39 | 0.058 | 2.10 |
| R1 · C1₁ = θ0 + GT only (control) | 0.994 | 0.994 | 0.994 | 23.83 | 11.01 | 23.29 | 15.71 | 19.35 | 0.062 | 2.07 |
| R1 · F₁ = θ1 alone | 0.993 | 0.994 | 0.994 | 23.71 | 11.08 | 23.13 | 16.18 | 19.43 | 0.062 | 1.98 |
| R1 · N₁ = θ1 + H1 (18 modules) | 0.990 | 0.990 | 0.990 | 23.51 | 10.36 | 22.54 | 20.02 | 19.98 | 0.064 | 1.26 |
| R1 · C2₁ = θ0 + H1 (control) | 0.991 | 0.991 | 0.992 | 23.22 | 10.25 | 22.39 | 18.28 | 19.80 | 0.060 | 1.31 |
| R2 · C1₂ = θ1 + GT only (control) | 0.993 | 0.993 | 0.994 | 23.90 | 11.12 | 23.29 | 16.48 | 19.44 | 0.062 | 2.02 |
| R2 · F₂ = θ2 alone | 0.992 | 0.992 | 0.993 | 24.11 | 11.13 | 23.42 | 17.39 | 19.57 | 0.064 | 1.87 |
| R2 · N₂ = θ2 + H2 (24 modules) | 0.990 | 0.990 | 0.991 | 23.19 | 10.43 | 22.27 | 18.87 | 19.65 | 0.065 | 1.14 |
| R2 · C2₂ = θ0 + H2 (control) | 0.992 | 0.992 | 0.993 | 22.57 | 10.23 | 21.85 | 16.64 | 19.41 | 0.059 | 1.20 |
| R3 · C1₃ = θ2 + GT only (control) | 0.993 | 0.994 | 0.994 | 23.94 | 11.20 | 23.37 | 16.11 | 19.43 | 0.063 | 1.97 |
| R3 · F₃ = θ3 = θ2 alone | 0.992 | 0.992 | 0.993 | 24.11 | 11.13 | 23.42 | 17.39 | 19.57 | 0.064 | 1.87 |
| R3 · N₃ = θ2 + H3 (32 modules) | 0.991 | 0.991 | 0.992 | 23.07 | 10.34 | 22.23 | 18.60 | 19.44 | 0.064 | 1.06 |
| R3 · C2₃ = θ0 + H3 (control) | 0.993 | 0.993 | 0.994 | 22.44 | 10.14 | 21.80 | 16.19 | 19.21 | 0.058 | 1.11 |
| R4 · C1₄ = θ2 + GT only (control) | 0.993 | 0.993 | 0.993 | 24.07 | 11.19 | 23.40 | 17.08 | 19.45 | 0.063 | 1.97 |
| R4 · F₄ = θ4 = θ2 alone | 0.992 | 0.992 | 0.993 | 24.11 | 11.13 | 23.42 | 17.39 | 19.57 | 0.064 | 1.87 |
| R4 · N₄ = θ2 + H4 (37 modules) | 0.991 | 0.991 | 0.992 | 22.76 | 10.21 | 21.92 | 18.53 | 19.42 | 0.064 | 1.02 |
| R4 · C2₄ = θ0 + H4 (control) | 0.993 | 0.993 | 0.994 | 22.23 | 10.04 | 21.59 | 16.11 | 19.21 | 0.058 | 1.06 |
| R5 · C1₅ = θ2 + GT only (control) | 0.993 | 0.993 | 0.993 | 24.05 | 11.17 | 23.39 | 16.87 | 19.45 | 0.063 | 1.97 |
| R5 · F₅ = θ5 = θ2 alone | 0.992 | 0.992 | 0.993 | 24.11 | 11.13 | 23.42 | 17.39 | 19.57 | 0.064 | 1.87 |
| R5 · N₅ = θ2 + H5* (39 modules) | 0.991 | 0.991 | 0.992 | 22.77 | 10.21 | 21.93 | 18.61 | 19.44 | 0.064 | 1.01 |
| R5 · C2₅ = θ0 + H5* (control) | 0.993 | 0.993 | 0.994 | 22.24 | 10.03 | 21.59 | 16.13 | 19.22 | 0.058 | 1.05 |

## Manual decisions during the run

- R1: the authors chose the base Loop B recipe among variants (dev and test tables of the variants had been seen) and
  accepted the round.
- R2: one module was removed by hand after it degraded one recording; the worst-recording guard in
  `rsi/rounds/revalidate.py` now does this automatically.
- Two gate rules changed during the run: ARCTIC regressions stopped blocking (ARCTIC's train subjects were in
  ACE-Ego-Hand's training data), and predictions on unannotated HOT3D hands stopped counting as false positives.
- Between rounds the authors applied pipeline fixes proposed by Loop C that Loop C cannot apply itself.
