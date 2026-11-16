# Result files behind the technical report's tables

`../test_scores.json` (every stage of every round and the C1 / C2 controls) and `../baselines.json` (published
methods) hold the main results; `../../baselines/classical/results.json` holds the classical post-processing controls
(Tab. 13).
The files here hold the rest. All test numbers come from the release evaluator (`ace_eval`, unannotated hand slots not
scored) unless the file says otherwise. Table, figure and section numbers refer to the technical report.

| file | technical report |
|---|---|
| `tiers.json` | tier search and selection (Sec. 3.5 and 5.5, Fig. 9, Tab. 11, tier rows of Tab. 3): every timed sub-chain of H5* with its val Δ and end-to-end FPS, the ε-front, the selected Lite / Medium / Heavy chains and the sealed-test scores of every front point |
| `distill_ablation.json` | Tab. 9 (what to distill, round 1, HOT3D dev) |
| `loop_stats.json` | Loop A per round (Tab. 7: submissions, acceptances, output tokens, turns, per model) and the module families of the final harness (Sec. 5.3) |
| `pck_curves.json` | Fig. 4 (top), 3D PCK |
| `strict_hot3d.json` | Tab. 18, HOT3D under the strict detection rule |
| `analysis_hot3d_test.json` | additional analysis: error in view / out of sight and by image radius |
| `probes.json` | additional analysis: input-side probes (F_cs and the per-stage probe runs) |

The tier chains themselves are in `rsi/harness/trees/{Lite,Medium,Heavy}` (sub-chains of `H5`, same module files).
Replaying a tier reproduces its row, e.g. on HOT3D test:

```
cd rsi
python -m harness.run --tree harness/trees/Lite --cache captures/theta2/hot3d_test --view <views>/hot3d_test --out preds/lite
python -m eval.score --gt <views>/hot3d_test/gt --pred preds/lite        # MPJPE-p 9.06, Jitter 1.47
```
