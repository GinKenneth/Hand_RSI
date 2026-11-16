# Result files behind the paper's tables

`../test_scores.json` (every stage of every round and the C1 / C2 controls) and `../baselines.json` (published
methods) hold the main results; `../../baselines/classical/results.json` holds the classical post-processing controls.
The files here hold the rest. All test numbers come from the release evaluator (`ace_eval`, unannotated hand slots not
scored) unless the file says otherwise.

| file | paper |
|---|---|
| `tiers.json` | tier search and selection (Sec. 3.5, Fig. 7, Tab. 1 tier rows, supplement tier tables): every timed sub-chain of H5* with its val Δ and end-to-end FPS, the ε-front, the selected Lite / Medium / Heavy chains and the sealed-test scores of every front point |
| `distill_ablation.json` | Tab. 4 (what to distill, round 1, HOT3D dev) |
| `loop_stats.json` | Loop A per round (supplement Loop A table: submissions, acceptances, output tokens, turns, per model) and the module families of the final harness (Sec. 4.3) |
| `pck_curves.json` | Fig. 3 (top), 3D PCK |
| `strict_hot3d.json` | supplement, HOT3D under the strict detection rule |
| `analysis_hot3d_test.json` | supplement, error in view / out of sight and by image radius |
| `probes.json` | supplement, input-side probes (F_cs and the per-stage probe runs) |

The tier chains themselves are in `rsi/harness/trees/{Lite,Medium,Heavy}` (sub-chains of `H5`, same module files).
Replaying a tier reproduces its row, e.g. on HOT3D test:

```
cd rsi
python -m harness.run --tree harness/trees/Lite --cache captures/theta2/hot3d_test --view <views>/hot3d_test --out preds/lite
python -m eval.score --gt <views>/hot3d_test/gt --pred preds/lite        # MPJPE-p 9.06, Jitter 1.47
```
