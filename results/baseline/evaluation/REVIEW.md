# Baseline review

Frozen quantum checkpoint selected at step 83 by validation loss.

| Model | Output | Test MSE | Test MAE | Maximum test error |
|---|---|---:|---:|---:|
| qcl | sin | 0.22574955 | 0.40398791 | 0.84421617 |
| qcl | cos | 0.3109315 | 0.49938503 | 0.77490259 |
| polynomial | sin | 5.6693237e-05 | 0.0045398716 | 0.024253898 |
| polynomial | cos | 0.00064162411 | 0.018105749 | 0.076465058 |
| analytic_reference | sin | 3.4883406e-32 | 1.0697461e-16 | 5.8286709e-16 |
| analytic_reference | cos | 8.4821165e-33 | 7.1123663e-17 | 2.220446e-16 |

The polynomial has 12 fitted coefficients; QCL has 6 nominal trainable angles. The analytic reference directly encodes the target rotations and has zero fitted parameters. It is not a learned baseline. All three use local exact simulation or classical computation; no QPU.

The one-layer circuit has restricted expressivity. Final Z rotations on both wires commute with the measured Z observables (including through CNOT), so two of its six angles do not change these readouts. This baseline is preserved as declared; Depth study will study data re-uploading and deeper circuits using validation for selection. No architecture was changed based on these test scores.

Only one initialization seed and one dataset were evaluated here. Results establish neither quantum advantage nor reliable performance across initializations. The polynomial degree was fixed before evaluation. Test scores are now exposed for reporting and must not guide later selection.

Inspect sin_fit.png, cos_fit.png, metrics.csv, and the training loss plot in ../baseline/loss.png. The predictions archive contains all targets and predictions so metrics can be checked independently.
