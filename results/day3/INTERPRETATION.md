# Day 3 findings

The training implementation works, but the one-layer quantum model is not yet a good sine/cosine approximator. Its 250-update training loss fell by 77.30%, then plateaued. The validation-selected checkpoint was step 83, not the final update. Training took 85.16 seconds inside the measured optimization loop; this is not a cross-model speed benchmark.

| Learned model | Sine test MSE | Cosine test MSE |
|---|---:|---:|
| One-layer QCL | 0.22574955 | 0.31093150 |
| Degree-five polynomial | 0.00005669 | 0.00064162 |

The classical baseline fits far better on this fixed split. The QCL curves miss the sine peak and compress the cosine amplitude. Errors are substantial on training, validation, and test inputs, which supports describing this run as underfitting rather than merely failing to generalize. This observation does not prove an architectural lower bound or rule out a better optimization solution.

The circuit has six nominal trainable angles, but two final Z rotations do not affect these Z readouts. That is a concrete limitation of this one-layer parameterization, not a barren-plateau diagnosis. The next planned experiment compares depths 1, 2, and 3 across initialization seeds, retaining the same encoding at every layer. Later layers may make intermediate rotations observable and add representational capacity. Their actual benefit remains unmeasured.

The analytic quantum reference reproduces the targets to floating-point precision by directly choosing known rotation angles. It does not learn anything and is included to show why these elementary functions do not require a trained quantum model.

All evaluation uses the original fixed splits. The polynomial degree and quantum training budget were declared before test evaluation. Checkpoint selection used validation only. The test scores are now known and must remain reporting data, not a selection criterion for later experiments.

Only one quantum initialization and dataset have been evaluated here. There is no hardware result or advantage claim. Review the figures and complete metrics in [evaluation/REVIEW.md](evaluation/REVIEW.md). Run `review-day3.ps1` from the repository root to verify the saved evidence without retraining.
