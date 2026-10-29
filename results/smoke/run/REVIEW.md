# Training smoke test review

Training smoke test and checkpoint audit: PASS.

- Adam updates: 30; learning rate: 0.05; parameter seed: 7.
- Initial training loss: 1.94994098.
- Final training loss: 0.58133424.
- Training improvement: 70.19%.
- Selected step: 26; validation loss: 0.67837717.
- Minimum gradient norm: 0.13706088.
- Independent repeat identical: True.

Loss is the sample mean of the SUM of squared errors across outputs. These losses are not per-output MSE or held-out test scores.

Inspect loss.png and history.csv. The selected checkpoint is best.pt; summary.json records seeds, settings, versions, source revision and hashes. audit.json verifies the history, validation selection, and reloaded checkpoint.

This short run establishes gradient flow and loss reduction, not an accurate sine/cosine fit. No test labels or QPU jobs were used. Stop here for inspection before Baseline.
