# Real hardware pilot results

Job `dbhs6vg1h92s7382cmng` completed on **ibm_fez**. Physical qubits: [143, 144]. Three inputs, 1024 shots each, frozen model update 4288. No hardware training updates.

| t | Hardware sine | Exact QCL sine | Hardware cosine | Exact QCL cosine |
|---|---|---|---|---|
| -1 | -0.009766 | -0.013566 | -0.931641 | -0.950004 |
| 0 | -0.037109 | -0.016699 | 0.953125 | 0.945499 |
| 1 | 0.039062 | -0.012967 | -0.964844 | -0.986209 |

Provider-reported usage: `{'quantum_seconds': 3, 'seconds': 3, 'status': 'complete'}`.
Remaining allowance after the job: 591 seconds.

Errors relative to exact QCL combine finite-shot fluctuations and device effects; one run cannot separate these contributions. Error bars show sampling standard error only, not total device uncertainty. Three selected diagnostic inputs do not establish accuracy over the full domain. The other six proposed inputs were not submitted.

Matched ideal-shot reference: the previously saved 50 repeats at 1024 shots use the same frozen model and t=-1,0,1. Their means and standard deviations are included in audit.json. Gray bars show the theoretical one-standard-deviation ideal sampling range around zero deviation; colored bars show the hardware estimate and its sampling standard error.

Saved counts, receipt, compiled circuits, checkpoint fingerprints and reconstructed metrics are retained here. The audit checks stored results without contacting IBM or submitting another job.
