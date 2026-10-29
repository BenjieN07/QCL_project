# Hardware validation

The saved pilot evaluated the simulator-trained model on real qubits. It did not update its parameters.

| Item | Recorded value |
|---|---|
| Backend / physical qubits | ibm_fez / 143, 144 |
| Job | dbhs6vg1h92s7382cmng |
| Inputs / shots | -1, 0, 1 / 1,024 each |
| Model | Frozen update 4,288 |
| Actual job usage | 3 seconds |
| Remaining allowance immediately afterward | 591 seconds; recheck before further work |

[Results, measured counts and audit](results/hardware/real_pilot/REPORT.md). `python view_model.py` includes these measurements with the simulator and NumPy curves. `python -m qcl.hardware_report` reconstructs the saved hardware audit offline. Add `--show` for a separate native Python window.

## Tools

Install optional dependencies with `python -m pip install -r requirements-hardware.txt`.

- `python -m qcl.hardware --audit-only`: verify exported circuits and saved ideal-shot samples.
- `python ibm_preflight.py --credential-files --backend ibm_fez --pilot`: read-only access, allowance and compilation check.
- `python -m qcl.ibm_pilot --retrieve`: retrieve the existing job; never resubmits it.

Credentials stay in ignored local `api.txt` and `crn.txt`, or use hidden terminal entry by omitting `--credential-files`. Never commit secrets. Private account reports are in ignored `.ibm-private/`.

The completed pilot has a saved receipt and duplicate-submission guard. Do not resubmit it. Additional circuits or hardware-based training require a new scoped workload and explicit permission. The original estimate was about 6.8 seconds with a 60-second execution cap; actual usage was three seconds. [IBM usage estimation](https://quantum.cloud.ibm.com/docs/en/guides/estimate-job-run-time).
