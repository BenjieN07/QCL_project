# Local export and IBM preflight

The model is still trained with an exact classical simulation of a two-qubit quantum circuit. Hardware preparation does not change or retrain its weights.

Install the optional tools with `.\.venv\Scripts\python.exe -m pip install -r requirements-hardware.txt`.

After freezing the selected model, run `.\.venv\Scripts\python.exe -m qcl.hardware`. This checks the exported gates against the original simulator, saves nine measured circuits to `results/hardware/circuits.qpy`, and benchmarks ideal measurement sampling at 256, 1024 and 4096 shots with 50 repeats each. It saves a graph and numerical results. Ideal shot noise excludes real device noise. The generic local compilation is not a real backend readiness check.

For account access, run `.\.venv\Scripts\python.exe ibm_preflight.py` in your own terminal. Enter a new API **secret value** and the instance CRN at the hidden prompts. An `ApiKey-...` identifier is not the secret value. Replace any usable secret shared in chat before using it. Do not paste credentials into chat, command arguments or project files. This script keeps credentials in process memory only, does not save an account, and never submits jobs.

The first run lists accessible backend names. After choosing one, run `.\.venv\Scripts\python.exe ibm_preflight.py --backend YOUR_BACKEND_NAME`. It checks read-only usage and compiles the frozen circuits for that target. Reports and target circuits stay in ignored `.ibm-private/`. The usage summary only retains numerical usage fields; if incomplete, confirm the remaining allowance in the IBM dashboard. Account access alone does not establish plan allowance or approve a job.

If you choose local credential files, place only the secret value in `api.txt` and only the CRN in `crn.txt`, in the project root. Both exact filenames are excluded from Git. Run `.\.venv\Scripts\python.exe ibm_preflight.py --credential-files` (optionally with `--backend NAME --pilot`). The script reads these files without displaying their contents; empty files stop before any connection. Hidden terminal entry remains available without this flag.

The proposed pilot is three inputs, -1, 0 and 1, at 1024 shots each: 3072 shots total. After reviewing those results, six more inputs could complete the nine-input comparison, for 9216 shots total. Both outputs come from one measurement basis. Backend, plan, remaining allowance, compiled layout and execution estimate must be reviewed before authorizing a specific submission. No submission tool is included here.

The three pilot circuits are separately packaged in `results/hardware/pilot.qpy`, with their expected outputs and checkpoint fingerprint in `pilot_manifest.json`. Rebuild and verify them with `.\.venv\Scripts\python.exe -m qcl.hardware --prepare-pilot`. For the smaller target-compilation preflight, use `.\.venv\Scripts\python.exe ibm_preflight.py --backend YOUR_BACKEND_NAME --pilot`. This verifies the pilot still matches the active checkpoint before compiling it; it does not submit or estimate an unverified quota.

Review the results in `results/mirror/evaluation/REPORT.md`, run `view_model.py` for the current exact prediction graph, and inspect `results/hardware/shot_benchmark.png` for the separate ideal-shot comparison. Nothing in these commands runs hardware training.

## Authorized three-input pilot

`python -m qcl.ibm_pilot --prepare` checks the frozen checkpoint, actual ibm_fez target, output mapping, current allowance and timing estimate. It writes a reviewable plan under `results/hardware/real_pilot/`. The compiled circuit is reduced to its two active physical wires for exact equivalence checking. Target timings and the backend repetition delay provide a baseline estimate with an allowance of two seconds overhead per circuit; actual usage can differ. See [IBM workload usage](https://quantum.cloud.ibm.com/docs/en/guides/estimate-job-run-time).

`--submit` submits exactly one job containing three circuits and 1,024 shots per circuit, with a 60-second maximum execution time, no session, no twirling and no dynamical decoupling. This is for the pilot the user explicitly authorized; it is not an instruction to run additional experiments. A submission marker and saved job receipt prevent accidental duplicate submissions. If submission is interrupted, investigate the existing workload before retrying. No automatic resubmission occurs.

`--retrieve` checks the existing job and, once done, saves its two-bit counts, measured outputs, deviation from exact QCL/NumPy and actual usage. It does not block waiting for the queue and does not submit anything. Credentials stay in the ignored local files. This experiment evaluates fixed trained angles on real hardware; it performs zero hardware training updates.

The first pilot completed as job `dbhs6vg1h92s7382cmng` on ibm_fez. See `results/hardware/real_pilot/REPORT.md` and `comparison.png`. Run `.\.venv\Scripts\python.exe -m qcl.hardware_report` to reconstruct the count-based audit and figure offline. IBM reported three seconds of job usage and 591 seconds of remaining allowance immediately afterward. This is a three-input pilot, not full-domain hardware validation. Review it before authorizing additional hardware inputs.
