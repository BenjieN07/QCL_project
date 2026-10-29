# Quantum Circuit Learning: sine and cosine

An incremental educational experiment. No quantum advantage claim. Training uses a local simulator; an authorized three-input IBM hardware pilot has completed with frozen weights.

## View the current model

```powershell
.\.venv\Scripts\python.exe view_model.py
```

This opens a local browser report showing exact NumPy sine/cosine targets as black dashed curves and the saved QCL approximation in blue. The lower panels show signed errors; the table reports the original 24-input test metrics. It loads our circuit's checkpoint registered in `results/current_model.json` (mirrored-coverage study: five layers, seed 41, selected update 4,288) and does not retrain. Outputs are refreshed in `results/model_view/`. The active model has two qubits, five CNOTs, 30 nominal angles, and direct Z readouts without fitted output scales. Its fixed input preprocessing is x=0.9*t before our existing inverse-trigonometric encoder; this is recorded in each checkpoint's configuration.

If the environment is activated, `python view_model.py` is enough. Add `--no-open` to save without opening a browser, or `--run results/path/to/run` to inspect another checkpoint directory containing `best.pt`. The figures use a smooth 401-point grid; the metrics use the fixed disjoint test set, not that plotting grid.

To inspect the historical Day 3 comparison of initialization with the original small trained model and its loss history:

```powershell
.\.venv\Scripts\python.exe view_progress.py
```

This opens `results/progress/progress.png` and saves its underlying curves and metrics. It reconstructs the seed-7 initialization and verifies its loss against recorded step zero. The historical plot distinguishes the 250 updates performed from the validation-selected step-83 checkpoint. At that Day 3 checkpoint, recorded experiment updates totaled 310 across the two 30-step runs and the 250-step baseline; small unit-test optimizations are excluded. No retraining occurs.

## Review checkpoints

### Mirrored training and hardware preparation

The real-qubit pilot completed on **ibm_fez**, physical qubits 143/144, with 1,024 shots at each of t=-1,0,1. IBM reported three seconds of usage. Read [the pilot report](results/hardware/real_pilot/REPORT.md) and inspect its comparison figure. Reconstruct its count-based audit offline with `.\.venv\Scripts\python.exe -m qcl.hardware_report`. This validates three inputs only and does not train the model on hardware.

The controlled continuation study is specified in `results/mirror/PROTOCOL.md`. After its checkpoint selection is frozen, `results/mirror/evaluation/REPORT.md` compares all branches on a new stratified holdout. `results/mirror/evaluation/comparison.png` shows before/after curves and signed errors; `training_comparison.png` shows the three training histories.

Run `.\review-mirror.ps1` to audit the saved continuation states, checkpoint selection, evaluation and offline shot results without retraining. Optional hardware dependencies are required for the export audit. See [HARDWARE.md](HARDWARE.md) for the hidden-entry read-only IBM preflight and the proposed three-input pilot. No QPU job is submitted by these tools.

### Investigate current error peaks without training

```powershell
.\.venv\Scripts\python.exe -m qcl.error_peaks
```

The [peak investigation](results/error_peaks/REPORT.md) locates signed-error maxima/minima, checks nearby training coverage, compares the update-1,500 parent with update 3,000, measures input slopes and local gradient conflicts, and checks sine/cosine symmetry. `error_peaks.png` labels the largest residuals and training inputs. `analysis.json` and `curves.npz` contain the measured evidence. This diagnostic uses no optimizer updates or checkpoint selection and verifies that weights, checkpoint bytes, and the active registry remain unchanged. Proposed next experiments in the report have not been run.

### Boundary investigation and improved fit

The [boundary-study results](results/boundary/evaluation/REVIEW.md) compare eight approaches under a predeclared validation guard, retaining failures as well as improvements. Only two original training inputs were in |t|>=0.9. Parameter gradients were finite, while the inverse-trigonometric encoder was highly input-sensitive near endpoints. More updates alone helped only modestly. Boundary examples initially traded boundary error against the interior fit; a five-layer version of our own circuit with smoother encoding improved both.

```powershell
.\review-boundary.ps1
.\.venv\Scripts\python.exe view_model.py
```

The selected model was trained fresh for 1,500 updates, then genuinely resumed with its saved Adam state to update 3,000 at learning rate 0.003. The final independent holdout contains 256 interior and 128 inputs per boundary collar, generated after selection commit 1a4a102. Interior mean absolute errors are 0.02362 (sine) and 0.02440 (cosine). The worst new near-boundary errors are 0.02394 and 0.03857, compared with 0.40202 and 0.52811 before. Exact endpoints are now supervised anchors, so their errors are not independent test evidence. The aggregate holdout intentionally oversamples boundaries; regional scores are reported separately.

Endpoint and collar-validation goals passed; the stricter original-validation per-output MSE goal remains unmet for cosine. These are single-seed follow-up results, not a guarantee for every input. Inspect `results/boundary/evaluation/comparison.png`, `boundary_zoom.png`, and `trials.csv`, plus each trial's `history.png`. Raw data, start/best/final weights, Adam states, schedules, histories, and source hashes are preserved. The runner's explicit `schedule` and per-row learning rates are authoritative; inherited legacy learning-rate metadata is unused.

```powershell
.\.venv\Scripts\python.exe -m qcl.boundary_diagnosis
.\.venv\Scripts\python.exe -m qcl.boundary_study
.\.venv\Scripts\python.exe -m qcl.boundary_study --select-only
.\.venv\Scripts\python.exe -m qcl.boundary_report --verify
```

Completed matching training runs are reused rather than retrained; those commands do not add further updates to the active model. Any new continuation needs a newly declared trial and budget. Initial post-selection evaluation uses `python -m qcl.boundary_report` and protects existing reports from overwriting. No new holdout should be opened before checkpoints/selection are frozen. No removed comparison model, output clipping, or analytic boundary patch is used.

### Day 4 results and verification

The [Day 4 protocol](results/day4/PROTOCOL.md) compares only our original circuit at 1/2/3 layers across seeds 7/19/41, with 250 full-batch Adam updates per run. All nine configurations, checkpoints, optimizer states, logs, histories, versions, and loss plots are saved. The comparison model has been removed from the project files.

```powershell
.\review-day4.ps1
.\.venv\Scripts\python.exe view_model.py
```

The review command tests the code and audits saved results without training. Inspect [the results](results/day4/evaluation/REVIEW.md), `comparison.png`, and `runs.csv`. JSON includes per-output metrics, three-seed means/sample standard deviations, logical gate counts, gradients, loop runtimes, and the classical polynomial comparison. `predictions.npz` retains all evaluated predictions.

Three layers improved validation results across all three tested seeds, but the provisional accuracy target remains unmet. The selected model's new 256-input holdout MAE is 0.1460 for sine and 0.1323 for cosine. This holdout was generated after selection commit 4a23755 and is now reporting-only data. Further accuracy tuning needs a separate protocol. The selected run was still improving at step 250, so a controlled budget/learning-rate comparison is the next accuracy investigation before finite-shot work.

To reproduce or resume the fixed training grid and freeze selection:

```powershell
.\.venv\Scripts\python.exe -m qcl.study
.\.venv\Scripts\python.exe -m qcl.study --select-only
```

Complete matching runs are reused. `python -m qcl.depth_report --verify` audits the saved evaluation. Initial evaluation uses the separate `python -m qcl.depth_report` command; existing reports are protected from overwriting. Batching was verified against scalar circuits at all study depths, and the depth-1/seed-7 trajectory reproduces Day 3 losses within 1.33e-15. CPU timing and logical gate counts do not describe compiled QPU cost.

### Original roadmap

We stop after each day's exit gate. Day 1 establishes data and circuit validity; it does not train a model. See [the original roadmap](QCL_Implementation_Plan.md) and [current progress](PROGRESS.md).

## Setup (PowerShell, Python 3.11+)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Exact installed versions are saved with benchmark results. Configuration and split seeds are tracked in `config.json`.

## Run and review Day 1

```powershell
.\.venv\Scripts\python.exe -m qcl.prepare
.\.venv\Scripts\python.exe -m qcl.benchmark
```

Open `results/day1/REVIEW.md` for the review checklist, figures, and encoding description. `data_checks.json` and `benchmark.json` contain machine-readable pass/fail evidence, seeds, versions, gate counts, parameter sensitivity, and a timing measurement. A failed assertion exits with an error. Timings describe only this machine and batch, not quantum speedup.

For a single review command, run `.\review-day1.ps1`. It rebuilds the figures and checks, then prints the review note. For exact environment recreation use `pip install -r requirements-lock.txt` after creating the virtual environment.

The fixed splits are 48 training, 16 validation, and 24 test inputs. Validation is reserved for architecture selection; test labels are not used by the Day 1 circuit benchmark. The plotting grid has 101 points.

## Run and review Day 2

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m qcl.train --output results/day2/my-run
```

The smoke test runs 30 full-batch Adam updates with learning rate 0.05 and parameter seed 7. It saves `history.csv`, `loss.png`, `summary.json`, a copy of the configuration, and `best.pt` selected by validation loss. Existing completed run directories are protected from accidental overwrites: choose a fresh output path to repeat a run. Training must improve by at least 1% to pass this smoke gate; this is not a target accuracy claim. Loss is the batch mean of the sum of squared errors across sine and cosine.

The runner checks finite, nonzero gradient norm at every update and verifies checkpoint reload. Individual angle gradients may be zero because final Z rotations do not affect Z readouts; a nonzero total gradient does not imply all six angles are identifiable. Test labels are reserved for Day 3. Model checkpoints use PyTorch state dictionaries and can be loaded with `torch.load(path, weights_only=True)`.

Run `.\review-day2.ps1` to run the tests, independently audit the saved smoke checkpoint, and compare both saved runs. Add `-Repeat` to train a fresh run and require identical loss/gradient history and checkpoint weights. The audit regenerates `audit.json` and `REVIEW.md`, reporting whether a repeat was supplied. Run directories retain the original configuration and environment versions; timing differences between repeats are expected. The saved repeat's wall-clock duration includes a long execution interruption and must not be used as a compute-speed benchmark.

## Scientific context

Inspired by [Mitarai et al., Quantum Circuit Learning (2018)](https://arxiv.org/abs/1803.00745), rather than an exact reproduction. Implementation follows the [PennyLane PyTorch interface](https://docs.pennylane.ai/en/stable/introduction/interfaces/torch.html).

## Run and review Day 3

The fixed [Day 3 protocol](results/day3/PROTOCOL.md) specifies a one-layer, 250-update baseline. Checkpoints use validation loss; the test set is used only for reporting. Compare with a degree-five polynomial fitted on training data and a quantum direct-rotation analytic reference with no training.

To inspect saved results without retraining:

```powershell
.\review-day3.ps1
```

To reproduce in fresh directories:

```powershell
.\.venv\Scripts\python.exe -m qcl.train --config configs/day3.json --output results/day3/my-baseline
.\.venv\Scripts\python.exe -m qcl.evaluate --run results/day3/my-baseline --output results/day3/my-evaluation
.\.venv\Scripts\python.exe -m qcl.verify_evaluation --run results/day3/my-baseline --evaluation results/day3/my-evaluation
```

Inspect `results/day3/evaluation/REVIEW.md`, separate sine/cosine fit plots, and `metrics.csv` for per-output MSE, MAE, and maximum error on train/validation/test. `predictions.npz` retains all targets and predictions; the auditor independently reloads the checkpoint, reconstructs predictions, verifies all reported metric rows, and checks validation-based selection. No accuracy threshold or advantage is assumed: a poorly fitting circuit is a result to report. Day 4 comparisons must continue selecting by validation even though this baseline's test scores have now been reported.
