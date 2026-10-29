# Quantum Circuit Learning: sine and cosine

An incremental educational experiment. No quantum advantage claim. Local simulator only; IBM hardware requires separate explicit consent.

## View the current model

```powershell
.\.venv\Scripts\python.exe view_model.py
```

This opens a local browser report showing exact NumPy sine/cosine targets as black dashed curves and the saved QCL approximation in blue. The lower panels show signed errors; the table reports the original 24-input test metrics. It loads our original circuit's checkpoint registered in `results/current_model.json` (Day 3, Adam, selected step 83) and does not retrain. Outputs are refreshed in `results/model_view/`. The active model has two qubits, one layer, one CNOT, six nominal angles, and direct Z readouts without fitted output scales.

If the environment is activated, `python view_model.py` is enough. Add `--no-open` to save without opening a browser, or `--run results/path/to/run` to inspect another checkpoint directory containing `best.pt`. The figures use a smooth 401-point grid; the metrics use the fixed disjoint test set, not that plotting grid.

To inspect the historical Day 3 comparison of initialization with the original small trained model and its loss history:

```powershell
.\.venv\Scripts\python.exe view_progress.py
```

This opens `results/progress/progress.png` and saves its underlying curves and metrics. It reconstructs the seed-7 initialization and verifies its loss against recorded step zero. The historical plot distinguishes the 250 updates performed from the validation-selected step-83 checkpoint. At that Day 3 checkpoint, recorded experiment updates totaled 310 across the two 30-step runs and the 250-step baseline; small unit-test optimizations are excluded. No retraining occurs.

## Review checkpoints

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
