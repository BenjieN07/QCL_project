# Progress and review gates

| Phase | Status | Evidence |
|---|---|---|
| Day 1: setup, data, circuit | COMPLETE | results/day1/REVIEW.md and benchmark.json |
| Day 2: training loop | NOT STARTED | — |
| Day 3: fit and baselines | NOT STARTED | — |
| Day 4: depth and seed study | NOT STARTED | — |
| Day 5: shot study | NOT STARTED | — |
| Day 6: optional hardware | NOT STARTED | Requires explicit consent |
| Day 7: report | NOT STARTED | — |

## 2026-10-28 — checkpoint 1

Created the project specification, pinned PennyLane version, configuration, and review workflow. The repository began empty. System `python` is an unusable Windows alias; a project virtual environment is being created using the bundled Python runtime. No numerical results yet. QPU usage: zero.

Next: deterministic disjoint datasets and target figure; then circuit validity benchmark. Stop after Day 1.

## 2026-10-28 — checkpoint 2

Dataset checks passed: seed 2026 reproduces 48 training, 16 validation, and 24 test inputs, all distinct across splits. Target identity sin² + cos² passed. Saved `results/day1/data.npz`, `data_checks.json`, and `targets.png`. Installed dependencies in the project environment and saved exact versions to `requirements-lock.txt`; `pip check` found no broken requirements. Early attempts to run preparation before installation finished failed due to missing Matplotlib; the completed run passed.

GitHub push failed with invalid username/token. Commits remain local until authentication is repaired. No hardware access or training performed.

## 2026-10-28 — checkpoint 3 / Day 1 exit gate

The later GitHub push succeeded and published checkpoints 1 and 2. The earlier authentication failure is resolved for this session.

Circuit benchmark passed all seven checks on 51 inputs (48 training plus endpoints and zero). Outputs have shape (51, 2), are finite, reproducible, and within [-1, 1]; observed range was -0.0785247186 to 0.9969115004. Perturbing each wire's trainable Y rotation changed its measured output; maximum changes were about 0.19647 on Z0 and 0.08075 on Z1. Six trainable parameters, seven logical gates (Rot counted as one), one logical CNOT. No backend compilation claimed. Circuit diagram and encoding note saved; target and circuit figures visually inspected.

Evidence: `results/day1/benchmark.json`, `circuit.png`, `circuit.txt`, `untrained_predictions.npz`, and `REVIEW.md`. Dependency consistency and Python syntax checks passed. Benchmark timing is a single batch with first-call overhead, not a comparative speed claim. No optimizer steps, held-out accuracy claims, or QPU jobs.

Review gate: stop here for user inspection. Next authorized phase, once requested: Day 2 differentiable training smoke test, gradients, saved checkpoint, and loss history. Keep future phases in separate commits and stop at major review gates.

## 2026-10-28 — documentation cleanup

Renamed the roadmap to `QCL_Implementation_Plan.md`, updated the README link, and removed assistant branding from the project documentation. Source, measured results, and scientific requirements are unchanged. Verified the documentation references and whitespace; numerical checks were not rerun for this documentation-only change.
