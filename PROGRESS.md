# Progress and review gates

## Active direction — 2026-10-29

User restored our original circuit approach and excluded the paper-inspired model from the active program. `results/current_model.json` again points to the Day 3 Adam checkpoint (two qubits, one layer, selected step 83). The default viewer imports only the original model. The pilot remains archived as a historical comparison with its own frozen selection record; its stronger average accuracy is not assigned to our original model. No new training performed. Days 1–3 are complete as implementation/evaluation milestones; high-accuracy fitting is not complete. Day 4's matched 1/2/3-layer and initialization-seed study is next. Further optimizer/feature-map/entangler changes should be tested separately, with validation-based selection and honest reporting.

Restoration verified: all nine tests passed, the original Day 3 audit passed (1,134 reconstructed predictions and 18 metric rows), and the viewer reproduced the original checkpoint's test metrics exactly. The archived pilot's separate audit still passes without depending on the active model registry. Numerical experiment results were not altered.

## 2026-10-29 — paper-inspired accuracy pilot, checkpoint 1

User requested an accuracy-focused experiment following the paper's ingredients. Declared two fixed BFGS runs in `results/paper_pilot/PROTOCOL.md`: original circuit with optimizer change only, then a six-qubit/six-block fixed-Ising circuit with learned output scales. Keep original train/validation data; reserve a new 128-point holdout until both checkpoints are frozen. All nine tests passed, including Ising unitarity, batched versus scalar predictions, finite-difference gradients, and full model shape/parameter count. No new accuracy measurements yet. This pilot changes the next work item; the original matched-depth/multiple-seed study remains deferred and incomplete.

## 2026-10-29 — paper-inspired pilot, checkpoint 2: frozen before fresh holdout

Original circuit/BFGS: converged in 24 accepted updates and 38 objective evaluations. Final training loss 0.44263792328127377, essentially the same floor as Adam. Validation selected iteration 16 with joint loss 0.6243851717921121; the accuracy target was missed. This run supports that optimizer replacement alone did not resolve the baseline's underfit.

Six-qubit model/BFGS: completed the fixed 250-update budget, 263 objective evaluations, 110 fitted parameters. Solver status 1 (iteration limit), not convergence success. Selected iteration 247 by joint validation loss 0.00000766303486671098. Validation sine MSE 0.000006789835274628413, MAE 0.0010238579570834136; cosine MSE 0.0000008731995920825663, MAE 0.0004177053119073142. Provisional validation target met on both outputs. Reloaded checkpoint reproduced losses. Training process times were 8.38 and 20.85 seconds respectively, but implementation/batching/circuit differences prevent a controlled speed comparison.

Registered the validation-selected model in `results/current_model.json` before reading the new holdout. Both configurations and checkpoints are now frozen. No legacy test or fresh-holdout labels were used in training or selection. No causal entangler-only claim: the larger architecture, rotations, initialization, output scaling, and encoding schedule also changed. Fresh-holdout evaluation pending; no hardware jobs.

## 2026-10-29 — paper-inspired pilot review gate

Fresh 128-input holdout, disjoint from the original splits, evaluated after checkpoint/registry commit ed865bb. Paper-inspired model: sine MSE 0.0000175335480, MAE 0.00165905545, maximum error 0.0295115202; cosine MSE 0.0000223196053, MAE 0.00129491677, maximum error 0.0444751395. Original Adam fresh MAE was 0.354121843 and 0.473919221. The polynomial achieved fresh MSE 0.0000490756146 and 0.000639159020. The larger quantum model improved average errors here, with 110 parameters versus the polynomial's 12 coefficients; no advantage claim.

Important unresolved limitation: on the 401-point grid including exact boundaries, maximum absolute error is 0.1294157783 for sine and 0.3088931500 for cosine. Strong random-holdout averages do not imply uniform accuracy. Future work should explicitly add endpoint and near-boundary training coverage and repeat seeds in a separately declared experiment, reserving another untouched final holdout if tuning continues. Do not mark the full accuracy/generalization problem solved.

All nine tests passed again. The saved-run audit recomputed 4,936 prediction values, checked 120 scalar metrics and 40 CSV rows, verified checkpoint hashes, validation selection, classical coefficients, and fresh-holdout disjointness. Dependency consistency passed. Both comparison and current-model plots were visually inspected; the viewer's legacy-test metrics match the pilot report exactly. `view_model.py` now opens the registered six-qubit checkpoint by default; `view_progress.py` is labeled as the historical Day 3 snapshot. No new training during visualization/audit. Added `review-paper-pilot.ps1` for repeatable checks. Stop after this pilot for inspection; multiple-seed/depth/entangler studies and finite-shot work remain incomplete. No QPU jobs.

| Phase | Status | Evidence |
|---|---|---|
| Day 1: setup, data, circuit | COMPLETE | results/day1/REVIEW.md and benchmark.json |
| Day 2: training loop | COMPLETE | results/day2/smoke/REVIEW.md, audit.json, two saved runs |
| Day 3: fit and baselines | COMPLETE | results/day3/evaluation/REVIEW.md, metrics.csv, audit.json |
| Day 4: depth and seed study | NOT STARTED | Next active milestone for our original circuit family |
| Paper-inspired accuracy pilot | ARCHIVED | Historical comparison; excluded from active project model |
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

## 2026-10-28 — Day 2, checkpoint 1

Implemented joint regression loss (sample mean of summed output errors), checked gradients, strict CPU float64 inputs, and a bounded 30-update Adam runner. Four tests passed: loss convention, all six analytic parameter gradients against central finite differences (1e-6 perturbation, seven decimal places), five-update loss improvement on a representative batch, and actionable input/NaN/detached-gradient errors. The full 48-point smoke run is pending. Configuration: seed 7, one layer, learning rate 0.05. No test-set evaluation or hardware jobs.

## 2026-10-28 — Day 2, checkpoint 2

Completed two independent 30-update runs. Initial training loss 1.9499409826430554, final 0.5813342418080726 (70.1871% reduction). Selected step 26 by validation loss 0.6783771668649673. Minimum gradient norm 0.13706087841433165. Checkpoint reload reproduced its training and validation losses. The audit confirmed identical full loss/gradient histories and checkpoint weights across both runs. Preserved both runs under `results/day2/smoke` and `results/day2/repeat`, including checkpoints, histories, settings, versions, source hashes, and loss plots. Plot visually inspected. The repeat's elapsed wall time includes a long execution interruption; no speed comparison is supported.

Four training tests passed again; dependency consistency and syntax checks passed. Day 1 regression initially failed because its invalid-input test tensors defaulted to float32 after the model began requiring float64. Corrected the benchmark fixtures to float64; the full Day 1 review then passed all seven circuit checks plus data checks. Training results were unaffected. Added an independent saved-run auditor and `review-day2.ps1` for easy inspection. Final roadmap/report update pending. QPU jobs: zero; test labels unused in training and audit.

## 2026-10-28 — Day 2 exit gate

Day 2 deliverables are complete: differentiable joint loss, finite nonzero gradient verification, small optimization run, validation-selected checkpoint, optimizer state, saved configuration, full loss history, loss plot, reproducibility audit, and runnable README commands. Results establish working local training only; Day 3 fitting quality and per-output held-out metrics remain unmeasured. No hardware jobs submitted. Stop for inspection; next phase is the 200–300-update baseline fit and classical/analytic comparisons, after user continuation.

Final user-facing check: `review-day2.ps1` completed successfully, running all four tests and the saved-run audit with repeat comparison. Documentation and tracked filenames contain no assistant branding. All source and experiment artifacts are committed; local environments and temporary caches remain ignored.

## 2026-10-29 — Day 3, checkpoint 1

Declared the evaluation protocol before reading test scores. Fixed one-layer seed-7 training to 250 updates; select by minimum validation loss. Fixed the classical baseline to degree-five polynomial least squares and specified an analytic quantum rotation reference. Generalized the training command to accept a separate configuration file while retaining Day 2 defaults. Day 3 measurements pending.

## 2026-10-29 — Day 3, checkpoint 2

Completed all 250 updates without gradient failures. Training loss: 1.9499409826430554 initially, 0.4426379233622822 finally (77.30% reduction). Validation selected step 83 with loss 0.6320843068249645; the final step was not selected. Minimum gradient norm 2.963715646542432e-06. Saved checkpoint reload passed. Training loop wall time was 85.16 seconds, excluding process imports, plotting, and evaluation; earlier short-run timings were not a reliable predictor of this run. Seven tests passed before training, including known-error metric checks, polynomial coefficient recovery, and the direct-rotation quantum reference at domain boundaries. The existing Day 2 checkpoint/repeat audit still passed. Held-out evaluation and plot inspection pending.

## 2026-10-29 — Day 3 exit gate

Evaluated the validation-selected step-83 checkpoint without changing the architecture. Quantum test MSE: sine 0.22574954843030934, cosine 0.310931501370254. Quantum test MAE: sine 0.4039879100910813, cosine 0.499385033366666. Maximum absolute test error: sine 0.8442161653152518, cosine 0.7749025889244876. The degree-five classical polynomial achieved test MSE 0.00005669323704692145 (sine) and 0.0006416241116465886 (cosine). The direct-rotation analytic quantum reference agreed with target functions within floating-point precision and is explicitly not learned.

Both target-versus-prediction figures and the loss plot were visually inspected. The one-layer QCL visibly underfits and training loss plateaus; a single optimization run does not prove that expressivity alone caused the error. This circuit has six nominal angles, including two final Z rotations that do not affect the measured observables; seven logical gates include one CNOT. Counts are not backend-compiled costs. The preplanned next investigation is the matched 1/2/3-layer, multi-seed comparison with the same per-layer encoding; extra layers may improve capacity but no improvement is assumed. Select later models using validation, not the now-reported test scores.

Final review command `review-day3.ps1` passed all seven tests and the saved-evaluation audit. The audit reconstructed 1,134 prediction values, verified all 18 metric rows in JSON/CSV, verified original split/targets, refitted classical coefficients, and checked checkpoint selection across 251 history rows. Python syntax and whitespace checks passed. All predictions, classical coefficients, selected quantum weights, plots, settings, versions, and provenance are retained. No accuracy success claim, no quantum advantage claim, and zero QPU jobs. Stop for user inspection before Day 4.

## 2026-10-29 — model viewer

Added `view_model.py`: load the saved checkpoint without retraining, compare its predictions with exact NumPy sine/cosine curves on 401 plotting inputs, show signed-error panels and fixed-test metrics, and open an offline browser report. An explicit `--no-open` mode generates a PNG, HTML report, and metrics JSON without browser interaction. The default viewer reproduced the Day 3 test metrics exactly. No model updates or architecture selection performed.

## 2026-10-29 — before/after progress visualization

Added `view_progress.py` and a four-panel figure: reconstructed initial versus selected trained curves for sine/cosine, full loss history, and per-output test MSE across initialization, short-run checkpoint, baseline checkpoint, and polynomial comparison. Verified initial loss against saved step zero and current metrics against the audited evaluation. Recorded experiment training totals 310 updates across three independent runs, not one 310-step model; no new training performed. Preserved the plotted curves and numerical summary. Documented optimization priorities in `results/progress/NEXT_STEPS.md`. The visualization exposes that sine test error worsened from the short-run checkpoint while cosine improved; test scores did not guide checkpoint selection.
