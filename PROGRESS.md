# Progress and review gates

## Mirrored training and hardware preparation — 2026-10-29, checkpoint 1

User authorized the next model-improvement step and preparation toward real hardware. Declared three same-model branches from update 3,000 to 4,500: continuation control; mirrored main training inputs (96 unique points), same boundary weight 0.5; mirrored data with boundary weight 0.25. Same five-layer two-qubit circuit, scale 0.9, seed 41, Adam state, and rate 0.003. All runs share original/reflected validation plus collar validation. Per-output original-validation MSE and boundary-error guards protect prior progress. New holdout remains unread until checkpoints/selection are committed. Fifteen tests passed, including mirror symmetry/data-role separation and source eligibility. See results/mirror/PROTOCOL.md.

Optional Qiskit SDK installed locally for export verification. IBM Runtime installation is preparation only; no account was contacted and no jobs submitted. The user will provide account/backend details securely; no credential or account identifiers are written into tracked files. Hardware preflight must check current allowance, actual backend target/layout, compatibility and conservative workload before requesting consent to a specific QPU run. The currently supplied chat credential will not be reused or repeated; private local hidden entry is being prepared. No hardware-in-the-loop training is planned.

## Read-only error-peak investigation — 2026-10-29

User requested a report before deciding the next training step. No training or checkpoint selection was performed. Measured signed residuals on a 10,001-input diagnostic grid; refined stationary locations and verified the equal-slope condition with autodifferentiation and central finite differences. Largest highlighted errors: sine +0.04968284 at t=-0.33831377 and -0.04058191 at t=0.65021585; cosine +0.07537767 at t=-0.81186833 and -0.06509371 at t=0.02755903. Full peak list and before/after values are retained.

The largest cosine peak lies in an augmented-training gap between -0.9 and -0.64529362; there are zero original training inputs within ±0.05, versus three near its positive mirror. Mirrored cosine error is about +0.00901, compared with +0.07538 on the negative side. The circuit is not explicitly constrained to cosine evenness or sine oddness; sampled parity defects are about 0.06644 and 0.05037. This supports coverage/symmetry hypotheses, not a unique causal proof.

Other peaks occur near training examples with comparable residuals: the main sine peak has five nearby original inputs; the central cosine dip is 0.00275 from a training input with about -0.06508 error. Current predicted cosine maximum is 0.93619 rather than 1. These are local training underfits as well as generalization issues. Gradient cosines at the frozen model are -0.99086 (original-data versus anchors) and -0.99459 (sine versus cosine components), showing local competing update preferences; this does not prove an unavoidable capacity limit or global optimum. All six highlighted peak magnitudes decreased from the parent at update 1,500 to update 3,000.

Saved report, annotated plot, raw curves, and diagnostic values in `results/error_peaks/`. Model weights, serialized checkpoint, and active registry were verified unchanged. Candidate next tests, not executed: mirrored/stratified training coverage first, followed by a separate guarded loss-weighting ablation if needed. Await the user's next-step decision; do not automatically continue training based on these exposed diagnostic curves.

## Boundary-study review gate — 2026-10-29

All eight trials are complete and retained, including rejected trade-offs. Four three-layer branches resumed update 250 to 1,000. Two five-layer architectures started fresh and trained to 1,500. Two actual same-model refinements resumed the smooth five-layer parent to 3,000. Total new optimization work across branches was 9,000 updates; the active model itself has 3,000 updates, not 9,000. No removed model was reintroduced.

After selection/checkpoint commit 1a4a102, read the new stratified 512-input holdout. Selected model: original two-qubit Rot/CNOT family, five layers, seed 41, fixed encoder scale 0.9, 30 nominal angles, selected step 3,000. Interior MAE improved from 0.13263150/0.11249442 to 0.02361868/0.02440187 (sine/cosine). New boundary-collar maximum errors improved from 0.40202144/0.52810711 to 0.02394069/0.03856940. Dense near-boundary diagnostics found a maximum error about 0.039073. Original-validation joint loss is 0.0026893738 versus Day 4's 0.0494098296. The guard preserved and improved the original validation fit.

Known endpoint maximum error improved from 0.39128746 to 0.02675783, meeting the provisional endpoint goal; these endpoints are now supervised anchors, not independent test inputs. Collar-validation MAE goals also passed. Original-validation MSE is 0.00080905 (sine) and 0.00188032 (cosine); the stricter per-output 0.001 MSE goal remains unmet for cosine. The 1,001-point diagnostic grid maximum errors are 0.04968273/0.07537759, so residual interior error remains. The holdout oversamples boundaries; regional scores are reported rather than claiming uniform-domain aggregate risk.

Thirteen tests passed. The saved-results audit verified 8 trials, 9,008 history rows, all start/best/final optimizer counters and starting states, validation guard/selection, 8 CSV rows, and disjoint stratified holdout data; it reconstructed 32,458 prediction values including dense boundary curves. Old Day 4 audit still passes with default encoder scale 1. Dependency/syntax checks passed. Before/after plots visually inspected. Viewer regeneration completed and its legacy-test metrics then matched the saved selected-model report exactly; an earlier comparison while regeneration was still running read stale viewer data and was repeated successfully after completion. No numerical experiment artifacts were changed to address that ordering issue.

Findings are in `results/boundary/FINDINGS.md`; full data and trial comparisons in `results/boundary/evaluation/`. Stop for inspection before further tuning or finite-shot work. Next confirmation: repeat the successful recipe across seeds and reserved data; investigate residual cosine interior error without selecting on the now-exposed holdout. No QPU jobs.

## Boundary investigation — 2026-10-29, checkpoint 1

Measured sparse training coverage: 2/48 inputs in |t|>=0.9, minimum -0.9743190353 and maximum 0.9339244003, no exact endpoints. Endpoint angle gradients are finite (norm 0.6355755874), so no observed broken parameter-gradient path. Near-endpoint prediction secants increase sharply as distance shrinks; inverse-trigonometric input-angle derivatives are singular at the exact bounds. This is an input-sensitivity/coverage hypothesis, not proof of a single cause. Diagnosis saved without changing weights.

Declared four fixed 750-update branches from the Day 4 step-250 model and Adam state: constant-rate continuation, rate decay, boundary-balanced training with decay, and the same boundary training with input scale 0.9. The last modifies the encoder and is explicitly labeled a warm start under a changed feature map. Same two qubits, three layers, 18 angles, CNOT layout, targets, and no output scaling. Select by balanced original/collar validation score subject to not worsening original-validation loss. New stratified holdout remains unread until checkpoints and selection are committed. All 11 tests passed, including dataset separation, exact optimizer moments/counter restoration, and finite smooth-encoder input gradients at endpoints. See results/boundary/PROTOCOL.md. No training results yet; no QPU jobs.

Boundary checkpoint 2: all four continuation branches completed 750 extra updates (250 to 1,000 per branch). Constant rate: final endpoint maximum error 0.30287156 and original-validation loss 0.04673481. Decay: 0.32545583 / 0.04785396. Boundary-balanced: 0.14715017 / 0.05844260; the final iterate fails the original-validation guard, so an earlier eligible checkpoint is retained. Smooth encoding: 0.11291340 / 0.06986137, with no eligible checkpoint. None met the endpoint goal. No new holdout was generated or read. Full findings saved before follow-up in `results/boundary/PHASE1_FINDINGS.md`.

Declared a separate capacity follow-up before any test evaluation: two fresh five-layer versions of our same two-qubit Rot/CNOT circuit, seed 41, input scales 1.0 and 0.9, 1,500 Adam updates with fixed decay, and the same balanced boundary data and validation guard. These load neither old weights nor moments; they are not continued 3-layer models. Twelve tests passed, including batched/scalar equivalence at five layers and fresh-start state isolation. The independent stratified holdout remains reserved until all six outcomes and validation selection are frozen. No guarantee of improvement or causal single-factor diagnosis.

Boundary checkpoint 3: fresh five-layer models improved the boundary/interior trade-off. Original encoder selected score 0.0067273, endpoint maximum error 0.05849. Input scale 0.9 selected score 0.0022477013, endpoint maximum error 0.02721426, original-validation loss 0.0038949363, collar loss 0.0006004662. Both pass the Day 4 original-validation guard. The smoother model met the endpoint goal and was still improving at its 1,500-update budget. No independent holdout read. Frozen phase-2 selection points to this model.

Declared two same-model refinements from its exact saved angles and Adam moments at update 1,500: 1,500 extra updates at rate 0.003 versus 0.003/0.001/0.0003 decay. They preserve the five-layer circuit, scale 0.9, data and loss. Their original-validation guard is tightened to the parent score 0.0038949363. See PHASE2_FINDINGS.md and protocol addendum. This is genuine resumed training, distinct from the fresh five-layer experiments. The final independent holdout remains reserved for after all eight outcomes and final selection are frozen.

Boundary checkpoint 4, frozen before holdout: both five-layer refinements completed update 1,500 to 3,000 with restored Adam state. Constant rate selected balanced score 0.0015702956 and endpoint maximum error 0.02676; decay selected score 0.0018477 and endpoint maximum error 0.02677. Selected `refine_constant`, step 3,000, by validation only. Original-validation joint loss is about 0.0026894 and collar-validation loss about 0.0004512; both improve from the update-1,500 parent. The original-validation per-output MSE goal may still be unmet and will be reported separately. All eight trial outcomes, including ineligible approaches, are retained. No new holdout access yet. The active model is our original two-qubit circuit with five layers, 30 nominal angles, and fixed encoder input scale 0.9, without learned output scaling or analytic boundary replacement. Thirteen tests passed, including exact five-layer parent-state restoration. Independent stratified evaluation follows only after this checkpoint/selection commit.

## Day 4 exit gate — 2026-10-29

After validation-only selection commit 4a23755, evaluated a new disjoint 256-input holdout (seed 20261030). Selected model MSE: sine 0.028034364, cosine 0.029809244; MAE: sine 0.14600477, cosine 0.13229243; maximum error: sine 0.40201933, cosine 0.52852616. Endpoint-inclusive grid maximum errors: 0.40214201 and 0.52855948. Mean joint validation loss across three seeds decreased from 0.6294399 (1 layer) to 0.1339762 (2) and 0.05893151 (3). All seed results and sample standard deviations are reported. This is substantial progress but not a high-accuracy fit.

Eight tests passed. The study audit reconstructed 13,410 prediction values, checked all nine histories (2,259 rows), verified nine CSV rows, checkpoint hashes, validation selection, and disjoint holdout data. Batched versus original scalar Day 3 loss trajectories differed by at most 1.33226762955e-15. Syntax, whitespace, and dependency checks passed. Comparison plot visually inspected; the default viewer exactly reproduces the selected Day 4 checkpoint's legacy-test metrics. The one-time comparison model and associated tracked files are removed; only old Git commits retain them.

Active model: our original circuit, two qubits, three layers, seed 41, step 250, 18 nominal parameters, 21 logical gates including three CNOTs, no output scales. All nine runs used matched encoding, optimizer, and budget. Depth also changes the number of data re-uploads; no entangler-only, barren-plateau, or quantum-advantage conclusion is warranted. Gradient norm before the final update was 0.00868854, with validation loss still decreasing through step 250. A separately declared budget/learning-rate comparison is the next accuracy investigation; success is not assumed. Stop here for inspection before further training or finite-shot work. No QPU jobs.

## Active direction — 2026-10-29

Day 4 checkpoint 1: eight tests passed, including predictions and full parameter gradients from batched versus scalar circuit execution at depths 1, 2, and 3 (absolute tolerance 1e-12). This is an execution optimization only; circuit gates, feature map, initialization, loss, and optimizer are unchanged. Prepared nine fixed configurations (depths 1/2/3, seeds 7/19/41), 250 Adam updates each, before reading new results. Protocol saved in `results/day4/PROTOCOL.md`. No new holdout has been generated or read.

Day 4 checkpoint 2: all nine matched runs completed, totaling 2,250 optimizer updates. Best joint validation losses by run: depth 1 seeds 7/19/41 = 0.63208431 / 0.62657516 / 0.62966015; depth 2 = 0.11214614 / 0.11156673 / 0.17821568; depth 3 = 0.06857967 / 0.05880504 / 0.04940983. Selected depth 3, seed 41, step 250 using validation only. Its training loss decreased from 1.6612407915 to 0.0409861223. All checkpoint reload checks passed. `results/day4/selection.json` and the active registry are frozen before any new holdout evaluation; new holdout remains unread at this checkpoint. More layers improved validation results in this study, but the provisional target has not been achieved. No alternate model families or hardware jobs were used.

The one-time comparison model, its training tools, tests, configurations, and results have been removed from the current project files at the user's request. Only our original two-qubit circuit family remains. Days 1–3 are complete as implementation/evaluation milestones, not accuracy certification. Day 4's matched-depth and initialization-seed study is next. Earlier comparison commits remain in Git history; no history rewrite was requested.

| Phase | Status | Evidence |
|---|---|---|
| Day 1: setup, data, circuit | COMPLETE | results/day1/REVIEW.md and benchmark.json |
| Day 2: training loop | COMPLETE | results/day2/smoke/REVIEW.md, audit.json, two saved runs |
| Day 3: fit and baselines | COMPLETE | results/day3/evaluation/REVIEW.md, metrics.csv, audit.json |
| Day 4: depth and seed study | COMPLETE | results/day4/evaluation/REVIEW.md and audit.json; accuracy target still unmet |
| Boundary training follow-up | COMPLETE | results/boundary/evaluation/REVIEW.md; endpoint goals met, cosine validation-MSE target still unmet |
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
