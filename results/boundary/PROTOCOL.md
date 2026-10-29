# Boundary investigation and continuation study

Use our original two-qubit, three-layer circuit and its Day 4 seed-41 checkpoint at update 250. The removed one-time model is not used. The goal is to reduce boundary collapse while improving the overall fit, not simply accumulate updates.

## Diagnosis before training

Measure training coverage, exact endpoint predictions, finite parameter gradients, and near-endpoint prediction sensitivity. The original data contain only two inputs with |t|>=0.9 and no exact endpoints. Its largest positive training input is about 0.9339. The derivatives of arcsin(t) and arccos(t²) become unbounded as |t| approaches 1. This concerns sensitivity to the input, not proof that parameter-gradient optimization is broken. Separate measured facts from possible causes.

## Four fixed approaches

Each starts from identical saved angles AND the saved Adam optimizer state, then performs 750 additional updates (250 to 1,000 total). No random reinitialization. No trial changes the qubit/layer count, trainable rotations, CNOT layout, targets, or output scaling.

1. `continue_constant`: original 48 training inputs, constant learning rate 0.05. Tests whether more optimization alone helps.
2. `continue_decay`: same inputs; rates 0.05, 0.01, 0.003 for successive 250-update blocks. Isolates the scheduling change.
3. `boundary_balanced`: same decay plus 12 fixed training anchors: both signs of 0.9, 0.95, 0.98, 0.995, 0.999, 1. Training objective is half original-sample mean joint squared error plus half anchor-sample mean joint squared error. This isolates targeted coverage and weighting together; it is not a density-only ablation.
4. `boundary_smooth`: same as (3), with fixed input preprocessing x=0.9*t before the existing encoding. This keeps the circuit family but changes its feature map; it is a warm start under a modified encoder, not an unchanged-model continuation. It avoids the encoding's endpoint singularities without inserting the target sin(pi*t) or cos(pi*t) into the circuit. Adam moments are restored for the same controlled starting optimizer state.

Training anchors are known supervised constraints, including endpoints. Endpoint errors are training/diagnostic errors, never independent test evidence for augmented trials. Do not clip predictions or replace the model near boundaries with an analytic answer.

## Validation and selection

Keep the original 16 validation inputs and add ten distinct collar-validation inputs: both signs of 0.91, 0.93, 0.97, 0.99, 0.998. Verify that validation inputs do not overlap training inputs/anchors. Use these labels for checkpoint selection, not optimizer updates. They are tuning data, not final test data.

Score = half original-validation mean joint squared error plus half collar-validation mean joint squared error. An eligible candidate must not worsen the original-validation joint loss relative to the source checkpoint (tolerance 1e-12). Within a trial select the earliest eligible checkpoint with minimum score; if none is eligible, retain the unconstrained best only for reporting and exclude it from global selection. Compare all trials by the same score/guard. Exact ties follow the declared trial order. Initialization is a candidate, guaranteeing an eligible baseline. Freeze all candidates and the selected registry before reading a new holdout.

Provisional goals: maximum absolute endpoint error <=0.05; collar-validation MAE <=0.03 for each output; original-validation MSE <=0.001 for each output. Report which are met and any trade-offs. These are experimental goals, not promises.

## Independent post-selection evaluation

Only after checkpoint/selection commit, generate a fresh stratified holdout, seed 20261031: 256 inputs uniformly in (-0.9,0.9), 128 in (-1,-0.9), and 128 in (0.9,1). Verify disjointness from original splits, training anchors, collar validation, and previously exposed random holdouts (seeds 20261029 and 20261030). Report all three regions separately; the aggregate deliberately oversamples boundaries and is not uniform-domain risk. Do not use it for selection, stopping, or further tuning. Evaluate a separate 1,001-point plotting grid including exact endpoints to detect narrow spikes. Preserve before/after predictions and all trial outcomes, including rejected approaches.

Save configs, start/best/final model and optimizer states, 751-row histories, learning rates, gradient norms, versions, source hashes, and plots. Independently verify restored optimizer step counters, selected losses, checkpoint reloads, and saved reports. Stop for review after this study. No alternate model families, finite-shot claims, or hardware jobs.
