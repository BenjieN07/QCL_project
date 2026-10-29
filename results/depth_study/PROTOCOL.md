# Depth study: matched depth and initialization study

Use only our original two-qubit QCL circuit: encode RY(arcsin(t)), RZ(arccos(t²)), apply a trainable Rot per wire, then CNOT(0,1), repeated at every layer. No Ising evolution, fitted output scales, or alternate model families.

Fixed grid: depths 1, 2, 3 crossed with initialization seeds 7, 19, 41. Nine fresh training runs, each 250 full-batch Adam updates at learning rate 0.05. Same original 48 training, 16 validation, and 24 legacy-test inputs, data seed 2026. No early stopping or architecture adjustments within this grid. Angle initialization remains normal with standard deviation 0.1. Validate batched predictions and gradients against scalar circuit calls before training.

For each run retain the earliest checkpoint with minimum joint validation loss over steps 0–250. Select the active run with the smallest joint validation loss; exact ties prefer fewer layers, then smaller seed. No test-based selection. Report variation across all three seeds per depth, not just the best result.

Predeclared evaluation: per-output MSE, MAE, and maximum error; training/validation loss histories; initial/final gradient norms; nominal parameter count; logical gates/CNOTs; measured loop runtime. State that final rotations can be unobservable and nominal parameter count does not equal identifiable dimension. Record initial and final frozen checkpoint provenance. Do not infer a barren plateau from small circuits.

Keep legacy-test results for reporting. Only after all nine models and the validation-only registry are frozen, generate a new disjoint 256-input holdout, seed 20261030. Do not read it during training, selection, or report development. Check disjointness from original data and the previously exposed seed-20261029 holdout (inputs only, no targets needed). This new holdout is a one-time post-selection evaluation, not a tuning set.

For the selected checkpoint, separately assess the 401-point plotting grid including endpoints. This is a diagnostic, not independent test evidence or a selection criterion. A provisional good-fit target remains per-output validation MSE <=0.001 and MAE <=0.03. Report misses; the study is complete when the matched experiments and evidence are verified, even if the original circuit family still underfits.

Save each configuration, best weights, optimizer state, history, versions, source hashes, and loss plot. Compare the depth-1/seed-7 trajectory with the original Baseline run to verify batching preserved results within 1e-10. Stop after Depth study for review. No hardware jobs or finite-shot claims.
