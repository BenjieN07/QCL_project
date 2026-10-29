# Paper-inspired accuracy pilot — declared before running

Source: https://arxiv.org/pdf/1803.00745, Section III and footnote 25. Reference solver documentation: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-bfgs.html.

Two predeclared runs, both with analytic simulator gradients and SciPy BFGS (maximum 250 accepted iterations, gradient tolerance 1e-7):

1. Original two-qubit, one-layer model, same seed-7 initialization: isolate optimizer change relative to the Adam baseline.
2. Six-qubit, six-block paper-inspired model. Encode RY(arcsin(t)), then RZ(arccos(t²)) once on each wire. Each block executes RX–RZ–RX per wire, then fixed exp(-i*10*H), with H = sum a_i X_i + sum J_ij Z_i Z_j. Fixed coefficients drawn uniformly from [-1,1] with Hamiltonian seed 2026. Initialize angles uniformly on [0,2π] with parameter seed 7. Learn separate multiplicative scales (initially one) on Z0 and Z1. Total fitted parameters: 110.

Use the same 48 training and 16 validation inputs as before. Loss remains the sample mean of the sum of squared sine/cosine errors. Select the earliest accepted iterate with minimum validation loss, including initialization. Line-search evaluations are counted separately and are not eligible checkpoints. Stop at the declared budget or solver termination, recording success or failure accurately. A non-success solver status is not automatically an invalid fit.

Provisional accuracy target: validation MSE <= 0.001 separately on sine and cosine, plus validation MAE <= 0.03 on each. Report misses honestly. Scale-adjusted predictions may leave [-1,1]; do not clip them to improve metrics.

Freeze all configurations before inspecting either run's evaluation. Compare old test results for context only. A fresh 128-input holdout, seed 20261029, will be generated and read only in a separate post-training evaluation command after both validation-selected checkpoints are frozen. Verify disjointness from original splits. Do not use this holdout for selection, tuning, or stopping.

Differences from the paper: two simultaneous trigonometric outputs, 48 rather than 100 teacher inputs, two learned scales, a newly drawn Hamiltonian, explicitly specified block ordering, noiseless simulation, and an explicit optimizer budget. This is not an exact reproduction. The second run changes several factors together; improvement cannot be attributed specifically to its entangler, qubit count, depth, initialization, or scaling. A matched entangler ablation and multiple seeds remain later work. Dense exact Ising evolution has no hardware cost interpretation; no QPU jobs.

Stop after this pilot and its verified report for inspection. The earlier planned matched-depth/multiple-seed study is deferred, not marked complete.
