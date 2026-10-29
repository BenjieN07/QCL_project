# Baseline protocol — declared before evaluation

Train the existing one-layer circuit from initialization seed 7 for 250 full-batch Adam updates at learning rate 0.05. Keep the original data seed and split sizes. Select the earliest checkpoint with minimum validation loss across steps 0–250. Test labels must not affect training, checkpoint selection, or architecture changes.

Compare against degree-five polynomial least-squares regression: six coefficients per output, fitted only on the 48 training inputs. Degree five is fixed before test evaluation. This is a simple classical baseline with 12 fitted coefficients and no iterative optimizer.

Include a no-training quantum analytic reference: RY(pi*t - pi/2) on qubit 0 and RY(pi*t) on qubit 1, with Z readouts. These rotations directly produce sine and cosine; they are an analytic construction, not evidence of learning.

Report per-output MSE, MAE, maximum absolute error on all three splits; plot predictions on the 101-point grid. Test scores describe this predefined baseline only. Retain the one-layer circuit even if it underfits. The already planned Depth study depth study will use validation rather than test scores for selection.

Save all predictions, targets, polynomial coefficients, selected quantum weights, histories, package versions, and source provenance. Check evaluation by independently recomputing metrics and reloading checkpoints. No QPU submission.
