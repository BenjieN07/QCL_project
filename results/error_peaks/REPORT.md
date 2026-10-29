# Read-only error-peak investigation

Checkpoint: `results/boundary/runs/refine_constant/best.pt`, update 3000. SHA-256: `ed689a94014ca30637554ee2571013a3a88ed4a8e272f0027a94106406752894`.

The curve is signed residual = model prediction minus exact target, evaluated on 10,001 plotting inputs. These plotting inputs are not a new independent test set. No optimizer updates or checkpoint selection occurred. The checkpoint bytes, active registry, and in-memory weights were verified unchanged.

| Label | Output | t | Prediction | Exact target | Signed error | Parent error at the same t | Original training points within ±0.05 |
|---|---|---:|---:|---:|---:|---:|---:|
| S1 | sin | -0.338314 | -0.824059 | -0.873742 | +0.049683 | +0.061721 | 5 |
| S2 | sin | 0.650216 | 0.850117 | 0.890698 | -0.040582 | -0.046983 | 1 |
| S3 | sin | 0.140408 | 0.391278 | 0.426937 | -0.035660 | -0.042714 | 3 |
| C1 | cos | -0.811868 | -0.754988 | -0.830365 | +0.075378 | +0.083721 | 0 |
| C2 | cos | 0.027559 | 0.931161 | 0.996254 | -0.065094 | -0.077449 | 3 |
| C3 | cos | -0.362337 | 0.439527 | 0.419125 | +0.020402 | +0.027236 | 5 |

## What causes a peak mathematically

The residual increases wherever the prediction slope exceeds the target slope and decreases where it is smaller. At each interior peak/trough the slopes match, while the values remain separated. Both autodifferentiated slopes and independent central finite differences verified these stationary locations. The location of an error peak is not necessarily a maximum of the target function.

## Coverage and fitting evidence

The largest cosine residual (C1) lies at t=-0.811868. Its bracketing augmented training inputs are [-0.9, -0.6452936163539027]; no original or anchor training inputs lie within ±0.05. Validation has a nearby point at -0.813108, which measures the error but does not supply optimizer updates. This is direct evidence of a training-coverage gap, not proof that data coverage is the sole cause.

In contrast, the largest sine error (S1) has 5 original training inputs within ±0.05. Its closest training input is -0.322375, with error +0.049408. The cosine dip (C2) is only 0.002748 from an original training input, whose error is -0.065084. These are also training-fit errors, so simply adding examples cannot be assumed to remove them.

The predicted cosine maximum is 0.936189 near t=-0.0094, while the true maximum is 1 at zero. The central negative cosine residual therefore reflects an under-height peak plus a small shape/position mismatch. Sine extrema likewise have small amplitude/position mismatches.

## Local optimization trade-offs

Cosine similarity between original-data and anchor-loss gradients: -0.990860. Between sine and cosine component gradients: -0.994591. Values near -1 indicate almost opposite parameter-update preferences at these exact weights. Their summed gradient norm is 0.000613931, compared with original/anchor component norms 0.00909935 and 0.00898371.

This is measured local competition, not a proof of an unavoidable Pareto frontier, a global optimum, a barren plateau, or insufficient circuit capacity. The location and sign of per-peak gradient conflicts differ; reducing one point error can increase a different output or region loss. Full per-peak diagnostics are in analysis.json. Hardware/shot noise is not an explanation for this deterministic exact-simulator residual.

## Is error increasing with more training?

All six highlighted residual magnitudes are smaller at update 3,000 than at update 1,500 when compared at the same t. This comparison uses identical architecture, encoding and data, so these peaks are shrinking with the observed continuation, although not vanishing. It does not establish monotonic improvement at every input or guarantee future improvement.

## Candidate next tests — not executed

First, test added stratified interior training coverage in the negative-input gap while preserving all validation/test separation and boundary anchors. Separately, compare loss weighting under per-output and boundary guards: interior errors already present at training examples indicate that coverage alone is not a complete explanation. Keep a same-model continuation as a reference if running new experiments. If conflicts persist, compare initialization seeds before attributing them to an architectural limit. Any future choices should use declared validation criteria and a newly reserved final holdout, not the now-exposed diagnostic curves.

No new training, parameter perturbations, model replacement, or registry changes were performed. Recommendations await the user’s next-step decision.

## Asymmetric data and missing symmetry constraints

Sine is odd and cosine is even, but this circuit/training objective does not explicitly enforce those identities. The largest cosine peak has error +0.075378 at t=-0.811868; at its reflected input t=0.811868, error is only +0.009017. Original training counts within ±0.05 are 0 on the peak side and 3 on the mirrored side. This measured asymmetry is consistent with the coverage problem, without proving a unique cause.

Maximum sampled odd-symmetry defect for sine is 0.050373; even-symmetry defect for cosine is 0.066444. A first controlled next experiment could mirror the training inputs with their correct sine/cosine labels, while retaining boundary anchors and fixed validation guards. That is data augmentation within our existing method, not an analytic output patch. Test loss rebalancing separately because the dense-region peaks show local fitting conflicts too. Neither experiment has been run.
