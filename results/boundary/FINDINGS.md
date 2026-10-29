# What improved the boundaries

Measured before training: only 2 of 48 original training inputs were in |t|>=0.9, and neither endpoint was trained. The largest positive training input was 0.9339244. Parameter gradients at the endpoints were finite, so the observed failure was not a NaN/broken parameter-gradient path. Near-endpoint prediction sensitivity was large; the input derivatives of the inverse-trigonometric encoding become singular at the exact bounds.

The controlled results matter more than the initial hypotheses. Three-layer continuation to 1,000 updates made only modest progress. Added boundary training improved boundary validation while worsening original validation, so the final three-layer boundary-focused iterates were not promoted. Fixed input scale 0.9 helped the boundary score but did not rescue the interior at that configuration. These results do not prove a unique cause or a hard architectural limit.

Two fresh five-layer experiments then used the same two-qubit Rot/CNOT circuit, data, schedules, and seed, differing only in input scale. Both improved the trade-off; scale 0.9 performed better in this trial. Increasing depth also changed initialization and optimization budget relative to the three-layer continuations, so the whole gain cannot be attributed solely to capacity or smoothing. The successful configuration combines boundary supervision, more circuit layers, a finite-slope encoder on the domain, and further optimization.

The selected five-layer model trained fresh to update 1,500, then resumed exactly the same weights and Adam moments to update 3,000. Constant rate 0.003 beat the tested decaying refinement on the declared validation score. The updated model still uses our own circuit family and 30 nominal angles, with no output scaling, target-function gates, interpolation patch, or clipped predictions.

## Independent results

Selection was frozen in commit 1a4a102 before generating the new stratified 512-input holdout. On its 256 interior inputs, mean absolute error fell from 0.13263 to 0.02362 for sine and 0.11249 to 0.02440 for cosine. On the 256 boundary-collar inputs, maximum error fell from 0.40202 to 0.02394 for sine and 0.52811 to 0.03857 for cosine. These regions are reported separately; their combined distribution is not uniform over the domain.

The maximum known exact-endpoint error fell from 0.39129 to 0.02676. Endpoints were supervised training anchors in the improved model, so that number demonstrates constraint fitting, not independent generalization. New collar samples and dense boundary diagnostics check nearby behavior without substituting analytic predictions. The dense diagnostic maximum boundary error is about 0.03908.

The endpoint and collar-validation goals passed. The original-validation sine MSE is 0.00080905 and cosine MSE is 0.00188032, so the stricter 0.001-per-output goal remains unmet for cosine. Across the 1,001-point diagnostic grid, the largest errors are about 0.04968 (sine) and 0.07538 (cosine), now in the interior rather than an abrupt endpoint collapse. Grid samples are not a rigorous uniform error bound.

Only one initialization seed was used in the five-layer/refinement follow-ups. The next confirmation should repeat the successful recipe across seeds and independently reserved data, while protecting the boundary constraints and studying the remaining cosine interior error. Do not tune against the now-exposed holdout. Finite shots and hardware remain unrun. No quantum-advantage or percentage-accuracy claim follows from this small regression task.
