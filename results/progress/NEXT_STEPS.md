# Progress and optimization priorities

Completed experiment training: 30 updates, an independent 30-update repeat, and a fresh 250-update baseline. Total: 310 recorded experiment updates, representing 14,880 training sample presentations over the same 48 inputs. This excludes small unit-test optimizations. The runs were initialized independently, not chained into one 310-update model. The currently displayed checkpoint is step 83 of the 250-update run, selected by validation loss.

Training loss decreased from 1.94994 to 0.44264 by update 250, a 77.30% reduction. That is loss reduction, not accuracy. The model improved from initialization but remains a poor approximation relative to the degree-five classical baseline. The saved short-run checkpoint has better sine test MSE than the current checkpoint, while the current checkpoint has better cosine test MSE; improvement is not monotonic per output. Neither checkpoint is selected using these test scores.

## Next experiment: depth and initialization

Compare 1, 2, and 3 layers across three fixed initialization seeds under the same 250-update budget, split, optimizer, and per-layer encoding. This is the preplanned Day 4 study. Report validation scores, both output errors, parameter/gate counts, run time, and gradient norms. Select using validation only. More layers add capacity and make some intermediate rotations observable, but improvement is not guaranteed.

## Then investigate optimization if needed

1. Try a small predeclared learning-rate comparison or decay schedule if the deeper models still stall. Do not simply add thousands of steps to a flat loss curve.
2. Track sine and cosine validation errors separately while optimizing their equal-weight sum, so joint improvement does not hide deterioration in one output. Any loss-weighting experiment should be declared separately.
3. Consider removing the final Z rotations that do not affect these readouts. This reduces redundant angles; by itself it does not add expressivity. Treat a changed parameterization as a separate experiment rather than mixing it into the fixed-depth benchmark.
4. If depth and initialization do not resolve the fit, compare another feature map as a separately documented architecture experiment. Preserve a fixed validation criterion and avoid tuning against reported test scores.

The test set has already been exposed. Repeatedly adapting the project to these scores would weaken its independence; reserve a new final holdout if substantial exploratory tuning follows. Keep the classical baseline as a reality check. Finite-shot noise and hardware validation come after satisfactory local-model assessment, with explicit permission for real QPU use.
