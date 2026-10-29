# Boundary-study review

All models use our original two-qubit Rot/CNOT circuit family. No output clipping, analytic boundary replacement, or removed model.

| Trial | Layers | Input scale | Start | Eligible | Selected update | Validation score | Endpoint maximum error |
|---|---:|---:|---|---|---:|---:|---:|
| continue_constant | 3 | 1.0 | resume @ 250 | True | 1000 | 0.079764246 | 0.30287156 |
| continue_decay | 3 | 1.0 | resume @ 250 | True | 1000 | 0.088963601 | 0.32545583 |
| boundary_balanced | 3 | 1.0 | resume @ 250 | True | 252 | 0.089344089 | 0.34561267 |
| boundary_smooth | 3 | 0.9 | resume @ 250 | False | 1000 | 0.038378214 | 0.1129134 |
| capacity_original | 5 | 1.0 | fresh @ 0 | True | 1500 | 0.0067272814 | 0.058488562 |
| capacity_smooth | 5 | 0.9 | fresh @ 0 | True | 1500 | 0.0022477013 | 0.027214258 |
| refine_constant | 5 | 0.9 | resume @ 1500 | True | 3000 | 0.0015702956 | 0.026757827 |
| refine_decay | 5 | 0.9 | resume @ 1500 | True | 3000 | 0.0018477051 | 0.026768674 |

Selected by balanced validation score with the original-validation non-degradation guard: **refine_constant**. All checkpoints and this selection were frozen before the new holdout was read.

| Model / region | Output | MAE | MSE | Maximum error |
|---|---|---:|---:|---:|
| Depth study source / new_interior | sin | 0.1326315 | 0.022687424 | 0.26381145 |
| Depth study source / new_interior | cos | 0.11249442 | 0.016808943 | 0.28884117 |
| Depth study source / new_left_boundary | sin | 0.0852973 | 0.0098362065 | 0.16496623 |
| Depth study source / new_left_boundary | cos | 0.35531338 | 0.12768384 | 0.40546182 |
| Depth study source / new_right_boundary | sin | 0.24615584 | 0.073542897 | 0.40202144 |
| Depth study source / new_right_boundary | cos | 0.40257405 | 0.1706153 | 0.52810711 |
| Depth study source / new | sin | 0.14917903 | 0.032188488 | 0.40202144 |
| Depth study source / new | cos | 0.24571907 | 0.082979257 | 0.52810711 |
| Depth study source / endpoint | sin | 0.15048193 | 0.024786422 | 0.19675947 |
| Depth study source / endpoint | cos | 0.3619983 | 0.13190062 | 0.39128746 |
| Depth study source / grid | sin | 0.13810858 | 0.024911555 | 0.40228908 |
| Depth study source / grid | cos | 0.13607048 | 0.030275853 | 0.52846545 |
| refine_constant / new_interior | sin | 0.023618676 | 0.00075600409 | 0.049654007 |
| refine_constant / new_interior | cos | 0.024401868 | 0.0010685087 | 0.075344886 |
| refine_constant / new_left_boundary | sin | 0.016235851 | 0.00031956967 | 0.023940693 |
| refine_constant / new_left_boundary | cos | 0.01703868 | 0.00038214738 | 0.038569402 |
| refine_constant / new_right_boundary | sin | 0.013829745 | 0.00022373348 | 0.018634989 |
| refine_constant / new_right_boundary | cos | 0.0022441898 | 6.5517543e-06 | 0.0069482646 |
| refine_constant / new | sin | 0.019325737 | 0.00051382783 | 0.049654007 |
| refine_constant / new | cos | 0.017021651 | 0.00063142912 | 0.075344886 |
| refine_constant / endpoint | sin | 0.011585026 | 0.00013947434 | 0.013878825 |
| refine_constant / endpoint | cos | 0.017718595 | 0.00039565632 | 0.026757827 |
| refine_constant / grid | sin | 0.023663526 | 0.00073827902 | 0.04968273 |
| refine_constant / grid | cos | 0.023781244 | 0.0010458009 | 0.075377587 |

Provisional goals: {"endpoint_max_error_le_0_05": true, "each_collar_validation_mae_le_0_03": true, "each_original_validation_mse_le_0_001": false}

The 512-input holdout is stratified: 256 interior, 128 left collar, 128 right collar. Its aggregate oversamples boundaries and is not uniform-domain risk. It was not used for stopping, checkpoint selection, or tuning. It is now exposed. Original 24-input test metrics are available separately in metrics.json.

Exact endpoints are supervised training anchors for augmented trials; endpoint error is not independent test evidence. Region holdout errors evaluate generalization near those constraints. The 1,001-point grid and finer boundary zooms are diagnostics, not proof of uniform error bounds.

Four trials resume both weights and Adam state from update 250; their 750 extra updates are independent branches. Two five-layer trials start fresh with 1,500 updates and zero optimizer moments. Two refinements then resume the five-layer parent at update 1,500 and add 1,500 updates, with unchanged encoder, architecture, and training data. Their guard is tightened to their parent validation loss. Encoder scale 0.9 changes the feature map; it is not a fitted output multiplier. Fresh architecture trials must not be described as unchanged-model continuation.

The boundary runner uses the explicit schedule and the recorded learning_rate history. An inherited legacy learning_rate field in raw configurations is not used by this runner. No trial uses the independent holdout for choosing a schedule.

One initialization seed per follow-up architecture was studied. No hardware, finite-shot, advantage, or percentage-accuracy claim. Rejected approaches and final states are retained. Inspect the before/after figures and each run’s history.png for optimization trade-offs.
