# Mirrored coverage results

Validation-selected branch: **mirrored_rebalance**. All three branches continued the same source weights and Adam state from update 3000 to 4500. Selected checkpoints can precede update 4500.

Fresh stratified holdout: 256 interior, 128 left boundary, 128 right boundary; not a uniformly weighted domain score. No holdout-based selection.

| Model | Interior MAE sin / cos | Left MAE sin / cos | Right MAE sin / cos | Grid max sin / cos |
|---|---|---|---|---|
| before | 0.024432 / 0.025132 | 0.015516 / 0.016368 | 0.012349 / 0.002516 | 0.049683 / 0.075378 |
| continuation_control | 0.019664 / 0.023691 | 0.011010 / 0.013549 | 0.007560 / 0.002688 | 0.041365 / 0.079702 |
| mirrored_training | 0.019760 / 0.020864 | 0.008785 / 0.009754 | 0.008322 / 0.003035 | 0.046935 / 0.059847 |
| mirrored_rebalance | 0.017752 / 0.019721 | 0.010943 / 0.013865 | 0.012565 / 0.006401 | 0.041078 / 0.054862 |

Exact simulation only. The grid diagnoses visible residuals; it is not a proof of maximum error between grid points. See report.json for all branch outcomes, symmetry defects and checkpoint audit.
