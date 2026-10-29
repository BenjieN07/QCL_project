# Paper-inspired pilot results

Fresh holdout: 128 new inputs, seed 20261029. Checkpoints frozen and registered before evaluation.

| Model | Output | Fresh MSE | Fresh MAE | Fresh maximum error |
|---|---|---:|---:|---:|
| Original Adam | sin | 0.184672236 | 0.354121843 | 0.852494244 |
| Original Adam | cos | 0.28666827 | 0.473919221 | 0.797177612 |
| Original BFGS | sin | 0.197137219 | 0.364937909 | 0.876897001 |
| Original BFGS | cos | 0.315778078 | 0.495735406 | 0.816833069 |
| Paper-inspired BFGS | sin | 1.7533548e-05 | 0.00165905545 | 0.0295115202 |
| Paper-inspired BFGS | cos | 2.23196053e-05 | 0.00129491677 | 0.0444751395 |
| Polynomial | sin | 4.90756146e-05 | 0.00474141062 | 0.023906752 |
| Polynomial | cos | 0.00063915902 | 0.0191407496 | 0.0755745128 |

BFGS alone reached the original circuit’s Adam training-loss floor. The six-qubit/six-block Ising model with learned scales met the declared validation target. This supports the new bundled design, not a causal claim about the entangler alone.

The larger model ran 250 accepted BFGS updates and 263 objective evaluations, selecting iterate 247. It stopped at its iteration budget (solver success=False), not at the convergence tolerance. The original BFGS run converged after 24 accepted updates and 38 objective evaluations. All gradients were analytic simulator gradients. The new implementation batches inputs, so timings are not a controlled optimizer or circuit speed comparison.

Scaled prediction ranges without clipping: {'fresh': {'min': -1.0006137856469934, 'max': 0.9991851043321109}, 'grid': {'min': -1.002061177957568, 'max': 0.9998629183438743}}. The 401-point grid includes both endpoints; grid maximum errors are available in metrics.json. Random-holdout accuracy does not guarantee uniform accuracy at every point.

Important boundary limitation: maximum absolute error on the endpoint-inclusive grid is 0.129416 for sine and 0.308893 for cosine. The model is much better on average but not yet uniformly accurate over the closed interval. Endpoint and near-endpoint training coverage should be studied in a separately declared experiment; do not tune against this now-exposed fresh holdout.

Only one initialization and Hamiltonian seed were studied. Multiple-seed robustness, controlled entangler ablation, the earlier matched-depth study, finite shots, and hardware validation remain uncompleted. The new holdout is now exposed and must not guide later tuning.

This is a two-output, noiseless extension inspired by the paper, not an exact reproduction. No hardware jobs, quantum advantage, or accuracy percentage is claimed.
