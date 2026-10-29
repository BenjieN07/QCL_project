# Depth study review

Original two-qubit circuit only. Nine matched 250-update Adam runs; three seeds per depth.

| Depth | Parameters | CNOTs | Validation loss mean ± sample SD | New-holdout sine MSE mean ± SD | New-holdout cosine MSE mean ± SD |
|---|---:|---:|---:|---:|---:|
| 1 | 6 | 1 | 0.6294399 ± 0.00276 | 0.2475882 ± 0.0235 | 0.2554404 ± 0.0205 |
| 2 | 12 | 2 | 0.1339762 ± 0.0383 | 0.05500686 ± 0.0101 | 0.06181325 ± 0.0125 |
| 3 | 18 | 3 | 0.05893151 ± 0.00959 | 0.02808255 ± 0.00125 | 0.02629055 ± 0.00305 |

Active checkpoint: depth 3, seed 41, step 250, selected using joint validation loss only. The new 256-input holdout was generated after selection was frozen.

| Selected output | New-holdout MSE | New-holdout MAE | New-holdout maximum error | Endpoint-inclusive grid maximum error |
|---|---:|---:|---:|---:|
| sin | 0.028034364 | 0.14600477 | 0.40201933 | 0.40214201 |
| cos | 0.029809244 | 0.13229243 | 0.52852616 | 0.52855948 |

Provisional per-output validation target met: False. That target is MSE <=0.001 and MAE <=0.03 on each output. Study completion is not an accuracy claim.

Nominal parameters are 6 per layer; each layer has 7 logical gates including one CNOT. Rot is counted as one logical gate; these are not compiled hardware costs. Two terminal Z rotations are unobservable in these readouts. Extra layers also re-upload data, so this is a comparison of the predeclared layered architecture, not an isolated entanglement-only effect.

Three seeds provide a small sensitivity study, not a broad statistical guarantee. No barren-plateau or advantage claim follows. Seeds share a dataset and are not independent dataset replicates. The new holdout is now exposed; do not tune against its scores. Boundary coverage and feature-map changes require separate protocols if the fit remains poor. No finite-shot or QPU jobs ran.

Batching versus original scalar Baseline trajectory maximum loss difference: 1.33e-15. Timing measures CPU optimization loops under the same batched implementation; no cross-hardware speed claim.
