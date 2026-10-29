# Progress summary

| Milestone | Completed result |
|---|---|
| Circuit and training validation | Reproducible data, finite gradients and checkpoint restoration |
| Baseline | One-layer fit and classical polynomial reference |
| Depth/seed comparison | Nine matched runs; three layers improved the initial fit |
| Boundary optimization | Eight trials covering schedules, anchors, encoding scale and five-layer capacity |
| Mirrored-data optimization | Three matched continuations; selected update 4,288 |
| Efficiency | Equivalent batched gradients; measured median pass 2.164 s → 0.0188 s |
| Measurement benchmark | Three shot budgets, 50 ideal-sampling repeats each |
| Real hardware | One three-input ibm_fez pilot, 3,072 shots, 3 seconds reported usage |

**Current model:** two qubits, five layers, scale 0.9. Same 24-input test MSE: **0.000428 sine / 0.000596 cosine**, versus 0.225750 / 0.310932 for the initial trained baseline. [Full progress table and graph](results/progress/REPORT.md)

The active lineage trained fresh to 1,500 updates, continued to 3,000, then selected update 4,288 from a 4,500-update branch. Other branches are separate experiments, not extra sequential updates on this model. All training used a simulator; real hardware evaluated frozen parameters.

**Remaining limitations:** endpoint error 0.049996 is close to its guard, some boundary errors worsened, and original-validation cosine MSE 0.001314 exceeds the 0.001 goal. Hardware coverage is only three inputs.

**Next priorities:** improve boundary protection without losing interior fit; review whether to extend hardware coverage. Further training or QPU work needs a new experiment scope.

**Repository cleanup:** experiment-based names, one review command, short documentation and complete milestone visualization. The renamed saved-result audits and 19 tests pass; checkpoint bytes are preserved. No training or hardware submissions were performed during cleanup.
