# Model improvement

All rows use the same 24 reporting inputs. Models were selected using validation, never this comparison.

| Milestone | Layers | Selected update | Sine MSE | Cosine MSE |
|---|---:|---:|---:|---:|
| Untrained | 1 | 0 | 0.9934 | 0.616464 |
| One layer | 1 | 83 | 0.22575 | 0.310932 |
| Depth study | 3 | 250 | 0.0293532 | 0.0416375 |
| Five layers | 5 | 1500 | 0.0011951 | 0.0015056 |
| Continued fit | 5 | 3000 | 0.000750888 | 0.00102286 |
| Mirrored tuning | 5 | 4288 | 0.00042842 | 0.000595559 |

The five-layer model started fresh. Its later continuations restored both weights and Adam moments. Independent branches are not additional sequential updates on one model.

experiments.csv compares all 20 depth, boundary and mirrored-training runs, including rejected checkpoints. Loop times are archived measurements and are not a controlled cross-run speed test.

Hardware evidence is separate: three frozen-model inputs on physical qubits, not hardware training or full-domain validation. See ../hardware/real_pilot/REPORT.md.
