# Day 1 review

Validity benchmark: PASS. No training or held-out accuracy evaluation yet.

Inspect targets.png, circuit.png, circuit.txt, data_checks.json, and benchmark.json.

Parameters: 6; logical gates: 7; logical two-qubit gates: 1. These are not backend-compiled counts.

Encoding per layer and wire: RY(arcsin(t)), RZ(arccos(t²)), then trainable Rot(phi, theta, omega). CNOT(0, 1) ends each layer. Readouts: Z0 for sine, Z1 for cosine. Data are re-uploaded at each layer; the same rule will apply to depth comparisons. This circuit does not directly encode pi*t rotations.

Training will minimize the batch mean of the SUM of squared errors across the two outputs. Validation will select architectures; test labels remain reserved for reporting.
