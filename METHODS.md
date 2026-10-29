# Methods overview

## Quantum model

Each real input t is encoded on two qubits using RY(asin(s·t)) and RZ(acos((s·t)²)). Every layer applies trainable single-qubit Rot gates and a CNOT. Two Pauli-Z expectation values predict sine and cosine. The selected model uses five layers, seed 41 and fixed input scale s=0.9. No fitted output multiplier, clipping or analytic target substitution is used.

## Training and comparisons

PennyLane simulates the circuit; PyTorch differentiates its exact outputs and Adam updates the angles. Loss is mean squared error summed over both outputs. Fresh architecture experiments initialize new weights; continuation restores both weights and Adam moments.

| Approach | Controlled comparison | Observed outcome |
|---|---|---|
| Circuit capacity | 1/2/3 layers, three seeds each, 250 updates | Three layers improved fit; five-layer follow-ups improved it further |
| Learning rate | Constant versus decaying schedules | Constant 0.003 gave the better five-layer refinement |
| Boundary supervision | Original 48 inputs versus 12 additional anchors | Reduced endpoint collapse, with interior/boundary tradeoffs |
| Input scaling | Scale 1.0 versus 0.9 with five layers | Scale 0.9 improved fitting and avoided evaluating the encoder at singular endpoints |
| Mirrored data | Add -t with correct labels: 48 → 96 main inputs | Improved symmetry and interior fit beyond extra training alone |
| Loss weighting | Boundary weight 0.5 versus 0.25 | Weight 0.25 selected at update 4,288; later checkpoints violated the endpoint guard |
| Circuit batching | Same 108 inputs and gradients, scalar versus vectorized | Median CPU pass 2.164 s → 0.0188 s; weights unchanged |

These are changes within the QCL approach. We did not compare Adam against other optimizer families. The [full experiment table](results/progress/experiments.csv) retains schedules, budgets, eligibility, errors and recorded loop times, including unsuccessful approaches.

## Benchmarking

Original split: 48 training, 16 validation and 24 test inputs. Boundary trials add 12 supervised anchors and ten collar-validation inputs. Mirrored trials use 96 main training inputs and 32 original/reflected validation inputs. Data seeds and exact configurations are saved per experiment.

Final selection minimizes balanced symmetric/collar validation error, while protecting each output's original-validation MSE and keeping endpoint/collar maximum error ≤0.05. Fresh holdouts were evaluated only after model selection was committed. The latest holdout has 256 interior inputs and 128 per boundary collar; regional metrics are separate because it is not uniformly weighted over the domain.

The progress chart uses the same original 24 reporting inputs for every milestone. Its horizontal axis represents experiment order, not elapsed time or a single uninterrupted training run. Archived loop times are not controlled cross-run speed comparisons. The separate batching benchmark alternates scalar/batched passes for five repetitions with one CPU thread and checks prediction/gradient agreement.

## Measurement and hardware

Ideal sampling uses joint two-qubit probabilities at nine fixed inputs, with 256/1,024/4,096 shots and 50 repeats. It excludes device noise. The real-qubit pilot uses frozen weights at three matching inputs on ibm_fez. Both outputs share one measurement basis; classical bit 0 is sine and bit 1 is cosine. [Hardware evidence](results/hardware/real_pilot/REPORT.md)

The 2018 [QCL paper](https://arxiv.org/abs/1803.00745) demonstrates simulated learning. Our architecture differs. Physical-qubit training, larger hardware sweeps and quantum advantage have not been demonstrated here.

All 45 saved checkpoint files retain their original bytes and provenance hashes. The checkpoint reader adapts historical metadata to the current experiment paths without altering trained weights or optimizer state.
