"""Circuit validation validity benchmark, not an accuracy or training benchmark."""
import importlib.metadata
import json
import platform
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import torch
from .model import QCLModel


def main():
    config = json.loads(Path('configs/smoke.json').read_text())
    out = Path('results/validation')
    data = np.load(out / 'data.npz')
    model = QCLModel(config['layers'], config['parameter_seed'])
    # Test labels are never consulted in the Circuit validation benchmark.
    inputs = torch.tensor(np.concatenate(([-1., 0., 1.], data['train_t'])), dtype=torch.float64)
    started = time.perf_counter()
    predictions = model(inputs).detach()
    seconds = time.perf_counter() - started
    repeated = model(inputs).detach()
    assert predictions.shape == (len(inputs), 2)
    assert torch.isfinite(predictions).all() and predictions.abs().max() <= 1 + 1e-12
    assert torch.equal(predictions, repeated)
    assert torch.equal(predictions, QCLModel(config['layers'], config['parameter_seed'])(inputs).detach())
    changes = []
    # Check influence on each readout independently.
    for wire in range(2):
        altered = QCLModel(config['layers'], config['parameter_seed'])
        with torch.no_grad():
            altered.weights[0, wire, 1] += 0.2
        changes.append((altered(inputs).detach() - predictions).abs().max(dim=0).values.tolist())
    assert all(max(row[j] for row in changes) > 1e-6 for j in range(2))
    for invalid in (torch.tensor([1.1], dtype=torch.float64),
                    torch.tensor([float('nan')], dtype=torch.float64),
                    torch.tensor([], dtype=torch.float64)):
        try:
            model(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid input accepted')
    sample = torch.tensor(0.25, dtype=torch.float64)
    diagram = qml.draw(model.circuit)(sample, model.weights)
    (out / 'circuit.txt').write_text(diagram + '\n', encoding='utf-8')
    fig, _ = qml.draw_mpl(model.circuit)(sample, model.weights)
    fig.savefig(out / 'circuit.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    tape = qml.workflow.construct_tape(model.circuit)(sample, model.weights)
    report = {
        'status': 'PASS', 'purpose': 'Untrained circuit validity only', 'config': config,
        'checks': ['shape', 'finite', 'range', 'repeatability', 'seed_recreation',
                   'both_readouts_parameter_sensitive', 'invalid_inputs_rejected'],
        'output_shape': list(predictions.shape), 'output_min': predictions.min().item(),
        'output_max': predictions.max().item(), 'parameter_perturbation_max_changes': changes,
        'parameters': model.weights.numel(), 'logical_gates': len(tape.operations),
        'logical_two_qubit_gates': sum(len(op.wires) == 2 for op in tape.operations),
        'batch_seconds': seconds, 'timing_note': 'Single CPU batch including first-call overhead; not a speed comparison',
        'python': platform.python_version(),
        'versions': {p: importlib.metadata.version(p) for p in ('pennylane', 'torch', 'numpy', 'matplotlib')},
        'qpu_jobs': 0,
    }
    (out / 'benchmark.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez(out / 'untrained_predictions.npz', inputs=inputs.numpy(), predictions=predictions.numpy(),
             weights=model.weights.detach().numpy())
    (out / 'REVIEW.md').write_text(
        '# Circuit validation review\n\nValidity benchmark: PASS. No training or held-out accuracy evaluation yet.\n\n'
        'Inspect targets.png, circuit.png, circuit.txt, data_checks.json, and benchmark.json.\n\n'
        f'Parameters: {report["parameters"]}; logical gates: {report["logical_gates"]}; '
        f'logical two-qubit gates: {report["logical_two_qubit_gates"]}. These are not backend-compiled counts.\n\n'
        'Encoding per layer and wire: RY(arcsin(t)), RZ(arccos(t²)), then trainable Rot(phi, theta, omega). '
        'CNOT(0, 1) ends each layer. Readouts: Z0 for sine, Z1 for cosine. '
        'Data are re-uploaded at each layer; the same rule will apply to depth comparisons. '
        'This circuit does not directly encode pi*t rotations.\n\n'
        'Training will minimize the batch mean of the SUM of squared errors across the two outputs. '
        'Validation will select architectures; test labels remain reserved for reporting.\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
