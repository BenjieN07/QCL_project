"""Evaluate a frozen Day 3 checkpoint against fixed classical and analytic references."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import torch
from .data import make_data
from .model import QCLModel


def metrics(predictions, targets):
    predictions, targets = np.asarray(predictions), np.asarray(targets)
    if predictions.shape != targets.shape or predictions.ndim != 2 or predictions.shape[1] != 2 or not len(predictions):
        raise ValueError('Predictions and targets must share nonempty shape (batch, 2)')
    if not np.isfinite(predictions).all() or not np.isfinite(targets).all():
        raise ValueError('Predictions and targets must be finite')
    error = predictions - targets
    return {name: {'mse': float(np.mean(error[:, j] ** 2)),
                   'mae': float(np.mean(np.abs(error[:, j]))),
                   'max_absolute_error': float(np.max(np.abs(error[:, j])))}
            for j, name in enumerate(('sin', 'cos'))}


def polynomial_features(inputs, degree):
    return np.polynomial.polynomial.polyvander(inputs, degree)


def analytic_reference(inputs):
    @qml.qnode(qml.device('default.qubit', wires=2, shots=None))
    def circuit(t):
        qml.RY(np.pi * t - np.pi / 2, wires=0)
        qml.RY(np.pi * t, wires=1)
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))
    return np.column_stack(circuit(np.asarray(inputs)))


def evaluate(run, output):
    if (output / 'metrics.json').exists():
        raise FileExistsError('Evaluation exists; use a fresh --output directory')
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((run / 'config.json').read_text())
    summary = json.loads((run / 'summary.json').read_text())
    checkpoint = torch.load(run / 'best.pt', weights_only=True)
    with (run / 'history.csv').open() as stream:
        history = list(csv.DictReader(stream))
    best_step = int(np.argmin([float(r['validation_loss']) for r in history]))
    assert checkpoint['step'] == summary['best_step'] == best_step
    assert checkpoint['config'] == summary['config'] == config
    data = make_data(config)
    # Confirm the original split has not changed before consulting held-out labels.
    with np.load('results/day1/data.npz') as original:
        assert all(np.array_equal(data[k], original[k]) for k in data)
    model = QCLModel(config['layers'], config['parameter_seed'])
    model.load_state_dict(checkpoint['model_state_dict'])
    coefficients, _, rank, singular = np.linalg.lstsq(
        polynomial_features(data['train_t'], config['polynomial_degree']), data['train_y'], rcond=None)
    assert rank == config['polynomial_degree'] + 1
    np.savez(output / 'classical_model.npz', coefficients=coefficients, singular_values=singular)
    predictions = {}
    scores = {name: {} for name in ('qcl', 'polynomial', 'analytic_reference')}
    for split in ('train', 'validation', 'test', 'grid'):
        inputs = data[split + '_t']
        with torch.no_grad():
            q_predictions = model(torch.tensor(inputs, dtype=torch.float64)).numpy()
        values = {'qcl': q_predictions,
                  'polynomial': polynomial_features(inputs, config['polynomial_degree']) @ coefficients,
                  'analytic_reference': analytic_reference(inputs)}
        for name, prediction in values.items():
            predictions[name + '_' + split] = prediction
            if split != 'grid':
                scores[name][split] = metrics(prediction, data[split + '_y'])
    # Independent checkpoint consistency check using per-output metrics.
    assert abs(sum(v['mse'] for v in scores['qcl']['validation'].values()) - checkpoint['validation_loss']) < 1e-12
    assert abs(sum(v['mse'] for v in scores['qcl']['train'].values()) - checkpoint['train_loss']) < 1e-12
    np.savez(output / 'predictions.npz', **data, **predictions)
    for j, name in enumerate(('sin', 'cos')):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(data['grid_t'], data['grid_y'][:, j], color='black', linewidth=2, label='Target')
        ax.plot(data['grid_t'], predictions['qcl_grid'][:, j], label='Learned QCL (1 layer)')
        ax.plot(data['grid_t'], predictions['polynomial_grid'][:, j], linestyle='--', label='Learned polynomial (degree 5)')
        ax.plot(data['grid_t'], predictions['analytic_reference_grid'][:, j], linestyle=':', label='Analytic quantum reference (no training)')
        ax.scatter(data['test_t'], predictions['qcl_test'][:, j], s=18, label='QCL test predictions')
        ax.set(xlabel='t', ylabel='Output', title=f'{name}(pi t): frozen-checkpoint predictions')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / f'{name}_fit.png', dpi=150)
        plt.close(fig)
    report = {'config': config, 'selected_step': best_step, 'selection': 'minimum validation loss; earliest tie',
              'metrics': scores, 'qcl_parameters': model.weights.numel(),
              'polynomial_coefficients': int(coefficients.size),
              'analytic_trainable_parameters': 0, 'qpu_jobs': 0,
              'checkpoint_sha256': hashlib.sha256((run / 'best.pt').read_bytes()).hexdigest(),
              'versions': {p: importlib.metadata.version(p) for p in ('pennylane', 'torch', 'numpy', 'matplotlib')},
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'test_used_for_selection': False}
    (output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    with (output / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'split', 'output', 'mse', 'mae', 'max_absolute_error'])
        for model_name, splits in scores.items():
            for split, outputs in splits.items():
                for name, values in outputs.items():
                    writer.writerow([model_name, split, name, values['mse'], values['mae'], values['max_absolute_error']])
    lines = ['# Day 3 review', '', f'Frozen quantum checkpoint selected at step {best_step} by validation loss.', '',
             '| Model | Output | Test MSE | Test MAE | Maximum test error |',
             '|---|---|---:|---:|---:|']
    for model_name in scores:
        for name, values in scores[model_name]['test'].items():
            lines.append(f'| {model_name} | {name} | {values["mse"]:.8g} | {values["mae"]:.8g} | {values["max_absolute_error"]:.8g} |')
    lines += ['', 'The polynomial has 12 fitted coefficients; QCL has 6 nominal trainable angles. '
              'The analytic reference directly encodes the target rotations and has zero fitted parameters. '
              'It is not a learned baseline. All three use local exact simulation or classical computation; no QPU.', '',
              'The one-layer circuit has restricted expressivity. Final Z rotations on both wires commute '
              'with the measured Z observables (including through CNOT), so two of its six angles do not '
              'change these readouts. This baseline is preserved as declared; Day 4 will study data re-uploading '
              'and deeper circuits using validation for selection. No architecture was changed based on these test scores.', '',
              'Only one initialization seed and one dataset were evaluated here. Results establish neither '
              'quantum advantage nor reliable performance across initializations. The polynomial degree was fixed '
              'before evaluation. Test scores are now exposed for reporting and must not guide later selection.', '',
              'Inspect sin_fit.png, cos_fit.png, metrics.csv, and the training loss plot in ../baseline/loss.png. '
              'The predictions archive contains all targets and predictions so metrics can be checked independently.']
    (output / 'REVIEW.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('results/day3/baseline'))
    parser.add_argument('--output', type=Path, default=Path('results/day3/evaluation'))
    args = parser.parse_args()
    evaluate(args.run, args.output)
