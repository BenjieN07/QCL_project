"""Audit saved Baseline predictions, metrics, and checkpoint selection without training."""
from qcl.checkpoints import load_checkpoint
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .data import make_data
from .evaluate import polynomial_features, analytic_reference
from .model import QCLModel


def verify(run, evaluation):
    report = json.loads((evaluation / 'metrics.json').read_text())
    config = report['config']
    summary = json.loads((run / 'summary.json').read_text())
    assert config == summary['config']
    assert report['checkpoint_sha256'] == hashlib.sha256((run / 'best.pt').read_bytes()).hexdigest()
    checkpoint = load_checkpoint(run / 'best.pt', weights_only=True)
    assert checkpoint['config'] == config
    with (run / 'history.csv').open() as stream:
        history = list(csv.DictReader(stream))
    assert [int(row['step']) for row in history] == list(range(config['steps'] + 1))
    assert all(np.isfinite(float(row['gradient_norm'])) and float(row['gradient_norm']) > 1e-12
               for row in history[:-1])
    losses = np.array([[float(row[key]) for key in ('train_loss', 'validation_loss')] for row in history])
    assert np.isfinite(losses).all()
    selected = int(np.argmin(losses[:, 1]))
    assert selected == checkpoint['step'] == report['selected_step'] == summary['best_step']
    assert abs(losses[selected, 0] - checkpoint['train_loss']) < 1e-12
    assert abs(losses[selected, 1] - checkpoint['validation_loss']) < 1e-12
    model = QCLModel(config['layers'], config['parameter_seed'])
    model.load_state_dict(checkpoint['model_state_dict'])
    expected_data = make_data(config)
    coefficients = np.load(evaluation / 'classical_model.npz')['coefficients']
    fitted = np.linalg.lstsq(polynomial_features(expected_data['train_t'], config['polynomial_degree']),
                            expected_data['train_y'], rcond=None)[0]
    np.testing.assert_allclose(coefficients, fitted, atol=1e-13, rtol=0)
    checked_predictions = 0
    with np.load(evaluation / 'predictions.npz') as archive:
        for key, expected in expected_data.items():
            np.testing.assert_array_equal(archive[key], expected)
        for split in ('train', 'validation', 'test', 'grid'):
            inputs = archive[split + '_t']
            with torch.no_grad():
                quantum = model(torch.tensor(inputs, dtype=torch.float64)).numpy()
            fresh = {'qcl': quantum, 'polynomial': polynomial_features(inputs, config['polynomial_degree']) @ coefficients,
                     'analytic_reference': analytic_reference(inputs)}
            for name, prediction in fresh.items():
                np.testing.assert_allclose(prediction, archive[name + '_' + split], atol=1e-13, rtol=0)
                checked_predictions += prediction.size
                if split != 'grid':
                    absolute = np.abs(archive[name + '_' + split] - archive[split + '_y'])
                    for index, output in enumerate(('sin', 'cos')):
                        expected = (np.mean(absolute[:, index] ** 2), np.mean(absolute[:, index]), np.max(absolute[:, index]))
                        saved = report['metrics'][name][split][output]
                        np.testing.assert_allclose([saved[k] for k in ('mse', 'mae', 'max_absolute_error')], expected,
                                                   atol=1e-13, rtol=0)
        for split, column in (('train', 0), ('validation', 1)):
            mse_sum = sum(v['mse'] for v in report['metrics']['qcl'][split].values())
            assert abs(mse_sum - losses[selected, column]) < 1e-12
    with (evaluation / 'metrics.csv').open() as stream:
        metric_rows = list(csv.DictReader(stream))
    assert len(metric_rows) == 18
    for row in metric_rows:
        for key in ('mse', 'mae', 'max_absolute_error'):
            assert float(row[key]) == report['metrics'][row['model']][row['split']][row['output']][key]
    result = {'status': 'PASS', 'optimizer_updates': config['steps'], 'history_rows': len(history),
              'prediction_values_recomputed': checked_predictions, 'metric_rows_verified': len(metric_rows),
              'checkpoint_selection_verified': True, 'classical_fit_verified': True,
              'split_and_targets_verified': True, 'test_used_for_selection': False, 'qpu_jobs': 0}
    (evaluation / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('results/baseline/run'))
    parser.add_argument('--evaluation', type=Path, default=Path('results/baseline/evaluation'))
    args = parser.parse_args()
    verify(args.run, args.evaluation)
