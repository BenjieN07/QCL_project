"""Independently audit a saved smoke run and optionally compare a repeat."""
from qcl.checkpoints import load_checkpoint
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .data import make_data
from .model import QCLModel
from .training import regression_loss


def audit(run, compare=None):
    summary = json.loads((run / 'summary.json').read_text())
    config = json.loads((run / 'config.json').read_text())
    assert config == summary['config'], 'Saved configurations differ'
    with (run / 'history.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(r['step']) for r in rows] == list(range(config['smoke_steps'] + 1))
    losses = np.array([[float(r[k]) for k in ('train_loss', 'validation_loss')] for r in rows])
    gradients = np.array([float(r['gradient_norm']) for r in rows[:-1]])
    assert np.isfinite(losses).all() and (losses >= 0).all()
    assert np.isfinite(gradients).all() and (gradients > 1e-12).all()
    best_step = int(np.argmin(losses[:, 1]))
    assert best_step == summary['best_step']
    checkpoint = load_checkpoint(run / 'best.pt', weights_only=True)
    assert checkpoint['step'] == best_step and checkpoint['config'] == config
    model = QCLModel(config['layers'], config['parameter_seed'])
    model.load_state_dict(checkpoint['model_state_dict'])
    data = make_data(config)
    reloaded = []
    with torch.no_grad():
        for split in ('train', 'validation'):
            x = torch.tensor(data[split + '_t'], dtype=torch.float64)
            y = torch.tensor(data[split + '_y'], dtype=torch.float64)
            reloaded.append(regression_loss(model(x), y).item())
    assert np.allclose(reloaded, losses[best_step], atol=1e-12, rtol=0)
    assert abs(summary['initial_train_loss'] - losses[0, 0]) < 1e-12
    assert abs(summary['final_train_loss'] - losses[-1, 0]) < 1e-12
    assert abs(summary['best_validation_loss'] - losses[best_step, 1]) < 1e-12
    repeated = None
    if compare is not None:
        other_summary = json.loads((compare / 'summary.json').read_text())
        assert config == other_summary['config'], 'Repeat configuration differs'
        with (compare / 'history.csv').open() as stream:
            other_rows = list(csv.DictReader(stream))
        assert rows == other_rows, 'Repeated losses or gradients differ'
        other = load_checkpoint(compare / 'best.pt', weights_only=True)
        assert all(torch.equal(v, other['model_state_dict'][k]) for k, v in checkpoint['model_state_dict'].items())
        repeated = True
    report = {'status': 'PASS', 'history_rows': len(rows), 'checkpoint_losses_match': True,
              'validation_selection_matches': True, 'repeat_history_and_weights_identical': repeated,
              'checkpoint_sha256': hashlib.sha256((run / 'best.pt').read_bytes()).hexdigest(),
              'test_labels_used': False, 'qpu_jobs': 0}
    (run / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    note = (
        '# Training smoke test review\n\nTraining smoke test and checkpoint audit: PASS.\n\n'
        f'- Adam updates: {config["smoke_steps"]}; learning rate: {config["learning_rate"]}; '
        f'parameter seed: {config["parameter_seed"]}.\n'
        f'- Initial training loss: {losses[0, 0]:.8f}.\n'
        f'- Final training loss: {losses[-1, 0]:.8f}.\n'
        f'- Training improvement: {100 * (1 - losses[-1, 0] / losses[0, 0]):.2f}%.\n'
        f'- Selected step: {best_step}; validation loss: {losses[best_step, 1]:.8f}.\n'
        f'- Minimum gradient norm: {gradients.min():.8f}.\n'
        f'- Independent repeat identical: {repeated}.\n\n'
        'Loss is the sample mean of the SUM of squared errors across outputs. '
        'These losses are not per-output MSE or held-out test scores.\n\n'
        'Inspect loss.png and history.csv. The selected checkpoint is best.pt; '
        'summary.json records seeds, settings, versions, source revision and hashes. '
        'audit.json verifies the history, validation selection, and reloaded checkpoint.\n\n'
        'This short run establishes gradient flow and loss reduction, not an accurate sine/cosine fit. '
        'No test labels or QPU jobs were used. Stop here for inspection before Baseline.\n')
    (run / 'REVIEW.md').write_text(note)
    print(note)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('results/smoke/run'))
    parser.add_argument('--compare', type=Path)
    args = parser.parse_args()
    audit(args.run, args.compare)
