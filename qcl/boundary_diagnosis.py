"""Measure coverage, endpoint errors, and encoding sensitivity without training."""
from qcl.checkpoints import load_checkpoint
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from .data import make_data, targets
from .model import QCLModel
from .training import regression_loss


def diagnose():
    spec = json.loads(Path('configs/boundary_study.json').read_text())
    checkpoint = load_checkpoint(Path(spec['source_run']) / 'best.pt', weights_only=True)
    config = checkpoint['config']
    model = QCLModel(config['layers'], config['parameter_seed'])
    model.load_state_dict(checkpoint['model_state_dict'])
    original_weights = model.weights.detach().clone()
    data = make_data(config)
    endpoints = torch.tensor([-1., 1.], dtype=torch.float64)
    endpoint_targets = torch.tensor(targets(endpoints.numpy()), dtype=torch.float64)
    predictions = model(endpoints)
    regression_loss(predictions, endpoint_targets).backward()
    gradients_finite = bool(torch.isfinite(model.weights.grad).all())
    assert gradients_finite
    assert torch.equal(model.weights.detach(), original_weights)
    sensitivity = []
    with torch.no_grad():
        for distance in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
            for side in (-1., 1.):
                inputs = torch.tensor([side * (1-distance), side], dtype=torch.float64)
                output = model(inputs).numpy()
                sensitivity.append({'side': side, 'distance': distance,
                                    'prediction_secant_magnitudes': (np.abs(output[1]-output[0])/distance).tolist(),
                                    'arcsin_slope_inside': float(1/np.sqrt(1-(1-distance)**2)),
                                    'arccos_squared_slope_magnitude_inside': float(2*(1-distance)/np.sqrt(1-(1-distance)**4))})
    report = {'source_run': spec['source_run'], 'training_min': float(data['train_t'].min()),
              'training_max': float(data['train_t'].max()),
              'training_inputs_in_boundary_collar': data['train_t'][np.abs(data['train_t']) >= .9].tolist(),
              'exact_endpoints_in_training': bool(np.isin([-1.,1.], data['train_t']).any()),
              'endpoint_predictions': predictions.detach().tolist(), 'endpoint_targets': endpoint_targets.tolist(),
              'endpoint_absolute_errors': (predictions.detach()-endpoint_targets).abs().tolist(),
              'endpoint_parameter_gradients_finite': gradients_finite,
              'endpoint_parameter_gradient_norm': model.weights.grad.norm().item(),
              'sensitivity': sensitivity, 'weights_unchanged': True, 'new_training_performed': False}
    output = Path('results/boundary/diagnosis')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'diagnosis.json').write_text(json.dumps(report, indent=2) + '\n')
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for row, bounds in enumerate(((-1., -.9), (.9, 1.))):
        grid = np.linspace(*bounds, 301)
        with torch.no_grad():
            values = model(torch.tensor(grid, dtype=torch.float64)).numpy()
        for column, name in enumerate(('sin', 'cos')):
            ax = axes[row, column]
            ax.plot(grid, targets(grid)[:, column], color='black', linestyle='--', label='Exact target')
            ax.plot(grid, values[:, column], color='#1668bb', label='Depth study model')
            mask = (data['train_t'] >= bounds[0]) & (data['train_t'] <= bounds[1])
            ax.scatter(data['train_t'][mask], data['train_y'][mask, column], color='#bd6513', label='Training targets', zorder=3)
            ax.set(xlim=bounds, xlabel='t', ylabel='Output', title=f'{name}: {bounds[0]} to {bounds[1]}')
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.suptitle('Boundary diagnosis: sparse training coverage and sharp input sensitivity')
    fig.tight_layout()
    fig.savefig(output / 'boundary_zoom.png', dpi=150)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    diagnose()
