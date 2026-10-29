"""Run python view_model.py to view the saved QCL model against exact NumPy targets."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from qcl.data import make_data
from qcl.evaluate import metrics
from qcl.model import QCLModel
from qcl.display import show_figure

ROOT = Path(__file__).resolve().parent


def current_run():
    registry = ROOT / 'results/current_model.json'
    if registry.is_file():
        return ROOT / json.loads(registry.read_text())['run']
    return ROOT / 'results/day3/baseline'


def create_view(run, output, show_window=True):
    checkpoint_path = run / 'best.pt'
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'No saved checkpoint at {checkpoint_path}. Choose a trained run with --run.')
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
    config = checkpoint['config']
    if config.get('model', 'original') != 'original':
        raise ValueError('This viewer supports only the original QCL model family.')
    model = QCLModel(config['layers'], config['parameter_seed'], config.get('input_scale', 1.0))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    data = make_data(config)
    grid = np.linspace(-1, 1, 401)
    # These are the mathematical targets, not a fitted classical model.
    exact = np.column_stack((np.sin(np.pi * grid), np.cos(np.pi * grid)))
    with torch.no_grad():
        predicted = model(torch.tensor(grid, dtype=torch.float64)).numpy()
        test_predictions = model(torch.tensor(data['test_t'], dtype=torch.float64)).numpy()
    scores = metrics(test_predictions, data['test_y'])
    assert np.isfinite(predicted).all()
    selected_step = checkpoint.get('step', checkpoint.get('iteration'))
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex='col')
    for j, name in enumerate(('sin', 'cos')):
        ax = axes[0, j]
        ax.plot(grid, exact[:, j], color='black', linewidth=2.5, linestyle='--', label='NumPy exact target')
        ax.plot(grid, predicted[:, j], color='#1668bb', linewidth=2, label='Saved QCL approximation')
        ax.scatter(data['test_t'], test_predictions[:, j], color='#1668bb', s=20, label='QCL test predictions')
        ax.set(title=f'{name}(πt) | test MSE {scores[name]["mse"]:.5f}', ylabel='Function value',
               ylim=(min(-1.15, float(predicted[:, j].min())-.05), max(1.15, float(predicted[:, j].max())+.05)))
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)
        error_ax = axes[1, j]
        error_ax.plot(grid, predicted[:, j] - exact[:, j], color='#b54a16')
        error_ax.axhline(0, color='gray', linewidth=1)
        error_ax.set(xlabel='t', ylabel='QCL − exact', xlim=(-1, 1))
        error_ax.grid(alpha=0.2)
    fig.suptitle(f'Saved quantum model: {config.get("qubits", 2)} qubits, {config["layers"]} layers, input scale {config.get("input_scale", 1.0)}, selected update {selected_step}')
    fig.tight_layout()
    figure_path = output / 'model_comparison.png'
    fig.savefig(figure_path, dpi=150)
    (output / 'metrics.json').write_text(json.dumps({'checkpoint_step': selected_step, 'config': config,
                                                  'test_metrics': scores, 'training_performed': False}, indent=2) + '\n')
    print(f'Graph: {figure_path}\nTest set: {len(data["test_t"])} inputs; no training performed.')
    print(json.dumps(scores, indent=2))
    if show_window:
        show_figure(fig, 'QCL model performance')
    plt.close(fig)
    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=current_run(), help='Directory containing best.pt; defaults to validation-selected current model')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/model_view', help='Viewer outputs; refreshed each run')
    parser.add_argument('--no-open', action='store_true', help='Save plots and metrics without opening a window')
    args = parser.parse_args()
    create_view(args.run.resolve(), args.output.resolve(), not args.no_open)
