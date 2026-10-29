"""Run python view_model.py to view the saved QCL model against exact NumPy targets."""
from qcl.checkpoints import load_checkpoint
import argparse
import hashlib
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
    return ROOT / 'results/baseline/run'


def saved_hardware(checkpoint_path):
    folder = ROOT / 'results/hardware/real_pilot'
    if not (folder / 'results.json').is_file() or not (folder / 'job.json').is_file():
        return None
    receipt = json.loads((folder / 'job.json').read_text())
    if receipt['checkpoint_sha256'] != hashlib.sha256(checkpoint_path.read_bytes()).hexdigest():
        print('Saved hardware measurements belong to another checkpoint; overlay omitted.')
        return None
    result = json.loads((folder / 'results.json').read_text())
    if result['status'] != 'DONE' or result['job_id'] != receipt['job_id']:
        raise ValueError('Hardware results do not match the completed job receipt')
    return result


def create_view(run, output, show_window=True):
    checkpoint_path = run / 'best.pt'
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'No saved checkpoint at {checkpoint_path}. Choose a trained run with --run.')
    checkpoint = load_checkpoint(checkpoint_path, weights_only=True, map_location='cpu')
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
    hardware = saved_hardware(checkpoint_path)
    if hardware is not None:
        hardware_t = np.array([row['t'] for row in hardware['rows']])
        hardware_predictions = np.array([row['hardware_prediction'] for row in hardware['rows']])
        hardware_se = np.array([row['measurement_standard_error'] for row in hardware['rows']])
        hardware_targets = np.column_stack((np.sin(np.pi * hardware_t), np.cos(np.pi * hardware_t)))
    assert np.isfinite(predicted).all()
    selected_step = checkpoint.get('step', checkpoint.get('iteration'))
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex='col')
    for j, name in enumerate(('sin', 'cos')):
        ax = axes[0, j]
        ax.plot(grid, exact[:, j], color='black', linewidth=2.5, linestyle='--', label='NumPy exact target')
        ax.plot(grid, predicted[:, j], color='#1668bb', linewidth=2, label='Saved QCL approximation')
        ax.scatter(data['test_t'], test_predictions[:, j], color='#1668bb', s=20, label='QCL test predictions')
        if hardware is not None:
            ax.errorbar(hardware_t, hardware_predictions[:, j], yerr=hardware_se[:, j],
                        fmt='o', color='#e87912', capsize=4, zorder=5,
                        label=f'Real {hardware["backend"]}: {len(hardware_t)} inputs ± 1 sampling SE')
        ax.set(title=f'{name}(πt) | test MSE {scores[name]["mse"]:.5f}', ylabel='Function value',
               ylim=(min(-1.15, float(predicted[:, j].min())-.05), max(1.15, float(predicted[:, j].max())+.05)))
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)
        error_ax = axes[1, j]
        error_ax.plot(grid, predicted[:, j] - exact[:, j], color='#b54a16', label='Simulator − target')
        if hardware is not None:
            error_ax.errorbar(hardware_t, hardware_predictions[:, j] - hardware_targets[:, j],
                             yerr=hardware_se[:, j], fmt='o', color='#e87912', capsize=4,
                             label='Hardware − target')
            error_ax.legend(fontsize=7)
        error_ax.axhline(0, color='gray', linewidth=1)
        error_ax.set(xlabel='t', ylabel='QCL − exact', xlim=(-1.05, 1.05) if hardware is not None else (-1, 1))
        error_ax.grid(alpha=0.2)
    fig.suptitle(f'Saved quantum model: {config.get("qubits", 2)} qubits, {config["layers"]} layers, input scale {config.get("input_scale", 1.0)}, selected update {selected_step}')
    if hardware is not None:
        fig.text(.5, .012, f'Hardware: {len(hardware_t)} measured inputs using frozen weights; no hardware training. Error bars show sampling uncertainty only.',
                 ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .035 if hardware is not None else 0, 1, 1))
    figure_path = output / 'model_comparison.png'
    fig.savefig(figure_path, dpi=150)
    (output / 'metrics.json').write_text(json.dumps({'checkpoint_step': selected_step, 'config': config,
                                                  'test_metrics': scores, 'training_performed': False,
                                                  'hardware': hardware}, indent=2) + '\n')
    print(f'Graph: {figure_path}\nTest set: {len(data["test_t"])} inputs; no training performed.')
    print(json.dumps(scores, indent=2))
    if hardware is not None:
        print(f'Showing saved real hardware: {hardware["backend"]}, job {hardware["job_id"]}, {len(hardware_t)} inputs. No new job submitted.')
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
