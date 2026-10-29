"""Compare initialization, saved training checkpoints, and exact targets without retraining."""
import argparse
import csv
import json
from pathlib import Path
import webbrowser

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from qcl.data import make_data
from qcl.evaluate import metrics
from qcl.model import QCLModel

ROOT = Path(__file__).resolve().parent


def build_progress(output, open_browser=True):
    baseline = ROOT / 'results/day3/baseline'
    config = json.loads((baseline / 'config.json').read_text())
    summary = json.loads((baseline / 'summary.json').read_text())
    data = make_data(config)
    grid = np.linspace(-1, 1, 401)
    exact = np.column_stack((np.sin(np.pi * grid), np.cos(np.pi * grid)))
    models = {'Initial': QCLModel(config['layers'], config['parameter_seed'])}
    for label, path in (('30-step run', ROOT / 'results/day2/smoke/best.pt'),
                        ('250-step run', baseline / 'best.pt')):
        checkpoint = torch.load(path, weights_only=True, map_location='cpu')
        model = QCLModel(checkpoint['config']['layers'], checkpoint['config']['parameter_seed'])
        model.load_state_dict(checkpoint['model_state_dict'])
        models[label] = model
    grid_predictions, test_scores = {}, {}
    with torch.no_grad():
        for label, model in models.items():
            grid_predictions[label] = model(torch.tensor(grid, dtype=torch.float64)).numpy()
            predictions = model(torch.tensor(data['test_t'], dtype=torch.float64)).numpy()
            test_scores[label] = metrics(predictions, data['test_y'])
        initial_train = models['Initial'](torch.tensor(data['train_t'], dtype=torch.float64)).numpy()
    initial_loss = np.mean(np.sum((initial_train - data['train_y']) ** 2, axis=1))
    assert abs(initial_loss - summary['initial_train_loss']) < 1e-12
    saved = json.loads((ROOT / 'results/day3/evaluation/metrics.json').read_text())
    assert test_scores['250-step run'] == saved['metrics']['qcl']['test']
    test_scores['Polynomial'] = saved['metrics']['polynomial']['test']
    with (baseline / 'history.csv').open() as stream:
        history = list(csv.DictReader(stream))
    runs = [ROOT / 'results/day2/smoke', ROOT / 'results/day2/repeat', baseline]
    run_updates = []
    for run in runs:
        run_config = json.loads((run / 'config.json').read_text())
        run_updates.append(run_config.get('steps', run_config.get('smoke_steps')))
    total_updates = sum(run_updates)
    assert total_updates == 310
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for index, name in enumerate(('sin', 'cos')):
        ax = axes[0, index]
        ax.plot(grid, exact[:, index], color='black', linewidth=2, label='Exact NumPy target')
        ax.plot(grid, grid_predictions['Initial'][:, index], color='#888888', linestyle='--', label='Beginning: untrained')
        ax.plot(grid, grid_predictions['250-step run'][:, index], color='#1668bb', linewidth=2, label='Day 3: selected step 83')
        ax.set(title=f'{name}(πt): beginning versus Day 3 baseline', xlabel='t', ylabel='Function value', ylim=(-1.1, 1.1))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9, loc='lower center')
    loss_ax = axes[1, 0]
    steps = [int(row['step']) for row in history]
    loss_ax.plot(steps, [float(row['train_loss']) for row in history], label='Training', color='#1668bb')
    loss_ax.plot(steps, [float(row['validation_loss']) for row in history], label='Validation', color='#bd6513')
    loss_ax.axvline(summary['best_step'], color='gray', linestyle=':', label='Selected step 83')
    loss_ax.set(title='Training: rapid improvement, then plateau', xlabel='Optimizer updates completed',
                ylabel='Mean sum of squared errors', ylim=(0, 2.05))
    loss_ax.grid(alpha=0.2)
    loss_ax.legend(fontsize=9)
    error_ax = axes[1, 1]
    positions = np.arange(len(test_scores))
    for index, (name, color) in enumerate((('sin', '#1668bb'), ('cos', '#bd6513'))):
        values = [scores[name]['mse'] for scores in test_scores.values()]
        bars = error_ax.bar(positions + (index - 0.5) * 0.34, values, width=0.34, label=name, color=color)
        for bar, value in zip(bars, values):
            error_ax.text(bar.get_x() + bar.get_width()/2, value * 1.18, f'{value:.2g}',
                          ha='center', va='bottom', fontsize=9)
    error_ax.set_yscale('log')
    error_ax.set(xticks=positions, xticklabels=list(test_scores), ylabel='Test MSE (log scale; lower is better)',
                 title='Saved-checkpoint test error', ylim=(1e-5, 3))
    error_ax.legend(fontsize=9)
    error_ax.grid(axis='y', alpha=0.2)
    fig.suptitle('Historical Day 3 progress: 250-update baseline; selected step 83', fontsize=15)
    fig.text(0.5, 0.015, f'{total_updates} recorded experiment updates = 30 + 30 repeat + 250 | '
             '48 training inputs | 1 layer, 2 qubits, 6 nominal angles | No hardware runs', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    image_path = output / 'progress.png'
    fig.savefig(image_path, dpi=150)
    plt.close(fig)
    report = {'recorded_experiment_updates': total_updates, 'run_updates': run_updates,
              'training_sample_presentations': total_updates * config['train_size'],
              'unit_test_optimizations_included': False,
              'baseline_final_step': config['steps'], 'selected_step': summary['best_step'],
              'initial_training_loss': summary['initial_train_loss'],
              'final_training_loss': summary['final_train_loss'],
              'relative_training_loss_reduction': summary['relative_train_improvement'],
              'test_metrics': test_scores, 'new_training_performed': False,
              'initial_loss_matches_saved_step_zero': True,
              'initial_reconstruction': 'same architecture, parameter seed 7, float64 initialization'}
    (output / 'progress.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez(output / 'curves.npz', inputs=grid, exact=exact,
             initial=grid_predictions['Initial'], selected=grid_predictions['250-step run'])
    print(f'Progress figure: {image_path}\nRecorded experiment updates: {total_updates}')
    print('Saved current-model test metrics exactly match the verified Day 3 report.')
    if open_browser and not webbrowser.open(image_path.resolve().as_uri()):
        print('Open the progress figure above to inspect it.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-open', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/progress')
    args = parser.parse_args()
    build_progress(args.output.resolve(), not args.no_open)
