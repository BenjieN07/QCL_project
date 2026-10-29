"""Evaluate frozen pilot models on a new holdout, or verify saved results without training."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from .data import make_data, targets
from .evaluate import metrics, polynomial_features
from .paper_model import build_model

BASE = Path('results/paper_pilot')
RUNS = {'Original Adam': Path('results/day3/baseline'),
        'Original BFGS': BASE / 'original_bfgs', 'Paper-inspired BFGS': BASE / 'ising_bfgs'}


def calculate():
    registry = json.loads(Path('results/current_model.json').read_text())
    summaries = {label: json.loads((run / 'summary.json').read_text()) for label, run in RUNS.items()}
    selected_label = min(summaries, key=lambda name: summaries[name].get('selected_validation_loss', summaries[name].get('best_validation_loss')))
    assert registry['run'] == str(RUNS[selected_label]).replace('\\', '/')
    assert registry['test_or_fresh_holdout_used_for_selection'] is False
    config = summaries['Paper-inspired BFGS']['config']
    data = make_data(config)
    # No fresh input or target is generated until both checkpoints and validation-only registry exist.
    assert all((run / 'best.pt').is_file() for run in RUNS.values())
    inputs = np.random.default_rng(config['fresh_holdout_seed']).uniform(-1, 1, config['fresh_holdout_size'])
    inputs.sort()
    original_inputs = np.concatenate([data[name + '_t'] for name in ('train', 'validation', 'test')])
    assert len(np.unique(inputs)) == len(inputs) and not np.isin(inputs, original_inputs).any()
    data['fresh_t'], data['fresh_y'] = inputs, targets(inputs)
    data['grid_t'] = np.linspace(-1, 1, 401)
    data['grid_y'] = targets(data['grid_t'])
    archive = dict(data)
    scores, checkpoints = {}, {}
    for label, run in RUNS.items():
        checkpoint = torch.load(run / 'best.pt', weights_only=True)
        checkpoints[label] = checkpoint
        assert checkpoint['config'] == summaries[label]['config']
        same_data = make_data(checkpoint['config'])
        for split in ('train', 'validation', 'test'):
            np.testing.assert_array_equal(same_data[split + '_t'], data[split + '_t'])
            np.testing.assert_array_equal(same_data[split + '_y'], data[split + '_y'])
        with (run / 'history.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        assert int(np.argmin([float(row['validation_loss']) for row in rows])) == checkpoint.get('step', checkpoint.get('iteration'))
        model = build_model(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        scores[label] = {}
        with torch.no_grad():
            for split in ('train', 'validation', 'test', 'fresh', 'grid'):
                prediction = model(torch.tensor(data[split + '_t'], dtype=torch.float64)).numpy()
                archive[label + '_' + split] = prediction
                scores[label][split] = metrics(prediction, data[split + '_y'])
        assert abs(sum(v['mse'] for v in scores[label]['validation'].values()) - checkpoint['validation_loss']) < 1e-12
        assert abs(sum(v['mse'] for v in scores[label]['train'].values()) - checkpoint['train_loss']) < 1e-12
    coefficients = np.load('results/day3/evaluation/classical_model.npz')['coefficients']
    fitted = np.linalg.lstsq(polynomial_features(data['train_t'], 5), data['train_y'], rcond=None)[0]
    np.testing.assert_allclose(coefficients, fitted, atol=1e-13, rtol=0)
    scores['Polynomial'] = {}
    for split in ('train', 'validation', 'test', 'fresh', 'grid'):
        prediction = polynomial_features(data[split + '_t'], 5) @ coefficients
        archive['Polynomial_' + split] = prediction
        scores['Polynomial'][split] = metrics(prediction, data[split + '_y'])
    return scores, archive, summaries, registry


def main(verify=False):
    torch.set_num_threads(1)
    output = BASE / 'evaluation'
    if not verify and output.exists():
        raise FileExistsError('Evaluation already exists; use --verify to audit without overwriting')
    scores, archive, summaries, registry = calculate()
    if verify:
        report = json.loads((output / 'metrics.json').read_text())
        for label, splits in scores.items():
            for split, outputs in splits.items():
                for name, values in outputs.items():
                    for key, value in values.items():
                        assert abs(value - report['metrics'][label][split][name][key]) < 1e-13
        with np.load(output / 'predictions.npz') as saved:
            for key, value in archive.items():
                np.testing.assert_allclose(saved[key], value, atol=1e-13, rtol=0)
        assert report['checkpoint_sha256'] == {label: hashlib.sha256((run / 'best.pt').read_bytes()).hexdigest() for label, run in RUNS.items()}
        with (output / 'metrics.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 40
        for row in rows:
            for key in ('mse', 'mae', 'max_absolute_error'):
                assert abs(float(row[key]) - scores[row['model']][row['split']][row['output']][key]) < 1e-13
        audit = {'status': 'PASS', 'prediction_values_recomputed': sum(archive[label + '_' + split].size for label in scores for split in ('train', 'validation', 'test', 'fresh', 'grid')),
                 'scalar_metrics_verified': len(scores) * 5 * 2 * 3, 'csv_metric_rows_verified': len(rows),
                 'validation_selection_verified': True,
                 'fresh_holdout_disjoint': True, 'checkpoint_hashes_verified': True, 'qpu_jobs': 0}
        (output / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
        print(json.dumps(audit, indent=2))
        return
    output.mkdir(parents=True)
    np.savez(output / 'predictions.npz', **archive)
    paper = scores['Paper-inspired BFGS']
    bounds = {split: {'min': float(archive['Paper-inspired BFGS_' + split].min()),
                      'max': float(archive['Paper-inspired BFGS_' + split].max())}
              for split in ('fresh', 'grid')}
    report = {'metrics': scores, 'current_model': registry, 'fresh_holdout_seed': 20261029,
              'fresh_holdout_size': 128, 'fresh_holdout_used_for_selection': False,
              'scaled_prediction_range': bounds,
              'checkpoint_sha256': {label: hashlib.sha256((run / 'best.pt').read_bytes()).hexdigest() for label, run in RUNS.items()},
              'paper_validation_target_met': summaries['Paper-inspired BFGS']['validation_accuracy_target_met'],
              'paper_solver_success': summaries['Paper-inspired BFGS']['solver_success'],
              'qpu_jobs': 0}
    (output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    with (output / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'split', 'output', 'mse', 'mae', 'max_absolute_error'])
        for label, splits in scores.items():
            for split, outputs in splits.items():
                for name, values in outputs.items():
                    writer.writerow([label, split, name, values['mse'], values['mae'], values['max_absolute_error']])
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    colors = {'Original Adam': '#888888', 'Original BFGS': '#bd6513', 'Paper-inspired BFGS': '#1668bb'}
    for j, name in enumerate(('sin', 'cos')):
        ax = axes[0, j]
        ax.plot(archive['grid_t'], archive['grid_y'][:, j], color='black', linestyle='--', linewidth=2.5, label='Exact NumPy target')
        for label in RUNS:
            ax.plot(archive['grid_t'], archive[label + '_grid'][:, j], color=colors[label], label=label)
        ax.set(xlabel='t', ylabel='Prediction', title=f'{name}(πt): frozen selected checkpoints')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    for j, key in enumerate(('train_loss', 'validation_loss')):
        ax = axes[1, j]
        for label, run in RUNS.items():
            with (run / 'history.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            ax.plot(np.arange(len(rows)), [float(row[key]) for row in rows], color=colors[label], label=label)
        ax.set_yscale('log')
        ax.set(xlabel='Accepted optimizer updates (not equal compute cost)', ylabel='Joint squared error',
               title=key.replace('_', ' ').capitalize())
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Paper-inspired pilot: better model fit; optimizer-only change was insufficient')
    fig.tight_layout()
    fig.savefig(output / 'comparison.png', dpi=150)
    plt.close(fig)
    lines = ['# Paper-inspired pilot results', '', 'Fresh holdout: 128 new inputs, seed 20261029. Checkpoints frozen and registered before evaluation.', '',
             '| Model | Output | Fresh MSE | Fresh MAE | Fresh maximum error |', '|---|---|---:|---:|---:|']
    for label, splits in scores.items():
        for name, values in splits['fresh'].items():
            lines.append(f'| {label} | {name} | {values["mse"]:.9g} | {values["mae"]:.9g} | {values["max_absolute_error"]:.9g} |')
    lines += ['', 'BFGS alone reached the original circuit’s Adam training-loss floor. The six-qubit/six-block '
              'Ising model with learned scales met the declared validation target. This supports the new bundled '
              'design, not a causal claim about the entangler alone.', '',
              'The larger model ran 250 accepted BFGS updates and 263 objective evaluations, selecting iterate 247. '
              'It stopped at its iteration budget (solver success=False), not at the convergence tolerance. '
              'The original BFGS run converged after 24 accepted updates and 38 objective evaluations. '
              'All gradients were analytic simulator gradients. The new implementation batches inputs, so timings '
              'are not a controlled optimizer or circuit speed comparison.', '',
              f'Scaled prediction ranges without clipping: {bounds}. The 401-point grid includes both endpoints; '
              'grid maximum errors are available in metrics.json. Random-holdout accuracy does not guarantee '
              'uniform accuracy at every point.', '',
              f'Important boundary limitation: maximum absolute error on the endpoint-inclusive grid is '
              f'{paper["grid"]["sin"]["max_absolute_error"]:.6f} for sine and '
              f'{paper["grid"]["cos"]["max_absolute_error"]:.6f} for cosine. '
              'The model is much better on average but not yet uniformly accurate over the closed interval. '
              'Endpoint and near-endpoint training coverage should be studied in a separately declared experiment; '
              'do not tune against this now-exposed fresh holdout.', '',
              'Only one initialization and Hamiltonian seed were studied. Multiple-seed robustness, controlled '
              'entangler ablation, the earlier matched-depth study, finite shots, and hardware validation remain '
              'uncompleted. The new holdout is now exposed and must not guide later tuning.', '',
              'This is a two-output, noiseless extension inspired by the paper, not an exact reproduction. '
              'No hardware jobs, quantum advantage, or accuracy percentage is claimed.']
    (output / 'REVIEW.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    main(parser.parse_args().verify)
