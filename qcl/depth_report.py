"""Evaluate or audit the frozen original-circuit depth study; never train or select on tests."""
from qcl.checkpoints import load_checkpoint
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
from .model import QCLModel

ROOT = Path('results/depth_study')


def calculate():
    spec = json.loads(Path('configs/depth_study.json').read_text())
    selection = json.loads((ROOT / 'selection.json').read_text())
    records, archive, histories = [], {}, {}
    # Freeze all nine configurations/checkpoints and the validation-only selection before new holdout generation.
    for depth in spec['depths']:
        for seed in spec['parameter_seeds']:
            run = ROOT / 'runs' / f'depth{depth}_seed{seed}'
            assert (run / 'summary.json').is_file() and (run / 'best.pt').is_file()
    chosen = min(selection['candidates'], key=lambda row: (row['selected_validation_loss'], row['layers'], row['parameter_seed']))
    assert len(selection['candidates']) == 9
    assert selection['selected']['run'] == chosen['run']
    assert selection['selected']['test_or_fresh_holdout_used_for_selection'] is False
    data = make_data(spec)
    with np.load('results/validation/data.npz') as original:
        for key, value in data.items():
            np.testing.assert_array_equal(value, original[key])
    new_inputs = np.sort(np.random.default_rng(spec['new_holdout_seed']).uniform(-1, 1, spec['new_holdout_size']))
    exposed_inputs = np.random.default_rng(20261029).uniform(-1, 1, 128)
    original_inputs = np.concatenate([data[split + '_t'] for split in ('train', 'validation', 'test')])
    assert len(np.unique(new_inputs)) == len(new_inputs)
    assert not np.isin(new_inputs, np.concatenate((original_inputs, exposed_inputs))).any()
    data.update(new_t=new_inputs, new_y=targets(new_inputs), grid_t=np.linspace(-1, 1, 401))
    data['grid_y'] = targets(data['grid_t'])
    archive.update(data)
    for depth in spec['depths']:
        for seed in spec['parameter_seeds']:
            name = f'depth{depth}_seed{seed}'
            run = ROOT / 'runs' / name
            config = json.loads((run / 'config.json').read_text())
            summary = json.loads((run / 'summary.json').read_text())
            checkpoint = load_checkpoint(run / 'best.pt', weights_only=True)
            assert config == summary['config'] == checkpoint['config']
            assert config['layers'] == depth and config['parameter_seed'] == seed
            for key in ('steps', 'learning_rate', 'data_seed', 'train_size', 'validation_size', 'test_size', 'grid_size'):
                assert config[key] == spec[key]
            with (run / 'history.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            assert [int(row['step']) for row in rows] == list(range(spec['steps'] + 1))
            history = np.array([[float(row[key]) for key in ('train_loss', 'validation_loss')] for row in rows])
            gradient_norms = np.array([float(row['gradient_norm']) for row in rows[:-1]])
            assert np.isfinite(history).all() and np.isfinite(gradient_norms).all() and (gradient_norms > 1e-12).all()
            assert abs(history[0, 0] - summary['initial_train_loss']) < 1e-12
            assert abs(history[-1, 0] - summary['final_train_loss']) < 1e-12
            best_step = int(np.argmin(history[:, 1]))
            assert best_step == checkpoint['step'] == summary['best_step']
            histories[name] = history
            model = QCLModel(depth, seed)
            model.load_state_dict(checkpoint['model_state_dict'])
            scores = {}
            with torch.no_grad():
                for split in ('train', 'validation', 'test', 'new', 'grid'):
                    prediction = model(torch.tensor(data[split + '_t'], dtype=torch.float64)).numpy()
                    archive[name + '_' + split] = prediction
                    scores[split] = metrics(prediction, data[split + '_y'])
            for split, column in (('train', 0), ('validation', 1)):
                loss = sum(v['mse'] for v in scores[split].values())
                assert abs(loss - history[best_step, column]) < 1e-12
                assert abs(loss - checkpoint[split + '_loss']) < 1e-12
            validation_joint = sum(v['mse'] for v in scores['validation'].values())
            assert abs(validation_joint - summary['best_validation_loss']) < 1e-12
            candidate = next(row for row in selection['candidates'] if row['run'] == str(run).replace('\\', '/'))
            assert abs(candidate['selected_validation_loss'] - validation_joint) < 1e-12
            assert candidate['layers'] == depth and candidate['parameter_seed'] == seed and candidate['selected_step'] == best_step
            records.append({'name': name, 'run': str(run).replace('\\', '/'), 'depth': depth, 'seed': seed,
                            'selected_step': best_step, 'validation_joint_loss': validation_joint,
                            'parameters': model.weights.numel(), 'logical_gates': 7 * depth, 'logical_cnots': depth,
                            'loop_seconds': summary['elapsed_seconds'], 'initial_gradient_norm': gradient_norms[0],
                            'last_update_gradient_norm': gradient_norms[-1], 'minimum_gradient_norm': gradient_norms.min(),
                            'validation_target_met': all(v['mse'] <= .001 and v['mae'] <= .03 for v in scores['validation'].values()),
                            'metrics': scores, 'checkpoint_sha256': hashlib.sha256((run / 'best.pt').read_bytes()).hexdigest()})
    # Independently verify depth-1/seed-7 batching against the previously measured scalar trajectory.
    with Path('results/baseline/run/history.csv').open() as stream:
        original_rows = list(csv.DictReader(stream))
    scalar_history = np.array([[float(row[key]) for key in ('train_loss', 'validation_loss')] for row in original_rows])
    batching_error = float(np.max(np.abs(histories['depth1_seed7'] - scalar_history)))
    assert batching_error <= 1e-10
    coefficients = np.load('results/baseline/evaluation/classical_model.npz')['coefficients']
    polynomial = {split: metrics(polynomial_features(data[split + '_t'], 5) @ coefficients, data[split + '_y'])
                  for split in ('test', 'new', 'grid')}
    aggregates = []
    for depth in spec['depths']:
        group = [row for row in records if row['depth'] == depth]
        aggregate = {'depth': depth, 'seeds': len(group), 'parameters': 6 * depth, 'logical_gates': 7 * depth,
                     'logical_cnots': depth, 'metrics': {}}
        for label, values in (('validation_joint_loss', [r['validation_joint_loss'] for r in group]),
                              ('loop_seconds', [r['loop_seconds'] for r in group])):
            aggregate[label + '_mean'] = float(np.mean(values))
            aggregate[label + '_sample_std'] = float(np.std(values, ddof=1))
        for split in ('validation', 'test', 'new'):
            aggregate['metrics'][split] = {}
            for output in ('sin', 'cos'):
                aggregate['metrics'][split][output] = {}
                for metric in ('mse', 'mae', 'max_absolute_error'):
                    values = [r['metrics'][split][output][metric] for r in group]
                    aggregate['metrics'][split][output][metric] = {'mean': float(np.mean(values)), 'sample_std': float(np.std(values, ddof=1))}
        aggregates.append(aggregate)
    selected = next(row for row in records if row['run'] == chosen['run'])
    return {'spec': spec, 'selection': selection['selected'], 'runs': records, 'by_depth': aggregates,
            'selected_metrics': selected['metrics'], 'polynomial_metrics': polynomial,
            'batching_trajectory_max_error': batching_error, 'new_holdout_disjoint': True,
            'new_holdout_used_for_selection': False, 'qpu_jobs': 0}, archive, histories


def write_plots(output, report, archive, histories):
    selected_name = next(row['name'] for row in report['runs'] if row['run'] == report['selection']['run'])
    colors = {1: '#888888', 2: '#bd6513', 3: '#1668bb'}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    with np.load('results/baseline/evaluation/predictions.npz') as old:
        for j, name in enumerate(('sin', 'cos')):
            ax = axes[0, j]
            ax.plot(archive['grid_t'], archive['grid_y'][:, j], color='black', linestyle='--', linewidth=2, label='Exact NumPy target')
            ax.plot(old['grid_t'], old['qcl_grid'][:, j], color='#888888', label='One-layer baseline')
            ax.plot(archive['grid_t'], archive[selected_name + '_grid'][:, j], color='#1668bb', label=f'Selected: {selected_name}')
            ax.set(title=f'{name}(πt): our original circuit family', xlabel='t', ylabel='Prediction')
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
    for depth in report['spec']['depths']:
        group = [row for row in report['runs'] if row['depth'] == depth]
        train = np.stack([histories[row['name']][:, 0] for row in group])
        steps = np.arange(train.shape[1])
        axes[1, 0].plot(steps, train.mean(axis=0), color=colors[depth], label=f'{depth} layer(s): three-seed mean')
        axes[1, 0].fill_between(steps, train.min(axis=0), train.max(axis=0), color=colors[depth], alpha=.1)
        values = np.array([row['validation_joint_loss'] for row in group])
        axes[1, 1].scatter(depth + np.linspace(-.08, .08, len(values)), values, color=colors[depth])
        axes[1, 1].errorbar(depth, values.mean(), yerr=values.std(ddof=1), fmt='s', color=colors[depth], capsize=5)
    axes[1, 0].set(yscale='log', xlabel='Adam updates', ylabel='Joint training loss', title='Training mean and min–max range')
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].set(yscale='log', xlabel='Trainable layers', ylabel='Joint validation loss', xticks=[1, 2, 3],
                   title='Validation: seed results and mean ± sample SD')
    for ax in axes[1]:
        ax.grid(alpha=.2)
    fig.suptitle('Depth study: matched depth and initialization study — original two-qubit QCL')
    fig.tight_layout()
    fig.savefig(output / 'comparison.png', dpi=150)
    plt.close(fig)


def main(verify=False):
    torch.set_num_threads(1)
    output = ROOT / 'evaluation'
    if not verify and output.exists():
        raise FileExistsError('Evaluation exists; use --verify instead')
    report, archive, histories = calculate()
    if verify:
        saved = json.loads((output / 'metrics.json').read_text())
        assert report == saved, 'Recomputed report differs from saved report'
        with np.load(output / 'predictions.npz') as frozen:
            assert set(frozen.files) == set(archive)
            for key, value in archive.items():
                np.testing.assert_allclose(frozen[key], value, atol=1e-13, rtol=0)
        with (output / 'runs.csv').open() as stream:
            csv_rows = list(csv.DictReader(stream))
        assert len(csv_rows) == len(report['runs'])
        for row, record in zip(csv_rows, report['runs']):
            for key in ('depth', 'seed', 'selected_step', 'parameters', 'logical_gates', 'logical_cnots', 'validation_joint_loss', 'loop_seconds'):
                assert float(row[key]) == record[key]
            for split, prefix in (('test', 'test'), ('new', 'new')):
                for name in ('sin', 'cos'):
                    assert float(row[f'{prefix}_{name}_mse']) == record['metrics'][split][name]['mse']
        audit = {'status': 'PASS', 'runs_verified': 9, 'histories_verified': 9,
                 'history_rows_verified': 9 * 251, 'prediction_values_recomputed': 9 * 2 * (48+16+24+256+401),
                 'batching_trajectory_max_error': report['batching_trajectory_max_error'],
                 'validation_selection_verified': True, 'csv_rows_verified': len(csv_rows), 'new_holdout_disjoint': True, 'qpu_jobs': 0}
        (output / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
        print(json.dumps(audit, indent=2))
        return
    output.mkdir(parents=True)
    np.savez(output / 'predictions.npz', **archive)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    with (output / 'runs.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['depth', 'seed', 'selected_step', 'parameters', 'logical_gates', 'logical_cnots', 'validation_joint_loss',
                         'test_sin_mse', 'test_cos_mse', 'new_sin_mse', 'new_cos_mse', 'loop_seconds'])
        for row in report['runs']:
            writer.writerow([row['depth'], row['seed'], row['selected_step'], row['parameters'], row['logical_gates'], row['logical_cnots'],
                             row['validation_joint_loss'], row['metrics']['test']['sin']['mse'], row['metrics']['test']['cos']['mse'],
                             row['metrics']['new']['sin']['mse'], row['metrics']['new']['cos']['mse'], row['loop_seconds']])
    write_plots(output, report, archive, histories)
    lines = ['# Depth study review', '', 'Original two-qubit circuit only. Nine matched 250-update Adam runs; three seeds per depth.', '',
             '| Depth | Parameters | CNOTs | Validation loss mean ± sample SD | New-holdout sine MSE mean ± SD | New-holdout cosine MSE mean ± SD |',
             '|---|---:|---:|---:|---:|---:|']
    for row in report['by_depth']:
        sin = row['metrics']['new']['sin']['mse']
        cos = row['metrics']['new']['cos']['mse']
        lines.append(f'| {row["depth"]} | {row["parameters"]} | {row["logical_cnots"]} | '
                     f'{row["validation_joint_loss_mean"]:.7g} ± {row["validation_joint_loss_sample_std"]:.3g} | '
                     f'{sin["mean"]:.7g} ± {sin["sample_std"]:.3g} | {cos["mean"]:.7g} ± {cos["sample_std"]:.3g} |')
    selection = report['selection']
    lines += ['', f'Active checkpoint: depth {selection["layers"]}, seed {selection["parameter_seed"]}, step {selection["selected_step"]}, '
              'selected using joint validation loss only. The new 256-input holdout was generated after selection was frozen.', '',
              '| Selected output | New-holdout MSE | New-holdout MAE | New-holdout maximum error | Endpoint-inclusive grid maximum error |',
              '|---|---:|---:|---:|---:|']
    for name in ('sin', 'cos'):
        new = report['selected_metrics']['new'][name]
        grid = report['selected_metrics']['grid'][name]
        lines.append(f'| {name} | {new["mse"]:.8g} | {new["mae"]:.8g} | {new["max_absolute_error"]:.8g} | {grid["max_absolute_error"]:.8g} |')
    selected = next(row for row in report['runs'] if row['run'] == selection['run'])
    lines += ['', f'Provisional per-output validation target met: {selected["validation_target_met"]}. '
              'That target is MSE <=0.001 and MAE <=0.03 on each output. Study completion is not an accuracy claim.', '',
              'Nominal parameters are 6 per layer; each layer has 7 logical gates including one CNOT. '
              'Rot is counted as one logical gate; these are not compiled hardware costs. Two terminal Z rotations '
              'are unobservable in these readouts. Extra layers also re-upload data, so this is a comparison of '
              'the predeclared layered architecture, not an isolated entanglement-only effect.', '',
              'Three seeds provide a small sensitivity study, not a broad statistical guarantee. No barren-plateau '
              'or advantage claim follows. Seeds share a dataset and are not independent dataset replicates. '
              'The new holdout is now exposed; do not tune against its scores. Boundary coverage and feature-map '
              'changes require separate protocols if the fit remains poor. No finite-shot or QPU jobs ran.', '',
              f'Batching versus original scalar Baseline trajectory maximum loss difference: {report["batching_trajectory_max_error"]:.3g}. '
              'Timing measures CPU optimization loops under the same batched implementation; no cross-hardware speed claim.']
    (output / 'REVIEW.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    main(parser.parse_args().verify)
