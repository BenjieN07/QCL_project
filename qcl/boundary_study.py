"""Controlled continuation and boundary-data experiments on our original circuit."""
import argparse
from contextlib import redirect_stdout
import copy
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from .data import make_data, targets
from .model import QCLModel
from .training import regression_loss

ROOT = Path('results/boundary')
SPEC = Path('configs/boundary_study.json')


def make_datasets(spec, config):
    data = make_data(config)
    for label, key in (('anchor', 'training_anchor_positive'), ('collar', 'validation_collar_positive')):
        positive = np.asarray(spec[key], dtype=float)
        values = np.sort(np.concatenate((-positive, positive)))
        assert len(np.unique(values)) == len(values) and np.isfinite(values).all() and (np.abs(values) <= 1).all()
        data[label + '_t'], data[label + '_y'] = values, targets(values)
    training = np.concatenate((data['train_t'], data['anchor_t']))
    validation = np.concatenate((data['validation_t'], data['collar_t']))
    assert not np.isin(training, validation).any()
    assert not np.isin(training, data['test_t']).any()
    assert len(np.unique(validation)) == len(validation)
    data['endpoint_t'] = np.array([-1., 1.])
    data['endpoint_y'] = targets(data['endpoint_t'])
    return data


def restore_start(config, source):
    model = QCLModel(config['layers'], config['parameter_seed'], config['input_scale'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('schedule', [[0, .05]])[0][1])
    if config.get('start_mode', 'resume') == 'fresh':
        return model, optimizer
    model.load_state_dict(source['model_state_dict'])
    optimizer.load_state_dict(copy.deepcopy(source['optimizer_state_dict']))
    assert int(optimizer.state[model.weights]['step'].item()) == source['step']
    assert torch.equal(model.weights.detach(), source['model_state_dict']['weights'])
    return model, optimizer


def prepare():
    spec = json.loads(SPEC.read_text())
    source_path = Path(spec['source_run']) / 'best.pt'
    source = torch.load(source_path, weights_only=True)
    data = make_datasets(spec, source['config'])
    ROOT.mkdir(parents=True, exist_ok=True)
    data_path = ROOT / 'training_data.npz'
    if data_path.exists():
        with np.load(data_path) as saved:
            assert set(saved.files) == set(data)
            for key, value in data.items():
                np.testing.assert_array_equal(saved[key], value)
    else:
        np.savez(data_path, **data)
    jobs = []
    for trial in spec['trials']:
        config = dict(source['config'])
        start_step = 0 if trial.get('start_mode') == 'fresh' else source['step']
        extra = trial.get('additional_updates', spec['additional_updates'])
        config.update(trial, source_run=spec['source_run'], source_step=start_step,
                      steps=start_step + extra, additional_updates=extra,
                      phase=f'Boundary study: {trial["name"]}', baseline_validation_loss=source['validation_loss'],
                      selection_boundary_weight=spec['selection_boundary_weight'],
                      augmented_training_boundary_weight=spec['augmented_training_boundary_weight'],
                      source_checkpoint_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest())
        path = Path('configs/boundary') / f'{trial["name"]}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            assert json.loads(path.read_text()) == config
        else:
            path.write_text(json.dumps(config, indent=2) + '\n')
        jobs.append((path, ROOT / 'runs' / trial['name']))
    return spec, jobs


def train(config_path, output, spec):
    if (output / 'summary.json').exists():
        raise FileExistsError('Completed run exists')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = json.loads(config_path.read_text())
    source = torch.load(Path(config['source_run']) / 'best.pt', weights_only=True)
    model, optimizer = restore_start(config, source)
    data = make_datasets(spec, config)
    splits = ('train', 'anchor', 'validation', 'collar', 'endpoint')
    lengths = [len(data[s + '_t']) for s in splits]
    all_x = torch.tensor(np.concatenate([data[s + '_t'] for s in splits]), dtype=torch.float64)
    ys = {s: torch.tensor(data[s + '_y'], dtype=torch.float64) for s in splits}
    train_x = torch.tensor(np.concatenate((data['train_t'], data['anchor_t'])) if config['augment'] else data['train_t'], dtype=torch.float64)
    history, best, fallback = [], None, None
    started = time.perf_counter()
    source_hashes = {str(p).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('qcl').glob('*.py'))}

    def snapshot(total, score, losses, endpoint_error):
        return {'step': total, 'config': config, 'model_state_dict': copy.deepcopy(model.state_dict()),
                'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()), 'selection_score': score,
                'train_loss': losses['train'], 'validation_loss': losses['validation'],
                'collar_validation_loss': losses['collar'], 'endpoint_max_error': endpoint_error}

    torch.save({'step': config['source_step'], 'config': config, 'model_state_dict': copy.deepcopy(model.state_dict()),
                'optimizer_state_dict': copy.deepcopy(optimizer.state_dict())}, output / 'start.pt')
    for update in range(config['additional_updates'] + 1):
        if update < config['additional_updates']:
            rate = next(rate for begin, rate in reversed(config['schedule']) if update >= begin)
            for group in optimizer.param_groups:
                group['lr'] = rate
        total = config['source_step'] + update
        with torch.no_grad():
            predictions = dict(zip(splits, torch.split(model(all_x), lengths)))
            losses = {s: regression_loss(predictions[s], ys[s]).item() for s in splits}
            endpoint_error = (predictions['endpoint'] - ys['endpoint']).abs().max().item()
        boundary_weight = config['selection_boundary_weight']
        score = (1-boundary_weight)*losses['validation'] + boundary_weight*losses['collar']
        eligible = losses['validation'] <= config['baseline_validation_loss'] + 1e-12
        if fallback is None or score < fallback['selection_score']:
            fallback = snapshot(total, score, losses, endpoint_error)
        if eligible and (best is None or score < best['selection_score']):
            best = snapshot(total, score, losses, endpoint_error)
        weight = config['augmented_training_boundary_weight'] if config['augment'] else 0.
        row = {'additional_step': update, 'total_step': total,
               'training_objective': (1-weight)*losses['train'] + weight*losses['anchor'],
               'train_original_loss': losses['train'], 'train_anchor_loss': losses['anchor'],
               'validation_original_loss': losses['validation'], 'validation_collar_loss': losses['collar'],
               'selection_score': score, 'eligible': eligible, 'endpoint_max_error': endpoint_error,
               'learning_rate': optimizer.param_groups[0]['lr'], 'gradient_norm': None}
        if update < config['additional_updates']:
            optimizer.zero_grad(set_to_none=True)
            train_prediction = model(train_x)
            objective = regression_loss(train_prediction[:lengths[0]], ys['train'])
            if config['augment']:
                objective = (1-weight)*objective + weight*regression_loss(train_prediction[lengths[0]:], ys['anchor'])
            objective.backward()
            if model.weights.grad is None or not torch.isfinite(model.weights.grad).all():
                raise RuntimeError('Broken or nonfinite parameter gradient')
            row['gradient_norm'] = model.weights.grad.norm().item()
            optimizer.step()
        history.append(row)
        if update % 100 == 0 or update == config['additional_updates']:
            print(f'{config["name"]} total step {total}: validation={losses["validation"]:.7f}, collar={losses["collar"]:.7f}, endpoint max={endpoint_error:.5f}', flush=True)
    has_eligible = best is not None
    best = best if has_eligible else fallback
    torch.save(best, output / 'best.pt')
    torch.save(snapshot(config['steps'], score, losses, endpoint_error), output / 'final.pt')
    final_counter = int(optimizer.state[model.weights]['step'].item())
    assert final_counter == config['steps']
    restored = QCLModel(config['layers'], config['parameter_seed'], config['input_scale'])
    restored.load_state_dict(torch.load(output / 'best.pt', weights_only=True)['model_state_dict'])
    with torch.no_grad():
        restored_values = dict(zip(splits, torch.split(restored(all_x), lengths)))
        original_validation = regression_loss(restored_values['validation'], ys['validation']).item()
        collar_validation = regression_loss(restored_values['collar'], ys['collar']).item()
    assert abs(original_validation - best['validation_loss']) < 1e-12
    assert abs(collar_validation - best['collar_validation_loss']) < 1e-12
    with (output / 'history.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    xs = [r['total_step'] for r in history]
    for label, key in (('Original validation', 'validation_original_loss'), ('Collar validation', 'validation_collar_loss')):
        axes[0].plot(xs, [r[key] for r in history], label=label)
    axes[0].set(xlabel='Total updates on this branch', ylabel='Joint squared error', yscale='log')
    axes[0].legend(fontsize=8)
    axes[1].plot(xs, [r['endpoint_max_error'] for r in history])
    axes[1].axhline(.05, color='gray', linestyle=':', label='Provisional endpoint goal')
    axes[1].set(xlabel='Total updates on this branch', ylabel='Maximum exact-endpoint error')
    axes[1].legend(fontsize=8)
    fig.suptitle(config['name'])
    fig.tight_layout()
    fig.savefig(output / 'history.png', dpi=150)
    plt.close(fig)
    summary = {'config': config, 'source_step': config['source_step'], 'additional_updates': config['additional_updates'],
               'final_total_step': config['steps'], 'optimizer_final_counter': final_counter,
               'selected_step': best['step'], 'selected_score': best['selection_score'],
               'selected_validation_original_loss': best['validation_loss'],
               'selected_validation_collar_loss': best['collar_validation_loss'],
               'selected_endpoint_max_error': best['endpoint_max_error'], 'eligible': has_eligible,
               'checkpoint_reload_passed': True, 'optimizer_state_restored': config.get('start_mode', 'resume') != 'fresh',
               'source_weights_restored': config.get('start_mode', 'resume') != 'fresh',
               'input_encoding_changed': config['input_scale'] != 1.,
               'elapsed_seconds': time.perf_counter()-started,
               'versions': {p: importlib.metadata.version(p) for p in ('pennylane', 'torch', 'numpy', 'matplotlib')},
               'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
               'source_sha256': source_hashes, 'new_holdout_read': False, 'test_used_for_selection': False, 'qpu_jobs': 0}
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


def select(jobs):
    candidates = []
    for path, output in jobs:
        summary = json.loads((output / 'summary.json').read_text())
        assert summary['config'] == json.loads(path.read_text())
        candidates.append({'run': str(output).replace('\\', '/'), 'name': summary['config']['name'],
                           'eligible': summary['eligible'], 'score': summary['selected_score'],
                           'selected_step': summary['selected_step'], 'input_scale': summary['config']['input_scale'],
                           'layers': summary['config']['layers'], 'parameter_seed': summary['config']['parameter_seed']})
    eligible = [r for r in candidates if r['eligible']]
    best = min(eligible, key=lambda row: row['score'])
    registry = dict(best, model='original',
                    selection='minimum balanced validation score among eligible trials; original-validation loss cannot worsen from Day 4',
                    test_or_fresh_holdout_used_for_selection=False, new_holdout_read_at_registration=False)
    (ROOT / 'selection.json').write_text(json.dumps({'candidates': candidates, 'selected': registry}, indent=2) + '\n')
    Path('results/current_model.json').write_text(json.dumps(registry, indent=2) + '\n')
    print(json.dumps(registry, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--select-only', action='store_true')
    args = parser.parse_args()
    spec, jobs = prepare()
    if args.prepare_only:
        print(f'Prepared {len(jobs)} fixed experiments; no new holdout read.')
    elif args.select_only:
        select(jobs)
    else:
        for path, output in jobs:
            if (output / 'summary.json').exists():
                assert json.loads((output / 'summary.json').read_text())['config'] == json.loads(path.read_text())
                print(f'Already complete: {output}', flush=True)
                continue
            output.mkdir(parents=True, exist_ok=True)
            print(f'Starting {output}', flush=True)
            with (output / 'training.log').open('w') as log, redirect_stdout(log):
                train(path, output, spec)
            result = json.loads((output / 'summary.json').read_text())
            print(f'Completed {output}: score={result["selected_score"]:.7f}, endpoint max={result["selected_endpoint_max_error"]:.5f}, eligible={result["eligible"]}', flush=True)
        print('Training complete; freeze selection separately before new holdout evaluation.', flush=True)
