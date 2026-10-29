"""Prepare, run, and select the predeclared original-circuit Day 4 study."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
from .train import run

SPEC = Path('configs/day4_study.json')
ROOT = Path('results/day4')


def prepare():
    spec = json.loads(SPEC.read_text())
    jobs = []
    for depth in spec['depths']:
        for seed in spec['parameter_seeds']:
            name = f'depth{depth}_seed{seed}'
            config = {key: spec[key] for key in ('data_seed', 'train_size', 'validation_size', 'test_size', 'grid_size', 'steps', 'learning_rate')}
            config.update(phase=f'Day 4 depth {depth}, seed {seed}', layers=depth, parameter_seed=seed)
            path = Path('configs/day4') / f'{name}.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and json.loads(path.read_text()) != config:
                raise ValueError(f'Existing configuration differs: {path}')
            path.write_text(json.dumps(config, indent=2) + '\n')
            jobs.append((path, ROOT / 'runs' / name))
    return jobs


def select(jobs):
    records = []
    for config_path, output in jobs:
        config = json.loads(config_path.read_text())
        summary = json.loads((output / 'summary.json').read_text())
        if summary['config'] != config:
            raise ValueError(f'Mismatched study result: {output}')
        records.append({'run': str(output).replace('\\', '/'), 'layers': config['layers'],
                        'parameter_seed': config['parameter_seed'], 'selected_step': summary['best_step'],
                        'selected_validation_loss': summary['best_validation_loss']})
    best = min(records, key=lambda row: (row['selected_validation_loss'], row['layers'], row['parameter_seed']))
    selection = dict(best, model='original', selection='minimum joint validation loss across all nine fixed runs; ties prefer depth then seed',
                     test_or_fresh_holdout_used_for_selection=False, new_holdout_read_at_registration=False)
    (ROOT / 'selection.json').write_text(json.dumps({'candidates': records, 'selected': selection}, indent=2) + '\n')
    Path('results/current_model.json').write_text(json.dumps(selection, indent=2) + '\n')
    print('Frozen validation-only selection:\n' + json.dumps(selection, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--select-only', action='store_true')
    args = parser.parse_args()
    jobs = prepare()
    if args.prepare_only:
        print(f'Prepared {len(jobs)} matched configurations; no training or test evaluation.')
    elif args.select_only:
        select(jobs)
    else:
        completed = []
        for config_path, output in jobs:
            if (output / 'summary.json').exists():
                if json.loads((output / 'summary.json').read_text())['config'] != json.loads(config_path.read_text()):
                    raise ValueError(f'Resume configuration mismatch: {output}')
                print(f'Already complete: {output}', flush=True)
            else:
                print(f'Starting {output}', flush=True)
                output.mkdir(parents=True, exist_ok=True)
                with (output / 'training.log').open('w') as log, redirect_stdout(log):
                    run(output, config_path)
                result = json.loads((output / 'summary.json').read_text())
                print(f'Completed {output}: validation={result["best_validation_loss"]:.8f}', flush=True)
            completed.append(str(output).replace('\\', '/'))
            (ROOT / 'status.json').write_text(json.dumps({'completed': len(completed), 'total': len(jobs), 'runs': completed,
                                                        'test_labels_used': False, 'new_holdout_read': False}, indent=2) + '\n')
        print('All nine runs complete. Freeze selection separately with --select-only before holdout evaluation.', flush=True)
