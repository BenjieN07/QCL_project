"""Run a bounded Day 2 training smoke test and save reproducible evidence."""
import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from .data import make_data
from .model import QCLModel
from .training import regression_loss, checked_backward


def run(output):
    config = json.loads(Path('config.json').read_text())
    steps = config['smoke_steps']
    rate = config['learning_rate']
    if not isinstance(steps, int) or steps < 1 or not 0 < rate < 1:
        raise ValueError('Use a positive smoke_steps integer and learning_rate in (0, 1)')
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'summary.json').exists():
        raise FileExistsError(f'{output} already contains a run; choose a fresh --output directory')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    data = make_data(config)
    # Only train and validation labels are used. Test metrics belong to Day 3.
    train_x = torch.tensor(data['train_t'], dtype=torch.float64)
    train_y = torch.tensor(data['train_y'], dtype=torch.float64)
    val_x = torch.tensor(data['validation_t'], dtype=torch.float64)
    val_y = torch.tensor(data['validation_y'], dtype=torch.float64)
    model = QCLModel(config['layers'], config['parameter_seed'])
    optimizer = torch.optim.Adam(model.parameters(), lr=rate)
    history = []
    best_validation = float('inf')
    best = None
    started = time.perf_counter()
    for step in range(steps + 1):
        with torch.no_grad():
            train_loss = regression_loss(model(train_x), train_y).item()
            validation_loss = regression_loss(model(val_x), val_y).item()
        if validation_loss < best_validation:
            best_validation = validation_loss
            best = {'step': step, 'model_state_dict': copy.deepcopy(model.state_dict()),
                    'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                    'train_loss': train_loss, 'validation_loss': validation_loss, 'config': config}
        row = {'step': step, 'train_loss': train_loss, 'validation_loss': validation_loss,
               'gradient_norm': None}
        if step < steps:
            optimizer.zero_grad()
            row['gradient_norm'] = checked_backward(regression_loss(model(train_x), train_y), model)
            optimizer.step()
        history.append(row)
        if step % 5 == 0 or step == steps:
            print(f'Step {step:3d}: train={train_loss:.6f}, validation={validation_loss:.6f}', flush=True)
    elapsed = time.perf_counter() - started
    torch.save(best, output / 'best.pt')
    # Verify serialized weights reproduce the selected checkpoint's loss.
    restored = QCLModel(config['layers'], config['parameter_seed'])
    restored.load_state_dict(torch.load(output / 'best.pt', weights_only=True)['model_state_dict'])
    with torch.no_grad():
        restored_loss = regression_loss(restored(val_x), val_y).item()
    assert abs(restored_loss - best_validation) < 1e-12
    improvement = (history[0]['train_loss'] - history[-1]['train_loss']) / history[0]['train_loss']
    assert improvement > 0.01, 'Smoke test did not improve training loss by at least 1%'
    with (output / 'history.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r['step'] for r in history], [r['train_loss'] for r in history], label='Train')
    ax.plot([r['step'] for r in history], [r['validation_loss'] for r in history], label='Validation')
    ax.axvline(best['step'], color='gray', linestyle=':', label='Selected checkpoint')
    ax.set(xlabel='Optimizer updates completed', ylabel='Mean sum of squared errors',
           title=f'Day 2: {steps}-step training smoke test')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'loss.png', dpi=150)
    plt.close(fig)
    sources = {str(p).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(Path('qcl').glob('*.py'))}
    summary = {
        'status': 'PASS', 'purpose': 'Training smoke test; not a final accuracy study',
        'config': config, 'optimizer': 'Adam', 'loss_reduction': 'mean over samples, sum over outputs',
        'initial_train_loss': history[0]['train_loss'], 'final_train_loss': history[-1]['train_loss'],
        'relative_train_improvement': improvement, 'best_step': best['step'],
        'best_validation_loss': best_validation, 'checkpoint_reload_passed': True,
        'minimum_gradient_norm': min(r['gradient_norm'] for r in history[:-1]),
        'elapsed_seconds': elapsed, 'python': platform.python_version(),
        'versions': {p: importlib.metadata.version(p) for p in ('pennylane', 'torch', 'numpy', 'matplotlib')},
        'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': sources, 'test_labels_used': False, 'qpu_jobs': 0,
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/day2/smoke'))
    run(parser.parse_args().output)
