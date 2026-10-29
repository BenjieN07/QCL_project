"""Train a predeclared local pilot with exact gradients and accepted-iterate validation."""
import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import torch
from .data import make_data
from .evaluate import metrics
from .paper_model import build_model
from .training import regression_loss


def train(config_path, output):
    config = json.loads(config_path.read_text())
    if output.exists():
        raise FileExistsError('Use a fresh output directory for each pilot')
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = build_model(config)
    data = make_data(config)
    train_x, train_y = (torch.tensor(data[k], dtype=torch.float64) for k in ('train_t', 'train_y'))
    val_x, val_y = (torch.tensor(data[k], dtype=torch.float64) for k in ('validation_t', 'validation_y'))
    parameters = list(model.parameters())
    initial = torch.nn.utils.parameters_to_vector(parameters).detach().numpy().copy()
    cache = {}
    evaluations = 0
    history = []
    best = None
    started = time.perf_counter()

    def assign(vector):
        offset = 0
        with torch.no_grad():
            for parameter in parameters:
                size = parameter.numel()
                parameter.copy_(torch.tensor(vector[offset:offset+size], dtype=torch.float64).reshape(parameter.shape))
                offset += size

    def objective(vector):
        nonlocal evaluations
        if 'x' in cache and np.array_equal(cache['x'], vector):
            return cache['loss'], cache['gradient'].copy()
        assign(vector)
        model.zero_grad(set_to_none=True)
        loss = regression_loss(model(train_x), train_y)
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
            raise RuntimeError('Missing or nonfinite gradient')
        gradient = torch.cat([p.grad.flatten() for p in parameters]).detach().numpy().copy()
        evaluations += 1
        cache.update(x=np.array(vector, copy=True), loss=loss.item(), gradient=gradient)
        return loss.item(), gradient.copy()

    def record(vector):
        nonlocal best
        train_loss, gradient = objective(vector)
        assign(vector)
        with torch.no_grad():
            prediction = model(val_x)
            validation_loss = regression_loss(prediction, val_y).item()
        iteration = len(history)
        row = {'iteration': iteration, 'train_loss': train_loss, 'validation_loss': validation_loss,
               'gradient_norm': float(np.linalg.norm(gradient)), 'objective_evaluations': evaluations}
        history.append(row)
        if best is None or validation_loss < best['validation_loss']:
            best = {'iteration': iteration, 'train_loss': train_loss, 'validation_loss': validation_loss,
                    'model_state_dict': copy.deepcopy(model.state_dict()), 'config': config}
        if iteration % 10 == 0:
            print(f'{config["model"]} iteration {iteration}: train={train_loss:.8f}, validation={validation_loss:.8f}', flush=True)

    record(initial)
    result = minimize(objective, initial, jac=True, method='BFGS', callback=record,
                      options={'maxiter': config['max_iterations'], 'gtol': config['gradient_tolerance']})
    if not np.array_equal(cache['x'], result.x):
        objective(result.x)
    torch.save(best, output / 'best.pt')
    restored = build_model(config)
    restored.load_state_dict(torch.load(output / 'best.pt', weights_only=True)['model_state_dict'])
    with torch.no_grad():
        val_prediction = restored(val_x).numpy()
        restored_train = regression_loss(restored(train_x), train_y).item()
    assert abs(restored_train - best['train_loss']) < 1e-12
    val_metrics = metrics(val_prediction, data['validation_y'])
    assert abs(sum(v['mse'] for v in val_metrics.values()) - best['validation_loss']) < 1e-12
    target_met = all(v['mse'] <= .001 and v['mae'] <= .03 for v in val_metrics.values())
    with (output / 'history.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    np.savez(output / 'optimizer_final.npz', x=result.x, gradient=result.jac, inverse_hessian=result.hess_inv)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([r['iteration'] for r in history], [r['train_loss'] for r in history], label='Train')
    ax.plot([r['iteration'] for r in history], [r['validation_loss'] for r in history], label='Validation')
    ax.axvline(best['iteration'], color='gray', linestyle=':', label='Selected iterate')
    ax.set(xlabel='Accepted BFGS updates', ylabel='Mean sum of squared errors', title=config['model'])
    ax.set_yscale('log')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'loss.png', dpi=150)
    plt.close(fig)
    summary = {'config': config, 'optimizer': 'SciPy BFGS with analytic PyTorch gradients',
               'solver_success': bool(result.success), 'solver_status': int(result.status),
               'solver_message': str(result.message), 'accepted_updates': int(result.nit),
               'scipy_function_evaluations': int(result.nfev), 'actual_objective_evaluations': evaluations,
               'initial_train_loss': history[0]['train_loss'], 'final_train_loss': float(result.fun),
               'selected_iteration': best['iteration'], 'selected_train_loss': best['train_loss'],
               'selected_validation_loss': best['validation_loss'], 'validation_metrics': val_metrics,
               'validation_accuracy_target_met': target_met, 'checkpoint_reload_passed': True,
               'parameters': len(initial), 'elapsed_seconds': time.perf_counter()-started,
               'python': platform.python_version(),
               'versions': {p: importlib.metadata.version(p) for p in ('pennylane', 'torch', 'numpy', 'scipy', 'matplotlib')},
               'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
               'source_sha256': {str(p).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('qcl').glob('*.py'))},
               'test_labels_used': False, 'fresh_holdout_read': False, 'qpu_jobs': 0}
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    train(args.config, args.output)
