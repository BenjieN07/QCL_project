"""Run with python -m qcl.prepare."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .data import make_data


def main():
    config = json.loads(Path('configs/smoke.json').read_text())
    data = make_data(config)
    repeat = make_data(config)
    assert all(np.array_equal(data[k], repeat[k]) for k in data)
    all_inputs = np.concatenate([data[n + '_t'] for n in ('train', 'validation', 'test')])
    assert len(np.unique(all_inputs)) == len(all_inputs)
    assert np.allclose(np.sum(data['grid_y'] ** 2, axis=1), 1)
    out = Path('results/validation')
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / 'data.npz', **data)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for j, ax in enumerate(axes):
        ax.plot(data['grid_t'], data['grid_y'][:, j], label='Target')
        ax.scatter(data['train_t'], data['train_y'][:, j], s=12, label='Train')
        ax.scatter(data['validation_t'], data['validation_y'][:, j], s=18, marker='x', label='Validation')
        ax.set(title=('sin(pi t)', 'cos(pi t)')[j], xlabel='t', ylabel='value')
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'targets.png', dpi=150)
    plt.close(fig)
    checks = {'deterministic': True, 'disjoint': True, 'target_identity': True,
              'train': len(data['train_t']), 'validation': len(data['validation_t']),
              'test': len(data['test_t']), 'config': config}
    (out / 'data_checks.json').write_text(json.dumps(checks, indent=2) + '\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
