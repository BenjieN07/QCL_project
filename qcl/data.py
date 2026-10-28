"""Fixed, disjoint training, validation, and test inputs."""
import numpy as np


def targets(t):
    return np.column_stack((np.sin(np.pi * t), np.cos(np.pi * t)))


def make_data(config):
    sizes = [config[k] for k in ('train_size', 'validation_size', 'test_size')]
    if any(not isinstance(n, int) or n < 1 for n in sizes):
        raise ValueError('Split sizes must be positive integers')
    rng = np.random.default_rng(config['data_seed'])
    points = rng.uniform(-1, 1, sum(sizes))
    if len(np.unique(points)) != len(points):
        raise ValueError('Duplicate inputs; choose another data seed')
    splits = np.split(points, np.cumsum(sizes)[:-1])
    result = {}
    for name, values in zip(('train', 'validation', 'test'), splits):
        result[name + '_t'] = np.sort(values)
        result[name + '_y'] = targets(result[name + '_t'])
    result['grid_t'] = np.linspace(-1, 1, config['grid_size'])
    result['grid_y'] = targets(result['grid_t'])
    return result
