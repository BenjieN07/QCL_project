"""Read immutable checkpoints while adapting retired experiment labels and paths."""
import torch


# Historical checkpoint bytes remain unchanged to preserve hardware/provenance hashes.
_LABELS = (
    ('day4_study', 'depth_study'), ('day3.json', 'baseline.json'),
    ('day1', 'validation'), ('day2', 'smoke'), ('day3', 'baseline'), ('day4', 'depth_study'),
    ('Day 1', 'Circuit validation'), ('Day 2', 'Training smoke test'),
    ('Day 3', 'Baseline'), ('Day 4', 'Depth study'),
    ('Baseline one-layer baseline', 'One-layer baseline'),
    ('results/baseline/baseline', 'results/baseline/run'),
    ('results/smoke/smoke', 'results/smoke/run'),
)


def normalize_metadata(value):
    if isinstance(value, str):
        for old, new in _LABELS:
            value = value.replace(old, new)
        return value
    if isinstance(value, dict):
        return {normalize_metadata(k): normalize_metadata(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_metadata(v) for v in value]
    if isinstance(value, tuple):
        return tuple(normalize_metadata(v) for v in value)
    return value


def load_checkpoint(path, **kwargs):
    return normalize_metadata(torch.load(path, **kwargs))
