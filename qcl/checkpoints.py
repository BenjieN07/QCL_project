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


def verify_saved_evidence():
    import csv
    import hashlib
    import json
    from pathlib import Path
    import numpy as np

    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'results/summary/checkpoints.json').read_text())
    for name,digest in manifest.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    progress=json.loads((root/'results/progress/progress.json').read_text())
    with np.load(root/'results/progress/curves.npz') as curves:
        for index,row in enumerate(progress['milestones']):
            mse=((curves[f'stage_{index}_test']-curves['test_y'])**2).mean(axis=0)
            np.testing.assert_allclose(mse,[row['sin_test_mse'],row['cos_test_mse']],atol=1e-12,rtol=0)
            if row['run']:
                name=row['run']+'/best.pt'
                assert manifest[name]==row['checkpoint_sha256']
    with (root/'results/progress/experiments.csv').open() as stream:
        trials=list(csv.DictReader(stream))
    assert len(trials)==20
    for row in trials:
        base={'Depth':'depth_study','Boundary':'boundary','Mirrored':'mirror'}[row['group']]
        cfg=json.loads((root/'results'/base/'runs'/row['trial']/'config.json').read_text())
        assert row['start']==cfg.get('start_mode','resume' if cfg.get('source_run') else 'fresh')
    audit={'status':'PASS','checkpoint_files':len(manifest),'checkpoint_bytes_unchanged':True,
           'progress_milestones':len(progress['milestones']),'experiment_rows':len(trials),
           'progress_metrics_recomputed':True,'fresh_and_resumed_labels_verified':True,
           'training_updates':0,'qpu_jobs':0}
    (root/'results/summary/cleanup_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify',action='store_true',required=True,help='Verify frozen checkpoint hashes and saved progress tables')
    parser.parse_args()
    verify_saved_evidence()
