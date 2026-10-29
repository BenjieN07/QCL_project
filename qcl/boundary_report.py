"""Post-selection boundary evaluation and saved-artifact audit; no training."""
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
from .boundary_study import make_datasets
from .data import targets
from .evaluate import metrics
from .model import QCLModel

ROOT = Path('results/boundary')
SPLITS = ('train', 'anchor', 'validation', 'collar', 'endpoint')


def load_model(checkpoint):
    config = checkpoint['config']
    model = QCLModel(config['layers'], config['parameter_seed'], config.get('input_scale', 1.))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def optimizer_counter(checkpoint):
    states = checkpoint['optimizer_state_dict']['state']
    counters = [int(state['step'].item()) for state in states.values()]
    assert not counters or len(set(counters)) == 1
    return counters[0] if counters else 0


def checkpoint_losses(checkpoint, data):
    lengths = [len(data[split + '_t']) for split in SPLITS]
    model = load_model(checkpoint)
    with torch.no_grad():
        prediction = model(torch.tensor(np.concatenate([data[s + '_t'] for s in SPLITS]), dtype=torch.float64)).numpy()
    values = np.split(prediction, np.cumsum(lengths)[:-1])
    losses = {s: float(np.mean(np.sum((p-data[s + '_y'])**2, axis=1))) for s, p in zip(SPLITS, values)}
    endpoint = float(np.max(np.abs(values[-1]-data['endpoint_y'])))
    return losses, endpoint


def calculate():
    spec = json.loads(Path('configs/boundary_study.json').read_text())
    selection = json.loads((ROOT / 'selection.json').read_text())
    source_path = Path(spec['source_run']) / 'best.pt'
    source = torch.load(source_path, weights_only=True)
    data = make_datasets(spec, source['config'])
    with np.load(ROOT / 'training_data.npz') as saved:
        for key, value in data.items():
            np.testing.assert_array_equal(saved[key], value)
    checkpoints = {'Day 4 source': source}
    hashes = {'Day 4 source': hashlib.sha256(source_path.read_bytes()).hexdigest()}
    records, histories = [], {}
    for trial in spec['trials']:
        name = trial['name']
        run = ROOT / 'runs' / name
        config = json.loads((run / 'config.json').read_text())
        summary = json.loads((run / 'summary.json').read_text())
        start = torch.load(run / 'start.pt', weights_only=True)
        best = torch.load(run / 'best.pt', weights_only=True)
        final = torch.load(run / 'final.pt', weights_only=True)
        parent_path = Path(config['source_run']) / 'best.pt'
        parent = torch.load(parent_path, weights_only=True)
        assert all(cp['config'] == config for cp in (start, best, final)) and summary['config'] == config
        assert config['source_checkpoint_sha256'] == hashlib.sha256(parent_path.read_bytes()).hexdigest()
        assert config['baseline_validation_loss'] == parent['validation_loss']
        assert optimizer_counter(start) == start['step'] == config['source_step']
        assert optimizer_counter(best) == best['step']
        assert optimizer_counter(final) == final['step'] == config['steps']
        if config.get('start_mode', 'resume') == 'resume':
            torch.testing.assert_close(start['model_state_dict']['weights'], parent['model_state_dict']['weights'], atol=0, rtol=0)
            source_state = next(iter(parent['optimizer_state_dict']['state'].values()))
            start_state = next(iter(start['optimizer_state_dict']['state'].values()))
            for key in ('exp_avg', 'exp_avg_sq', 'step'):
                torch.testing.assert_close(start_state[key], source_state[key], atol=0, rtol=0)
        else:
            assert not start['optimizer_state_dict']['state']
            expected = QCLModel(config['layers'], config['parameter_seed'], config['input_scale'])
            torch.testing.assert_close(start['model_state_dict']['weights'], expected.weights, atol=0, rtol=0)
        with (run / 'history.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == config['additional_updates'] + 1
        assert [int(row['total_step']) for row in rows] == list(range(config['source_step'], config['steps']+1))
        scores = np.array([float(row['selection_score']) for row in rows])
        eligible = np.array([float(row['validation_original_loss']) <= config['baseline_validation_loss']+1e-12 for row in rows])
        assert eligible.tolist() == [row['eligible'] == 'True' for row in rows]
        for i, row in enumerate(rows):
            score = .5 * (float(row['validation_original_loss']) + float(row['validation_collar_loss']))
            assert abs(score-scores[i]) < 1e-12
            weight = .5 if config['augment'] else 0.
            expected_loss = (1-weight)*float(row['train_original_loss']) + weight*float(row['train_anchor_loss'])
            assert abs(expected_loss-float(row['training_objective'])) < 1e-12
            rate = next(rate for begin, rate in reversed(config['schedule']) if min(i, config['additional_updates']-1) >= begin)
            assert float(row['learning_rate']) == rate
            if i < config['additional_updates']:
                assert np.isfinite(float(row['gradient_norm']))
        index = int(np.argmin(np.where(eligible, scores, np.inf))) if eligible.any() else int(np.argmin(scores))
        assert best['step'] == int(rows[index]['total_step']) == summary['selected_step']
        assert summary['eligible'] == bool(eligible.any())
        assert abs(summary['selected_score'] - scores[index]) < 1e-12
        for cp, row in ((start, rows[0]), (best, rows[index]), (final, rows[-1])):
            losses, endpoint = checkpoint_losses(cp, data)
            for split, key in (('train', 'train_original_loss'), ('anchor', 'train_anchor_loss'),
                               ('validation', 'validation_original_loss'), ('collar', 'validation_collar_loss')):
                assert abs(losses[split]-float(row[key])) < 1e-12
            assert abs(endpoint-float(row['endpoint_max_error'])) < 1e-12
        checkpoints[name] = best
        histories[name] = rows
        hashes[name] = hashlib.sha256((run / 'best.pt').read_bytes()).hexdigest()
        records.append({'name': name, 'run': str(run).replace('\\', '/'), 'eligible': bool(eligible.any()),
                        'selected_score': float(scores[index]), 'layers': config['layers'], 'input_scale': config['input_scale'],
                        'start_mode': config.get('start_mode', 'resume'), 'source_step': config['source_step'],
                        'additional_updates': config['additional_updates'], 'selected_step': best['step'],
                        'selected_original_validation_loss': float(rows[index]['validation_original_loss']),
                        'selected_collar_validation_loss': float(rows[index]['validation_collar_loss']),
                        'selected_endpoint_max_error': float(rows[index]['endpoint_max_error']),
                        'final_endpoint_max_error': float(rows[-1]['endpoint_max_error'])})
    expected = min([r for r in records if r['eligible']], key=lambda r: r['selected_score'])
    assert selection['selected']['name'] == expected['name']
    assert selection['selected']['test_or_fresh_holdout_used_for_selection'] is False
    prefix_keys = ('train_original_loss', 'validation_original_loss', 'validation_collar_loss', 'endpoint_max_error')
    prefix_difference = max(abs(float(a[k])-float(b[k])) for a,b in zip(histories['continue_constant'][:251], histories['continue_decay'][:251]) for k in prefix_keys)
    assert prefix_difference <= 1e-12
    refinement_prefix_difference = max(abs(float(a[k])-float(b[k])) for a,b in zip(histories['refine_constant'][:501], histories['refine_decay'][:501]) for k in prefix_keys)
    assert refinement_prefix_difference <= 1e-12
    # First new holdout access occurs only after all declared checkpoints and selection are frozen.
    rng = np.random.default_rng(spec['new_holdout_seed'])
    fresh = np.concatenate((rng.uniform(-.9, .9, spec['holdout_interior_size']),
                            rng.uniform(-1, -.9, spec['holdout_each_boundary_size']),
                            rng.uniform(.9, 1, spec['holdout_each_boundary_size'])))
    regions = np.repeat(np.arange(3), [spec['holdout_interior_size'], spec['holdout_each_boundary_size'], spec['holdout_each_boundary_size']])
    exposed = np.concatenate([data[s + '_t'] for s in ('train', 'anchor', 'validation', 'collar', 'test')]
                             + [np.random.default_rng(20261029).uniform(-1,1,128), np.random.default_rng(20261030).uniform(-1,1,256)])
    assert len(np.unique(fresh)) == len(fresh) and not np.isin(fresh, exposed).any() and (np.abs(fresh) < 1).all()
    data.update(new_t=fresh, new_y=targets(fresh), new_region=regions, grid_t=np.linspace(-1,1,1001))
    data['grid_y'] = targets(data['grid_t'])
    archive, scores = dict(data), {}
    for name, checkpoint in checkpoints.items():
        model = load_model(checkpoint)
        scores[name] = {}
        with torch.no_grad():
            for split in (*SPLITS, 'test', 'new', 'grid'):
                prediction = model(torch.tensor(data[split + '_t'], dtype=torch.float64)).numpy()
                assert np.isfinite(prediction).all() and np.abs(prediction).max() <= 1+1e-12
                archive[name + '_' + split] = prediction
                scores[name][split] = metrics(prediction, data[split + '_y'])
        for index, region in enumerate(('interior', 'left_boundary', 'right_boundary')):
            mask = regions == index
            scores[name]['new_' + region] = metrics(archive[name + '_new'][mask], data['new_y'][mask])
    dense = np.unique(np.concatenate((np.linspace(-1,-.9,301), -1+np.geomspace(1e-8,.1,101))))
    dense_metrics = {}
    for side, inputs in (('left', dense), ('right', -dense[::-1])):
        archive['zoom_' + side + '_t'] = inputs
        archive['zoom_' + side + '_y'] = targets(inputs)
        for name in ('Day 4 source', expected['name']):
            with torch.no_grad():
                prediction = load_model(checkpoints[name])(torch.tensor(inputs, dtype=torch.float64)).numpy()
            archive[name + '_zoom_' + side] = prediction
            dense_metrics[name + '_' + side] = metrics(prediction, targets(inputs))
    selected_scores = scores[expected['name']]
    goals = {'endpoint_max_error_le_0_05': max(v['max_absolute_error'] for v in selected_scores['endpoint'].values()) <= .05,
             'each_collar_validation_mae_le_0_03': all(v['mae'] <= .03 for v in selected_scores['collar'].values()),
             'each_original_validation_mse_le_0_001': all(v['mse'] <= .001 for v in selected_scores['validation'].values())}
    report = {'selection': selection['selected'], 'trials': records, 'metrics': scores, 'dense_boundary_metrics': dense_metrics,
              'goals': goals, 'checkpoint_sha256': hashes, 'new_holdout_seed': spec['new_holdout_seed'],
              'new_holdout_counts': {'interior': spec['holdout_interior_size'], 'left': spec['holdout_each_boundary_size'], 'right': spec['holdout_each_boundary_size']},
              'holdout_is_stratified_not_uniform_domain': True, 'holdout_used_for_selection': False,
              'shared_continuation_prefix_max_difference': prefix_difference,
              'shared_refinement_prefix_max_difference': refinement_prefix_difference, 'qpu_jobs': 0}
    return report, archive


def plots(output, report, archive):
    chosen = report['selection']['name']
    fig, axes = plt.subplots(2,2,figsize=(13,8))
    for j, name in enumerate(('sin','cos')):
        ax = axes[0,j]
        ax.plot(archive['grid_t'], archive['grid_y'][:,j], color='black', linestyle='--', label='Exact NumPy target')
        ax.plot(archive['grid_t'], archive['Day 4 source_grid'][:,j], color='#888888', label='Before: Day 4')
        ax.plot(archive['grid_t'], archive[chosen+'_grid'][:,j], color='#1668bb', label='After: '+chosen)
        ax.set(title=name+'(πt)', xlabel='t', ylabel='Prediction')
        ax.legend(fontsize=8)
        error = axes[1,j]
        error.plot(archive['grid_t'], archive['Day 4 source_grid'][:,j]-archive['grid_y'][:,j], color='#888888', label='Before')
        error.plot(archive['grid_t'], archive[chosen+'_grid'][:,j]-archive['grid_y'][:,j], color='#1668bb', label='After')
        error.axhline(0,color='black',linewidth=.7)
        error.set(xlabel='t',ylabel='Prediction − target',title='Signed error: '+name)
        error.legend(fontsize=8)
        for axis in (ax,error):
            axis.grid(alpha=.2)
    fig.suptitle('Boundary investigation: before and validation-selected model')
    fig.tight_layout()
    fig.savefig(output/'comparison.png',dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2,2,figsize=(12,7))
    for row, side in enumerate(('left','right')):
        for j, name in enumerate(('sin','cos')):
            ax=axes[row,j]
            inputs=archive['zoom_'+side+'_t']
            ax.plot(inputs,archive['zoom_'+side+'_y'][:,j],color='black',linestyle='--',label='Exact target')
            ax.plot(inputs,archive['Day 4 source_zoom_'+side][:,j],color='#888888',label='Before')
            ax.plot(inputs,archive[chosen+'_zoom_'+side][:,j],color='#1668bb',label='After')
            ax.set(xlabel='t',ylabel='Prediction',title=side+' boundary: '+name)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.suptitle('Dense boundary diagnostics include exact endpoints and logarithmic near-endpoint spacing')
    fig.tight_layout()
    fig.savefig(output/'boundary_zoom.png',dpi=150)
    plt.close(fig)


def main(verify=False):
    torch.set_num_threads(1)
    output=ROOT/'evaluation'
    if not verify and output.exists():
        raise FileExistsError('Evaluation already exists; use --verify')
    report, archive=calculate()
    if verify:
        assert report == json.loads((output/'metrics.json').read_text())
        with np.load(output/'predictions.npz') as frozen:
            assert set(frozen.files)==set(archive)
            for key,value in archive.items():
                np.testing.assert_allclose(frozen[key],value,atol=1e-13,rtol=0)
        with (output/'trials.csv').open() as stream:
            rows=list(csv.DictReader(stream))
        assert len(rows)==len(report['trials'])
        for row,record in zip(rows,report['trials']):
            for key,value in record.items():
                if isinstance(value,bool):
                    assert (row[key]=='True')==value
                elif isinstance(value,(int,float)):
                    assert float(row[key])==value
                else:
                    assert row[key]==value
        audit={'status':'PASS','trials_verified':len(report['trials']),
               'history_rows_verified':sum(r['additional_updates']+1 for r in report['trials']),
               'optimizer_start_best_final_counters_verified':True,'fresh_and_resumed_starts_verified':True,
               'validation_guard_and_selection_verified':True,'csv_rows_verified':len(rows),'stratified_holdout_disjoint':True,
               'prediction_values_recomputed':sum(value.size for key,value in archive.items() if any(key.startswith(name+'_') for name in report['metrics'])),
               'qpu_jobs':0}
        (output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        print(json.dumps(audit,indent=2))
        return
    output.mkdir(parents=True)
    (output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez(output/'predictions.npz',**archive)
    with (output/'trials.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(report['trials'][0]))
        writer.writeheader()
        writer.writerows(report['trials'])
    plots(output,report,archive)
    lines=['# Boundary-study review','','All models use our original two-qubit Rot/CNOT circuit family. No output clipping, analytic boundary replacement, or removed model.', '',
           '| Trial | Layers | Input scale | Start | Eligible | Selected update | Validation score | Endpoint maximum error |',
           '|---|---:|---:|---|---|---:|---:|---:|']
    for row in report['trials']:
        lines.append(f'| {row["name"]} | {row["layers"]} | {row["input_scale"]} | {row["start_mode"]} @ {row["source_step"]} | {row["eligible"]} | {row["selected_step"]} | {row["selected_score"]:.8g} | {row["selected_endpoint_max_error"]:.8g} |')
    chosen=report['selection']['name']
    lines+=['',f'Selected by balanced validation score with the original-validation non-degradation guard: **{chosen}**. All checkpoints and this selection were frozen before the new holdout was read.','',
            '| Model / region | Output | MAE | MSE | Maximum error |','|---|---|---:|---:|---:|']
    for model_name in ('Day 4 source',chosen):
        for region in ('new_interior','new_left_boundary','new_right_boundary','new','endpoint','grid'):
            for name,values in report['metrics'][model_name][region].items():
                lines.append(f'| {model_name} / {region} | {name} | {values["mae"]:.8g} | {values["mse"]:.8g} | {values["max_absolute_error"]:.8g} |')
    lines+=['','Provisional goals: '+json.dumps(report['goals']), '',
            'The 512-input holdout is stratified: 256 interior, 128 left collar, 128 right collar. Its aggregate oversamples boundaries and is not uniform-domain risk. It was not used for stopping, checkpoint selection, or tuning. It is now exposed. Original 24-input test metrics are available separately in metrics.json.', '',
            'Exact endpoints are supervised training anchors for augmented trials; endpoint error is not independent test evidence. Region holdout errors evaluate generalization near those constraints. The 1,001-point grid and finer boundary zooms are diagnostics, not proof of uniform error bounds.', '',
            'Four trials resume both weights and Adam state from update 250; their 750 extra updates are independent branches. Two five-layer trials start fresh with 1,500 updates and zero optimizer moments. Two refinements then resume the five-layer parent at update 1,500 and add 1,500 updates, with unchanged encoder, architecture, and training data. Their guard is tightened to their parent validation loss. Encoder scale 0.9 changes the feature map; it is not a fitted output multiplier. Fresh architecture trials must not be described as unchanged-model continuation.', '',
            'The boundary runner uses the explicit schedule and the recorded learning_rate history. An inherited legacy learning_rate field in raw configurations is not used by this runner. No trial uses the independent holdout for choosing a schedule.', '',
            'One initialization seed per follow-up architecture was studied. No hardware, finite-shot, advantage, or percentage-accuracy claim. Rejected approaches and final states are retained. Inspect the before/after figures and each run’s history.png for optimization trade-offs.']
    (output/'REVIEW.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify',action='store_true')
    main(parser.parse_args().verify)
