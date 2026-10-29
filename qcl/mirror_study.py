"""Matched continuation/data-augmentation/loss-weight study; original circuit only."""
import argparse
from contextlib import redirect_stdout
import copy
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from .data import targets
from .boundary_study import restore_start
from .model import QCLModel
from .training import regression_loss

ROOT=Path('results/mirror')
SPEC=Path('configs/mirror_study.json')


def datasets():
    with np.load('results/boundary/training_data.npz') as saved:
        data={key:saved[key] for key in saved.files}
    data['mirrored_t']=np.unique(np.concatenate((data['train_t'],-data['train_t'])))
    data['mirrored_y']=targets(data['mirrored_t'])
    data['symmetric_validation_t']=np.unique(np.concatenate((data['validation_t'],-data['validation_t'])))
    data['symmetric_validation_y']=targets(data['symmetric_validation_t'])
    assert len(data['mirrored_t'])==96 and len(data['symmetric_validation_t'])==32
    assert not np.isin(np.concatenate((data['mirrored_t'],data['anchor_t'])),np.concatenate((data['symmetric_validation_t'],data['collar_t'],data['test_t']))).any()
    return data


def evaluate_values(model,data,main):
    names=(main,'anchor','validation','symmetric_validation','collar','endpoint')
    sizes=[len(data[name+'_t']) for name in names]
    with torch.no_grad():
        prediction=model(torch.tensor(np.concatenate([data[name+'_t'] for name in names]),dtype=torch.float64))
        chunks=dict(zip(names,torch.split(prediction,sizes)))
        residuals={name:chunks[name]-torch.tensor(data[name+'_y'],dtype=torch.float64) for name in names}
        losses={name:float(value.square().sum(dim=1).mean()) for name,value in residuals.items()}
        original_mse=residuals['validation'].square().mean(dim=0).tolist()
        collar_max=float(residuals['collar'].abs().max())
        endpoint_max=float(residuals['endpoint'].abs().max())
    return losses,original_mse,collar_max,endpoint_max


def prepare():
    spec=json.loads(SPEC.read_text())
    source_path=Path(spec['source_run'])/'best.pt'
    source=torch.load(source_path,weights_only=True)
    data=datasets()
    model=QCLModel(source['config']['layers'],source['config']['parameter_seed'],source['config']['input_scale'])
    model.load_state_dict(source['model_state_dict'])
    _,guards,collar_max,endpoint_max=evaluate_values(model,data,'train')
    assert max(collar_max,endpoint_max)<=spec['boundary_max_error_guard']
    ROOT.mkdir(parents=True,exist_ok=True)
    path=ROOT/'training_data.npz'
    if path.exists():
        with np.load(path) as saved:
            for key,value in data.items():
                np.testing.assert_array_equal(saved[key],value)
    else:
        np.savez(path,**data)
    jobs=[]
    for trial in spec['trials']:
        config=dict(source['config'])
        config.update(trial,phase='Mirror study: '+trial['name'],source_run=spec['source_run'],
                      source_step=source['step'],start_mode='resume',additional_updates=spec['additional_updates'],
                      steps=source['step']+spec['additional_updates'],learning_rate=spec['learning_rate'],
                      schedule=[[0,spec['learning_rate']]],guard_original_validation_mse=guards,
                      boundary_max_error_guard=spec['boundary_max_error_guard'],
                      augmented_training_boundary_weight=trial['train_boundary_weight'],
                      source_checkpoint_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest())
        config_path=Path('configs/mirror')/(trial['name']+'.json')
        config_path.parent.mkdir(parents=True,exist_ok=True)
        if config_path.exists():
            assert json.loads(config_path.read_text())==config
        else:
            config_path.write_text(json.dumps(config,indent=2)+'\n')
        jobs.append((config_path,ROOT/'runs'/trial['name']))
    return spec,jobs


def train(config_path,output):
    config=json.loads(config_path.read_text())
    source=torch.load(Path(config['source_run'])/'best.pt',weights_only=True)
    model,optimizer=restore_start(config,source)
    for group in optimizer.param_groups:
        group['lr']=config['learning_rate']
    data=datasets()
    main='mirrored' if config['mirror_training'] else 'train'
    main_size=len(data[main+'_t'])
    xs=torch.tensor(np.concatenate((data[main+'_t'],data['anchor_t'])),dtype=torch.float64)
    main_y=torch.tensor(data[main+'_y'],dtype=torch.float64)
    anchor_y=torch.tensor(data['anchor_y'],dtype=torch.float64)
    history,best,fallback=[],None,None
    started=time.perf_counter()
    def snapshot(step,score,losses,mses,collar,endpoint):
        return {'step':step,'config':config,'model_state_dict':copy.deepcopy(model.state_dict()),
                'optimizer_state_dict':copy.deepcopy(optimizer.state_dict()),'selection_score':score,
                'train_loss':losses[main],'validation_loss':losses['validation'],
                'symmetric_validation_loss':losses['symmetric_validation'],'collar_validation_loss':losses['collar'],
                'original_validation_mse':mses,'collar_max_error':collar,'endpoint_max_error':endpoint}
    torch.save({'step':source['step'],'config':config,'model_state_dict':copy.deepcopy(model.state_dict()),
                'optimizer_state_dict':copy.deepcopy(optimizer.state_dict())},output/'start.pt')
    for update in range(config['additional_updates']+1):
        losses,mses,collar,endpoint=evaluate_values(model,data,main)
        score=.5*(losses['symmetric_validation']+losses['collar'])
        eligible=all(value<=guard+1e-12 for value,guard in zip(mses,config['guard_original_validation_mse'])) and max(collar,endpoint)<=config['boundary_max_error_guard']
        step=source['step']+update
        if fallback is None or score<fallback['selection_score']:
            fallback=snapshot(step,score,losses,mses,collar,endpoint)
        if eligible and (best is None or score<best['selection_score']):
            best=snapshot(step,score,losses,mses,collar,endpoint)
        weight=config['train_boundary_weight']
        row={'additional_step':update,'total_step':step,'main_training_loss':losses[main],
             'anchor_training_loss':losses['anchor'],'training_objective':(1-weight)*losses[main]+weight*losses['anchor'],
             'original_validation_loss':losses['validation'],'original_sin_mse':mses[0],'original_cos_mse':mses[1],
             'symmetric_validation_loss':losses['symmetric_validation'],'collar_validation_loss':losses['collar'],
             'score':score,'eligible':eligible,'collar_max_error':collar,'endpoint_max_error':endpoint,'gradient_norm':None}
        if update<config['additional_updates']:
            optimizer.zero_grad(set_to_none=True)
            predicted=model(xs)
            objective=(1-weight)*regression_loss(predicted[:main_size],main_y)+weight*regression_loss(predicted[main_size:],anchor_y)
            objective.backward()
            assert model.weights.grad is not None and torch.isfinite(model.weights.grad).all()
            row['gradient_norm']=float(model.weights.grad.norm())
            optimizer.step()
        history.append(row)
        if update%250==0:
            print(f'{config["name"]} update {step}: score={score:.8f}, eligible={eligible}, endpoint={endpoint:.5f}',flush=True)
    eligible=best is not None
    best=best if eligible else fallback
    torch.save(best,output/'best.pt')
    torch.save(snapshot(config['steps'],score,losses,mses,collar,endpoint),output/'final.pt')
    assert int(optimizer.state[model.weights]['step'].item())==config['steps']
    with (output/'history.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    steps=[r['total_step'] for r in history]
    for key,label in (('original_validation_loss','Original validation'),('symmetric_validation_loss','Symmetric validation'),('collar_validation_loss','Collar validation')):
        axes[0].plot(steps,[r[key] for r in history],label=label)
    axes[0].set(xlabel='Total updates on this branch',ylabel='Joint squared error',yscale='log'); axes[0].legend(fontsize=8)
    axes[1].plot(steps,[r['endpoint_max_error'] for r in history],label='Endpoints')
    axes[1].plot(steps,[r['collar_max_error'] for r in history],label='Collar validation')
    axes[1].axhline(.05,color='gray',linestyle=':'); axes[1].set(xlabel='Total updates',ylabel='Maximum error'); axes[1].legend(fontsize=8)
    fig.suptitle(config['name']); fig.tight_layout(); fig.savefig(output/'history.png',dpi=150); plt.close(fig)
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    summary={'config':config,'eligible':eligible,'selected_step':best['step'],'selected_score':best['selection_score'],
             'selected_original_validation_mse':best['original_validation_mse'],'selected_endpoint_max_error':best['endpoint_max_error'],
             'selected_collar_max_error':best['collar_max_error'],'optimizer_final_counter':config['steps'],
             'elapsed_seconds':time.perf_counter()-started,'versions':{p:importlib.metadata.version(p) for p in ('pennylane','torch','numpy','matplotlib')},
             'source_sha256':{str(p).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('qcl').glob('*.py'))},
             'new_holdout_read':False,'qpu_jobs':0}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


def select(jobs):
    candidates=[]
    for path,output in jobs:
        result=json.loads((output/'summary.json').read_text()); assert result['config']==json.loads(path.read_text())
        candidates.append({'run':str(output).replace('\\','/'),'name':result['config']['name'],'eligible':result['eligible'],
                           'score':result['selected_score'],'selected_step':result['selected_step']})
    best=min([r for r in candidates if r['eligible']],key=lambda r:r['score'])
    registry=dict(best,model='original',layers=5,parameter_seed=41,input_scale=.9,
                  selection='minimum symmetric/collar validation score, guarded by per-output original validation MSE and boundary errors',
                  test_or_fresh_holdout_used_for_selection=False,new_holdout_read_at_registration=False)
    (ROOT/'selection.json').write_text(json.dumps({'candidates':candidates,'selected':registry},indent=2)+'\n')
    Path('results/current_model.json').write_text(json.dumps(registry,indent=2)+'\n'); print(json.dumps(registry,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--prepare-only',action='store_true'); parser.add_argument('--select-only',action='store_true')
    args=parser.parse_args(); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    _,jobs=prepare()
    if args.prepare_only:
        print('Prepared three fixed branches; no training or new holdout read.')
    elif args.select_only:
        select(jobs)
    else:
        for path,output in jobs:
            if (output/'summary.json').exists():
                assert json.loads((output/'summary.json').read_text())['config']==json.loads(path.read_text()); continue
            output.mkdir(parents=True,exist_ok=True); print(f'Starting {output}',flush=True)
            with (output/'training.log').open('w') as log,redirect_stdout(log):
                train(path,output)
            result=json.loads((output/'summary.json').read_text()); print(f'Completed {output}: score={result["selected_score"]:.8f}, eligible={result["eligible"]}',flush=True)
        print('Freeze validation selection separately before independent evaluation.',flush=True)
