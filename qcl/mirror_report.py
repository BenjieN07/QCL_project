"""Audit frozen mirrored-training outcomes, then evaluate a fresh stratified holdout."""
from qcl.checkpoints import load_checkpoint
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from .boundary_report import load_model,optimizer_counter
from .data import targets
from .evaluate import metrics
from .mirror_study import datasets,evaluate_values


ROOT=Path('results/mirror')


def calculate():
    # Selection and every checkpoint must be committed before exposing the new holdout.
    selection=json.loads((ROOT/'selection.json').read_text())
    tracked=['results/mirror/selection.json','results/current_model.json']
    tracked += [r['run']+'/best.pt' for r in selection['candidates']]
    for path in tracked:
        subprocess.run(['git','ls-files','--error-unmatch',path],check=True,capture_output=True)
        subprocess.run(['git','diff','--exit-code','HEAD','--',path],check=True,capture_output=True)
    freeze_commit=subprocess.check_output(['git','log','-1','--format=%H','--','results/mirror/selection.json'],text=True).strip()
    spec=json.loads(Path('configs/mirror_study.json').read_text())
    data=datasets(); histories={}; checkpoints={}; audit=[]
    with np.load(ROOT/'training_data.npz') as saved:
        for key,value in data.items():
            np.testing.assert_array_equal(saved[key],value)
    source=load_checkpoint(Path(spec['source_run'])/'best.pt',weights_only=True)
    checkpoints['before']=source
    for record in selection['candidates']:
        run=Path(record['run']); config=json.loads((run/'config.json').read_text())
        start,best,final=[load_checkpoint(run/(s+'.pt'),weights_only=True) for s in ('start','best','final')]
        summary=json.loads((run/'summary.json').read_text())
        assert all(cp['config']==config for cp in (start,best,final)) and summary['config']==config
        assert summary['selected_step']==best['step'] and summary['selected_score']==record['score']
        assert optimizer_counter(start)==start['step']==3000
        assert optimizer_counter(final)==final['step']==4500
        assert optimizer_counter(best)==best['step']==record['selected_step']
        torch.testing.assert_close(start['model_state_dict']['weights'],source['model_state_dict']['weights'],atol=0,rtol=0)
        a=next(iter(start['optimizer_state_dict']['state'].values())); b=next(iter(source['optimizer_state_dict']['state'].values()))
        for key in ('step','exp_avg','exp_avg_sq'):
            torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)
        assert config['source_checkpoint_sha256']==hashlib.sha256((Path(spec['source_run'])/'best.pt').read_bytes()).hexdigest()
        with (run/'history.csv').open() as stream:
            rows=list(csv.DictReader(stream))
        assert len(rows)==1501 and [int(r['total_step']) for r in rows]==list(range(3000,4501))
        for row in rows:
            eligible=all(float(row[key])<=guard+1e-12 for key,guard in zip(('original_sin_mse','original_cos_mse'),config['guard_original_validation_mse'])) and max(float(row['collar_max_error']),float(row['endpoint_max_error']))<=.05
            assert eligible==(row['eligible']=='True')
            assert abs(float(row['score'])-.5*(float(row['symmetric_validation_loss'])+float(row['collar_validation_loss'])))<1e-12
            weight=config['train_boundary_weight']
            assert abs(float(row['training_objective'])-((1-weight)*float(row['main_training_loss'])+weight*float(row['anchor_training_loss'])))<1e-12
        chosen=min((r for r in rows if r['eligible']=='True'),key=lambda r:float(r['score']))
        assert int(chosen['total_step'])==best['step'] and abs(float(chosen['score'])-record['score'])<1e-12
        for cp,row in ((start,rows[0]),(best,chosen),(final,rows[-1])):
            losses,mses,collar,endpoint=evaluate_values(load_model(cp),data,'mirrored' if config['mirror_training'] else 'train')
            assert abs(.5*(losses['symmetric_validation']+losses['collar'])-float(row['score']))<1e-12
            np.testing.assert_allclose(mses,[float(row['original_sin_mse']),float(row['original_cos_mse'])],rtol=0,atol=1e-12)
            assert abs(collar-float(row['collar_max_error']))<1e-12 and abs(endpoint-float(row['endpoint_max_error']))<1e-12
        name=record['name']; histories[name]=rows; checkpoints[name]=best
        audit.append({'name':name,'history_rows':len(rows),'optimizer_start':3000,'optimizer_final':4500,'selected_step':best['step'],
                      'checkpoint_sha256':hashlib.sha256((run/'best.pt').read_bytes()).hexdigest()})
    expected=min((r for r in selection['candidates'] if r['eligible']),key=lambda r:r['score'])
    assert expected['name']==selection['selected']['name']
    rng=np.random.default_rng(spec['new_holdout_seed'])
    fresh=np.concatenate((rng.uniform(-.9,.9,256),rng.uniform(-1,-.9,128),rng.uniform(.9,1,128)))
    prior=np.random.default_rng(20261031)
    previous=np.concatenate((prior.uniform(-.9,.9,256),prior.uniform(-1,-.9,128),prior.uniform(.9,1,128)))
    exposed=np.concatenate([data[k] for k in ('mirrored_t','anchor_t','symmetric_validation_t','collar_t','test_t')]+[previous,np.random.default_rng(20261029).uniform(-1,1,128),np.random.default_rng(20261030).uniform(-1,1,256)])
    assert len(np.unique(fresh))==512 and not np.isin(fresh,exposed).any()
    grid=np.linspace(-1,1,1001); scores={}; archive={'fresh_t':fresh,'fresh_y':targets(fresh),'grid_t':grid,'grid_y':targets(grid)}
    for name,cp in checkpoints.items():
        model=load_model(cp)
        with torch.no_grad():
            pred=model(torch.tensor(fresh,dtype=torch.float64)).numpy()
            curve=model(torch.tensor(grid,dtype=torch.float64)).numpy()
        archive[name+'_fresh']=pred; archive[name+'_grid']=curve
        error=curve-targets(grid)
        scores[name]={region:metrics(pred[mask],targets(fresh[mask])) for region,mask in (
            ('interior',slice(0,256)),('left_boundary',slice(256,384)),('right_boundary',slice(384,512)))}
        scores[name]['grid']=metrics(curve,targets(grid))
        scores[name]['parity_max_error']=[float(abs(curve[:,0]+curve[::-1,0]).max()),float(abs(curve[:,1]-curve[::-1,1]).max())]
        scores[name]['largest_signed_error']=[{'t':float(grid[np.argmax(abs(error[:,j]))]),'error':float(error[np.argmax(abs(error[:,j])),j])} for j in (0,1)]
    report={'selection':selection,'freeze_commit':freeze_commit,'fresh_holdout_seed':spec['new_holdout_seed'],
            'fresh_holdout_counts':[256,128,128],'fresh_holdout_used_for_selection':False,'qpu_jobs':0,'audit':audit,'metrics':scores}
    return report,archive,histories


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report,archive,histories=calculate()
    chosen=report['selection']['selected']['name']
    root=ROOT/'evaluation'; root.mkdir(exist_ok=True)
    np.savez(root/'predictions.npz',**archive)
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    t=archive['grid_t']
    for j,name in enumerate(('Sine','Cosine')):
        axes[0,j].plot(t,archive['grid_y'][:,j],'k--',label='NumPy reference')
        for key,label in (('before','Before: update 3000'),(chosen,'After: '+chosen)):
            axes[0,j].plot(t,archive[key+'_grid'][:,j],label=label)
            axes[1,j].plot(t,archive[key+'_grid'][:,j]-archive['grid_y'][:,j],label=label)
        axes[0,j].set(title=name,ylabel='Output'); axes[1,j].set(xlabel='t',ylabel='QCL − reference')
        axes[1,j].axhline(0,color='gray',linewidth=.7)
        for ax in axes[:,j]:
            ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.suptitle('Same circuit: continued training with controlled coverage and weighting'); fig.tight_layout()
    fig.savefig(root/'comparison.png',dpi=150); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name,rows in histories.items():
        axes[0].plot([int(r['total_step']) for r in rows],[float(r['score']) for r in rows],label=name)
        axes[1].plot([int(r['total_step']) for r in rows],[float(r['endpoint_max_error']) for r in rows],label=name)
    axes[0].set(xlabel='Total updates per branch',ylabel='Validation selection score',yscale='log')
    axes[1].set(xlabel='Total updates per branch',ylabel='Endpoint maximum error'); axes[1].axhline(.05,color='gray',linestyle=':')
    for ax in axes: ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(root/'training_comparison.png',dpi=150); plt.close(fig)
    lines=['# Mirrored coverage results','',f'Validation-selected branch: **{chosen}**. All three branches continued the same source weights and Adam state from update 3000 to 4500. Selected checkpoints can precede update 4500.','',
           'Fresh stratified holdout: 256 interior, 128 left boundary, 128 right boundary; not a uniformly weighted domain score. No holdout-based selection.','',
           '| Model | Interior MAE sin / cos | Left MAE sin / cos | Right MAE sin / cos | Grid max sin / cos |','|---|---|---|---|---|']
    for name,values in report['metrics'].items():
        cells=[]
        for region,key in (('interior','mae'),('left_boundary','mae'),('right_boundary','mae'),('grid','max_absolute_error')):
            cells.append(' / '.join(f'{values[region][output][key]:.6f}' for output in ('sin','cos')))
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    lines+=['','Exact simulation only. The grid diagnoses visible residuals; it is not a proof of maximum error between grid points. See report.json for all branch outcomes, symmetry defects and checkpoint audit.','']
    (root/'REPORT.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__=='__main__':
    torch.set_num_threads(1)
    main()
