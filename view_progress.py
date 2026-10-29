"""Show model improvement across experiment milestones; never train or contact IBM."""
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
from qcl.checkpoints import load_checkpoint
from qcl.boundary_report import load_model
from qcl.data import make_data,targets
from qcl.display import show_figure
from qcl.evaluate import metrics
from qcl.model import QCLModel

ROOT=Path(__file__).resolve().parent


def build_progress(output,show_window=True):
    torch.set_num_threads(1)
    registry=json.loads((ROOT/'results/current_model.json').read_text())
    stages=[('Untrained',None),('One layer','results/baseline/run'),
            ('Depth study','results/depth_study/runs/depth3_seed41'),
            ('Five layers','results/boundary/runs/capacity_smooth'),
            ('Continued fit','results/boundary/runs/refine_constant'),('Mirrored tuning',registry['run'])]
    config=json.loads((ROOT/'configs/baseline.json').read_text())
    data=make_data(config); grid=np.linspace(-1,1,1001)
    records=[]; archive={'grid_t':grid,'target':targets(grid),'test_t':data['test_t'],'test_y':data['test_y']}
    for index,(label,run) in enumerate(stages):
        checkpoint=load_checkpoint(ROOT/run/'best.pt',weights_only=True) if run else None
        model=load_model(checkpoint) if checkpoint else QCLModel(1,7)
        with torch.no_grad():
            curve=model(torch.tensor(grid,dtype=torch.float64)).numpy()
            prediction=model(torch.tensor(data['test_t'],dtype=torch.float64)).numpy()
        score=metrics(prediction,data['test_y'])
        archive[f'stage_{index}_curve']=curve
        archive[f'stage_{index}_test']=prediction
        summary=json.loads((ROOT/run/'summary.json').read_text()) if run else {}
        records.append({'milestone':label,'run':run,'layers':len(model.weights),'selected_update':checkpoint['step'] if checkpoint else 0,
                        'sin_test_mse':score['sin']['mse'],'cos_test_mse':score['cos']['mse'],
                        'sin_test_mae':score['sin']['mae'],'cos_test_mae':score['cos']['mae'],
                        'recorded_run_seconds':summary.get('elapsed_seconds'),
                        'checkpoint_sha256':hashlib.sha256((ROOT/run/'best.pt').read_bytes()).hexdigest() if run else None})
    trials=[]
    groups=[('Depth','results/depth_study/runs'),('Boundary','results/boundary/runs'),('Mirrored','results/mirror/runs')]
    for group,folder in groups:
        for run in sorted((ROOT/folder).iterdir()):
            if not (run/'summary.json').exists(): continue
            checkpoint=load_checkpoint(run/'best.pt',weights_only=True); model=load_model(checkpoint)
            cfg=checkpoint['config']; summary=json.loads((run/'summary.json').read_text())
            with torch.no_grad(): prediction=model(torch.tensor(data['test_t'],dtype=torch.float64)).numpy()
            score=metrics(prediction,data['test_y'])
            trials.append({'group':group,'trial':run.name,'layers':cfg['layers'],'seed':cfg['parameter_seed'],
                           'input_scale':cfg.get('input_scale',1.),'start':cfg.get('start_mode','fresh'),
                           'updates_in_run':cfg.get('additional_updates',cfg.get('steps')),'selected_update':checkpoint['step'],
                           'boundary_weight':cfg.get('train_boundary_weight',.5 if cfg.get('augment') else 0.),
                           'schedule':json.dumps(cfg.get('schedule',[[0,cfg['learning_rate']]])),
                           'validation_eligible':summary.get('eligible',True),
                           'sin_test_mse':score['sin']['mse'],'cos_test_mse':score['cos']['mse'],
                           'loop_seconds':summary['elapsed_seconds']})
    output.mkdir(parents=True,exist_ok=True)
    for name,rows in (('milestones.csv',records),('experiments.csv',trials)):
        with (output/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    np.savez(output/'curves.npz',**archive)
    report={'milestones':records,'experiments':trials,'test_inputs':24,'test_used_for_selection':False,
            'milestones_are_not_elapsed_days':True,'training_updates':0,'qpu_jobs':0}
    (output/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(13,8.5))
    for j,name in enumerate(('Sine','Cosine')):
        axes[0,j].plot(grid,targets(grid)[:,j],'k--',label='NumPy reference')
        axes[0,j].plot(grid,archive['stage_0_curve'][:,j],color='gray',alpha=.7,label='Untrained one-layer circuit')
        axes[0,j].plot(grid,archive['stage_5_curve'][:,j],color='#1668bb',label='Current five-layer circuit')
        axes[0,j].set(title=name,xlabel='t',ylabel='Output'); axes[0,j].legend(fontsize=8); axes[0,j].grid(alpha=.2)
    positions=np.arange(len(records))
    for key,label in (('sin_test_mse','Sine'),('cos_test_mse','Cosine')):
        axes[1,0].plot(positions,[r[key] for r in records],marker='o',label=label)
    axes[1,0].set(yscale='log',ylabel='Same 24-input test MSE',xticks=positions,
                  xticklabels=[r['milestone'] for r in records],title='Improvement across experiment milestones')
    axes[1,0].tick_params(axis='x',labelrotation=20,labelsize=8)
    axes[1,0].legend(fontsize=8); axes[1,0].grid(alpha=.2)
    for name,label in (('continuation_control','Extra training'),('mirrored_training','Mirrored data'),('mirrored_rebalance','Mirrored + reweighted')):
        with (ROOT/'results/mirror/runs'/name/'history.csv').open() as stream: rows=list(csv.DictReader(stream))
        axes[1,1].plot([int(r['total_step']) for r in rows],[float(r['score']) for r in rows],label=label)
    axes[1,1].axvline(records[-1]['selected_update'],color='gray',linestyle=':',label='Selected checkpoint')
    axes[1,1].set(xlabel='Total updates on each continuation branch',ylabel='Validation score',title='Matched optimization comparison')
    axes[1,1].legend(fontsize=8); axes[1,1].grid(alpha=.2)
    fig.suptitle('Quantum circuit learning: measured development progress')
    fig.text(.5,.012,'Milestones compare different experiments, not elapsed days. Fresh architectures restart; continuation restores weights and Adam state.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,1)); fig.savefig(output/'progress.png',dpi=150)
    lines=['# Model improvement','', 'All rows use the same 24 reporting inputs. Models were selected using validation, never this comparison.','',
           '| Milestone | Layers | Selected update | Sine MSE | Cosine MSE |','|---|---:|---:|---:|---:|']
    for row in records:
        lines.append(f'| {row["milestone"]} | {row["layers"]} | {row["selected_update"]} | {row["sin_test_mse"]:.6g} | {row["cos_test_mse"]:.6g} |')
    lines+=['','The five-layer model started fresh. Its later continuations restored both weights and Adam moments. Independent branches are not additional sequential updates on one model.',
            '', 'experiments.csv compares all 20 depth, boundary and mirrored-training runs, including rejected checkpoints. Loop times are archived measurements and are not a controlled cross-run speed test.',
            '', 'Hardware evidence is separate: three frozen-model inputs on physical qubits, not hardware training or full-domain validation. See ../hardware/real_pilot/REPORT.md.','']
    (output/'REPORT.md').write_text('\n'.join(lines))
    print('\n'.join(lines)); print(f'Progress graph: {output / "progress.png"}')
    if show_window: show_figure(fig,'QCL model development progress')
    plt.close(fig)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-open',action='store_true',help='Save results without opening a window')
    parser.add_argument('--output',type=Path,default=ROOT/'results/progress')
    args=parser.parse_args(); build_progress(args.output.resolve(),not args.no_open)
