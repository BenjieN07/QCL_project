"""Audit saved real-device pilot counts and plot them against the frozen model."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from .hardware import z_outputs
from .boundary_report import load_model
from .data import targets
from .evaluate import metrics


def main(show_window=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path('results/hardware/real_pilot')
    report=json.loads((root/'results.json').read_text())
    receipt=json.loads((root/'job.json').read_text())
    plan=json.loads((root/'plan.json').read_text())
    assert report['status']=='DONE' and report['job_id']==receipt['job_id']
    assert report['backend']==receipt['backend']==plan['backend']
    registry=json.loads(Path('results/current_model.json').read_text())
    cp=Path(registry['run'])/'best.pt'
    assert hashlib.sha256(cp.read_bytes()).hexdigest()==plan['checkpoint_sha256']==receipt['checkpoint_sha256']
    assert hashlib.sha256((root/'submitted_circuits.qpy').read_bytes()).hexdigest()==plan['compiled_circuit_sha256']
    model=load_model(torch.load(cp,weights_only=True))
    rows=report['rows']; ts=np.array([r['t'] for r in rows])
    np.testing.assert_array_equal(ts,[-1,0,1])
    with torch.no_grad():
        exact=model(torch.tensor(ts,dtype=torch.float64)).numpy()
    hardware=[]; errors=[]
    for index,row in enumerate(rows):
        counts=row['counts']; shots=sum(counts.values()); assert shots==1024
        assert all(len(k)==2 and set(k)<=set('01') and isinstance(v,int) and v>=0 for k,v in counts.items())
        prediction=z_outputs(np.array([counts.get(format(i,'02b'),0)/shots for i in range(4)]))
        se=np.sqrt((1-prediction**2)/(shots-1))
        for actual,saved in ((prediction,row['hardware_prediction']),(exact[index],row['exact_qcl']),
                             (targets(ts)[index],row['numpy_target']),(prediction-exact[index],row['hardware_minus_exact']),
                             (prediction-targets(ts)[index],row['hardware_minus_target']),(se,row['measurement_standard_error'])):
            np.testing.assert_allclose(actual,saved,atol=1e-12,rtol=0)
        hardware.append(prediction); errors.append(se)
    hardware=np.array(hardware); errors=np.array(errors)
    with np.load('results/hardware/shot_samples.npz') as sampled:
        indices=[0,4,8]
        np.testing.assert_array_equal(sampled['t'][indices],ts)
        np.testing.assert_allclose(sampled['exact'][indices],exact,atol=1e-12,rtol=0)
        ideal_samples=sampled['samples_1024'][:,indices,:]
    ideal_sd=np.sqrt((1-exact**2)/1024)
    scores={'hardware_vs_target':metrics(hardware,targets(ts)),'hardware_vs_exact_qcl':metrics(hardware,exact),
            'exact_qcl_vs_target':metrics(exact,targets(ts))}
    grid=np.linspace(-1,1,401)
    with torch.no_grad(): curve=model(torch.tensor(grid,dtype=torch.float64)).numpy()
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for j,name in enumerate(('Sine','Cosine')):
        axes[0,j].plot(grid,targets(grid)[:,j],'k--',label='NumPy target')
        axes[0,j].plot(grid,curve[:,j],label='Frozen exact QCL simulator')
        axes[0,j].errorbar(ts,hardware[:,j],yerr=errors[:,j],fmt='o',capsize=4,label='Real hardware ± 1 sampling SE')
        axes[1,j].errorbar(ts,hardware[:,j]-exact[:,j],yerr=errors[:,j],fmt='o',capsize=4,label='Hardware − exact QCL')
        axes[1,j].errorbar(ts,np.zeros(3),yerr=ideal_sd[:,j],fmt='s',color='gray',alpha=.6,capsize=6,label='Ideal 1024-shot ± 1 SD')
        axes[1,j].axhline(0,color='gray',linewidth=.8)
        axes[0,j].set(title=name,ylabel='Output'); axes[1,j].set(xlabel='t',ylabel='Deviation from simulator')
        for ax in axes[:,j]:
            ax.set_xlim(-1.08,1.08); ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.suptitle(f'Real-qubit pilot: {report["backend"]}, 1024 shots/input; frozen update {plan["model_update"]}')
    fig.tight_layout(); fig.savefig(root/'comparison.png',dpi=150)
    audit={'passed':True,'job_id':report['job_id'],'counts_total':3072,'inputs':3,'training_updates_on_hardware':0,
           'metrics':scores,'usage':report['usage'],'usage_after':report['usage_after'],
           'matched_ideal_sampling':{'shots':1024,'repeats':50,'inputs':ts.tolist(),
                                     'mean':ideal_samples.mean(axis=0).tolist(),
                                     'empirical_sd':ideal_samples.std(axis=0,ddof=1).tolist(),
                                     'theoretical_sd':ideal_sd.tolist()}}
    (root/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    lines=['# Real hardware pilot results','',f'Job `{report["job_id"]}` completed on **{report["backend"]}**. Physical qubits: {plan["physical_qubits_by_output"][0]}. Three inputs, 1024 shots each, frozen model update {plan["model_update"]}. No hardware training updates.','',
           '| t | Hardware sine | Exact QCL sine | Hardware cosine | Exact QCL cosine |','|---|---|---|---|---|']
    for row in rows:
        lines.append(f'| {row["t"]:g} | {row["hardware_prediction"][0]:.6f} | {row["exact_qcl"][0]:.6f} | {row["hardware_prediction"][1]:.6f} | {row["exact_qcl"][1]:.6f} |')
    lines+=['',f'Provider-reported usage: `{report["usage"]}`.',
            f'Remaining allowance after the job: {report["usage_after"].get("usage_remaining_seconds","unavailable")} seconds.','',
            'Errors relative to exact QCL combine finite-shot fluctuations and device effects; one run cannot separate these contributions. Error bars show sampling standard error only, not total device uncertainty. Three selected diagnostic inputs do not establish accuracy over the full domain. The other six proposed inputs were not submitted.','',
            'Matched ideal-shot reference: the previously saved 50 repeats at 1024 shots use the same frozen model and t=-1,0,1. Their means and standard deviations are included in audit.json. Gray bars show the theoretical one-standard-deviation ideal sampling range around zero deviation; colored bars show the hardware estimate and its sampling standard error.','',
            'Saved counts, receipt, compiled circuits, checkpoint fingerprints and reconstructed metrics are retained here. The audit checks stored results without contacting IBM or submitting another job.','']
    (root/'REPORT.md').write_text('\n'.join(lines))
    print('\n'.join(lines))
    if show_window:
        from .display import show_figure
        show_figure(fig,'QCL real hardware comparison')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--show',action='store_true',help='Open the comparison in a native Python window')
    args=parser.parse_args()
    torch.set_num_threads(1)
    main(args.show)
