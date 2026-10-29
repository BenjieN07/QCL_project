"""Offline export and shot sampling of the existing circuit; no remote execution."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from qiskit import QuantumCircuit, qpy, transpile
from qiskit.quantum_info import Statevector
from .model import QCLModel
from .data import targets


def export_circuit(weights, t, scale=.9, measured=False):
    if not np.isfinite(t) or abs(t)>1:
        raise ValueError('Input must be finite and in [-1,1]')
    circuit=QuantumCircuit(2)
    encoded=scale*t
    for layer in weights:
        for wire in range(2):
            circuit.ry(float(np.arcsin(encoded)),wire)
            circuit.rz(float(np.arccos(encoded**2)),wire)
            # PennyLane Rot(phi,theta,omega) applies these three gates in order.
            circuit.rz(float(layer[wire,0]),wire)
            circuit.ry(float(layer[wire,1]),wire)
            circuit.rz(float(layer[wire,2]),wire)
        circuit.cx(0,1)
    if measured:
        circuit.measure_all()
    return circuit


def z_outputs(probabilities):
    """Qiskit strings are q1 q0: rightmost bit is the sine output."""
    return np.asarray(probabilities) @ np.array([[1,1],[-1,1],[1,-1],[-1,-1]])


def benchmark():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    registry=json.loads(Path('results/current_model.json').read_text())
    checkpoint_path=Path(registry['run'])/'best.pt'
    checkpoint=torch.load(checkpoint_path,weights_only=True)
    model=QCLModel(5,41,.9); model.load_state_dict(checkpoint['model_state_dict'])
    weights=model.weights.detach().numpy()
    ts=np.linspace(-1,1,9)
    with torch.no_grad():
        exact=model(torch.tensor(ts,dtype=torch.float64)).numpy()
    circuits=[export_circuit(weights,t) for t in ts]
    probabilities=np.array([Statevector.from_instruction(c).probabilities() for c in circuits])
    exported=z_outputs(probabilities)
    np.testing.assert_allclose(exported,exact,rtol=0,atol=1e-12)
    compiled=transpile(circuits,basis_gates=['rz','sx','x','cx'],coupling_map=[[0,1],[1,0]],
                       initial_layout=[0,1],optimization_level=3,seed_transpiler=2026)
    for c,expected in zip(compiled,exact):
        np.testing.assert_allclose(z_outputs(Statevector.from_instruction(c).probabilities()),expected,atol=1e-12,rtol=0)
    root=Path('results/hardware'); root.mkdir(exist_ok=True)
    with (root/'circuits.qpy').open('wb') as stream:
        qpy.dump([export_circuit(weights,t,measured=True) for t in ts],stream)
    rng=np.random.default_rng(20261102)
    rows=[]; arrays={'t':ts,'exact':exact,'target':targets(ts),'probabilities':probabilities}
    for shots in (256,1024,4096):
        sampled=np.stack([z_outputs(rng.multinomial(shots,p,size=50)/shots) for p in probabilities],axis=1)
        arrays[f'samples_{shots}']=sampled
        rows.append({'shots':shots,'repeats':50,'rmse_from_exact':np.sqrt(np.mean((sampled-exact)**2,axis=(0,1))).tolist(),
                     'expected_rmse_from_exact':np.sqrt(np.mean(1-exact**2,axis=0)/shots).tolist(),
                     'rmse_from_target':np.sqrt(np.mean((sampled-targets(ts))**2,axis=(0,1))).tolist()})
    np.savez(root/'shot_samples.npz',**arrays)
    report={'checkpoint':str(checkpoint_path),'sha256':hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            'step':checkpoint['step'],'qpu_jobs':0,'real_backend_verified':False,
            'simulation':'Ideal joint measurement sampling; excludes device noise',
            'max_export_difference':float(np.max(abs(exported-exact))), 'shot_benchmarks':rows,
            'generic_compilation':[{'t':float(t),'depth':c.depth(),'operations':dict(c.count_ops())} for t,c in zip(ts,compiled)],
            'proposed_pilot':{'inputs':[-1,0,1],'shots_per_input':1024,'total_shots':3072,'submitted':False},
            'proposed_full_workload':{'circuits':9,'total_shots':9216,'submitted':False}}
    (root/'offline_report.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for i,name in enumerate(('Sine','Cosine')):
        axes[i].plot(ts,targets(ts)[:,i],label='NumPy reference')
        axes[i].plot(ts,exact[:,i],label='Exact QCL')
        samples=arrays['samples_1024'][:,:,i]
        axes[i].errorbar(ts,samples.mean(axis=0),yerr=samples.std(axis=0,ddof=1),fmt='o',capsize=3,label='Ideal 1024 shots: mean ± 1 SD')
        axes[i].set(title=name,xlabel='t',ylabel='Output'); axes[i].legend(fontsize=7)
    fig.suptitle('Offline shot benchmark — no hardware execution'); fig.tight_layout()
    fig.savefig(root/'shot_benchmark.png',dpi=150); plt.close(fig)
    print(json.dumps(report,indent=2))


def audit():
    root=Path('results/hardware')
    report=json.loads((root/'offline_report.json').read_text())
    assert report['qpu_jobs']==0 and report['real_backend_verified'] is False
    path=Path(report['checkpoint'])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==report['sha256']
    registry=json.loads(Path('results/current_model.json').read_text())
    assert path.resolve()==(Path(registry['run'])/'best.pt').resolve()
    checkpoint=torch.load(path,weights_only=True)
    model=QCLModel(5,41,.9); model.load_state_dict(checkpoint['model_state_dict'])
    with (root/'circuits.qpy').open('rb') as stream:
        circuits=qpy.load(stream)
    with np.load(root/'shot_samples.npz') as saved:
        ts=saved['t']; assert len(circuits)==len(ts)==9
        np.testing.assert_allclose(saved['target'],targets(ts),atol=0,rtol=0)
        with torch.no_grad():
            exact=model(torch.tensor(ts,dtype=torch.float64)).numpy()
        np.testing.assert_allclose(saved['exact'],exact,atol=1e-12,rtol=0)
        rng=np.random.default_rng(20261102)
        probabilities=[]
        for circuit,expected in zip(circuits,exact):
            measures=[item for item in circuit.data if item.operation.name=='measure']
            assert [(circuit.find_bit(item.qubits[0]).index,circuit.find_bit(item.clbits[0]).index) for item in measures]==[(0,0),(1,1)]
            p=Statevector.from_instruction(circuit.remove_final_measurements(inplace=False)).probabilities()
            probabilities.append(p)
            np.testing.assert_allclose(z_outputs(p),expected,atol=1e-12,rtol=0)
        np.testing.assert_allclose(saved['probabilities'],probabilities,atol=1e-12,rtol=0)
        for row in report['shot_benchmarks']:
            shots=row['shots']
            sampled=np.stack([z_outputs(rng.multinomial(shots,p,size=50)/shots) for p in saved['probabilities']],axis=1)
            np.testing.assert_array_equal(sampled,saved[f'samples_{shots}'])
            np.testing.assert_allclose(np.sqrt(np.mean((sampled-exact)**2,axis=(0,1))),row['rmse_from_exact'],atol=1e-12,rtol=0)
            np.testing.assert_allclose(np.sqrt(np.mean((sampled-targets(ts))**2,axis=(0,1))),row['rmse_from_target'],atol=1e-12,rtol=0)
    (root/'audit.json').write_text(json.dumps({'passed':True,'circuits':9,'sampling_repeats_per_budget':50,'budgets':[256,1024,4096],'qpu_jobs':0},indent=2)+'\n')
    print('Hardware artifact audit passed: circuit mapping, exact outputs, deterministic sampling and metrics.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    audit() if args.audit_only else benchmark()
