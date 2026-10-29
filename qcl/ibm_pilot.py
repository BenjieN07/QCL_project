"""Prepare, submit once, or retrieve the explicitly authorized three-input IBM pilot."""
import argparse
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
from datetime import datetime,timezone
import warnings
import numpy as np
from qiskit import qpy,QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime import QiskitRuntimeService,SamplerV2
from .hardware import z_outputs
from ibm_preflight import numeric_usage

ROOT=Path('results/hardware/real_pilot')
BACKEND='ibm_fez'
SHOTS=1024
CAP=60


def save(name,value):
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/name).write_text(json.dumps(value,indent=2)+'\n')


def connect():
    root=Path(__file__).resolve().parents[1]
    token=(root/'api.txt').read_text(encoding='utf-8-sig').strip()
    instance=(root/'crn.txt').read_text(encoding='utf-8-sig').strip()
    if not token or token.startswith('ApiKey-') or not instance.startswith('crn:'):
        raise ValueError('Invalid local credential format')
    return QiskitRuntimeService(channel='ibm_quantum_platform',token=token,instance=instance)


def prepare(service):
    manifest=json.loads(Path('results/hardware/pilot_manifest.json').read_text())
    registry=json.loads(Path('results/current_model.json').read_text())
    assert hashlib.sha256((Path(registry['run'])/'best.pt').read_bytes()).hexdigest()==manifest['checkpoint_sha256']
    assert hashlib.sha256(Path(manifest['circuit_file']).read_bytes()).hexdigest()==manifest['circuit_sha256']
    preflight=json.loads(Path('.ibm-private/preflight.json').read_text())
    assert preflight['backends'][0]['name']==BACKEND
    assert preflight['pilot']['checkpoint_sha256']==manifest['checkpoint_sha256']
    backend=service.backend(BACKEND)
    if not backend.status().operational or backend.configuration().simulator:
        raise ValueError('Real backend unavailable')
    usage=numeric_usage(service.usage())
    remaining=usage.get('usage_remaining_seconds')
    if remaining is None or remaining<CAP:
        raise ValueError('Insufficient verified allowance for execution cap')
    with Path('.ibm-private/target_circuits.qpy').open('rb') as stream:
        circuits=qpy.load(stream)
    assert len(circuits)==3
    durations=[]; layouts=[]
    for circuit,point in zip(circuits,manifest['points']):
        assert circuit.num_clbits==2
        assert circuit.count_ops().get('measure')==2
        duration=0.
        measured=[]
        for item in circuit.data:
            name=item.operation.name
            if name=='barrier': continue
            qubits=tuple(circuit.find_bit(q).index for q in item.qubits)
            prop=backend.target[name][qubits]
            if prop is None or prop.duration is None:
                raise ValueError('Missing target timing information')
            duration+=prop.duration
            if name=='measure':
                measured.append((qubits[0],circuit.find_bit(item.clbits[0]).index))
        assert sorted(bit for _,bit in measured)==[0,1]
        physical=[wire for wire,bit in sorted(measured,key=lambda pair:pair[1])]
        compact=QuantumCircuit(2)
        for item in circuit.data:
            if item.operation.name in ('measure','barrier'): continue
            wires=[physical.index(circuit.find_bit(q).index) for q in item.qubits]
            compact.append(item.operation,wires)
        # Reconstruct only the two active physical wires; never simulate 156 qubits.
        np.testing.assert_allclose(z_outputs(Statevector.from_instruction(compact).probabilities()),point['exact_qcl'],rtol=0,atol=1e-12)
        reset=max(backend.target['reset'][(wire,)].duration for wire,_ in measured)
        durations.append(duration+reset)
        layouts.append([wire for wire,bit in sorted(measured,key=lambda pair:pair[1])])
    assert all(layout==layouts[0] for layout in layouts)
    repetition=float(backend.default_rep_delay)
    # Sum gate times ignores parallelism; allow 2 seconds overhead for each of 3 circuits.
    estimate=6+SHOTS*sum(d+repetition for d in durations)
    if estimate>=CAP:
        raise ValueError('Estimated pilot usage exceeds cap')
    circuit_path=ROOT/'submitted_circuits.qpy'
    ROOT.mkdir(parents=True,exist_ok=True)
    with circuit_path.open('wb') as stream:
        qpy.dump(circuits,stream)
    plan={'backend':BACKEND,'real_hardware':True,'inputs':[-1,0,1],'circuits':3,
          'prepared_at_utc':datetime.now(timezone.utc).isoformat(),
          'versions':{p:importlib.metadata.version(p) for p in ('qiskit','qiskit-ibm-runtime','numpy')},
          'shots_per_circuit':SHOTS,'total_shots':3*SHOTS,'model_update':manifest['model_update'],
          'checkpoint_sha256':manifest['checkpoint_sha256'],'physical_qubits_by_output':layouts,
          'compiled_circuit_sha256':hashlib.sha256(circuit_path.read_bytes()).hexdigest(),
          'compiled_output_equivalence_checked':True,
          'gate_time_sum_plus_reset_seconds':durations,'default_rep_delay_seconds':repetition,
          'conservative_baseline_estimate_seconds':estimate,'estimate_is_not_a_billing_guarantee':True,
          'max_execution_time_seconds':CAP,'usage_before':usage,
          'options':{'max_execution_time':CAP,'dynamical_decoupling':{'enable':False},
                     'twirling':{'enable_gates':False,'enable_measure':False},'execution':{'init_qubits':True}},
          'authorization':'User instructed proceeding on real hardware after the described three-input pilot.'}
    save('plan.json',plan)
    return backend,circuits,plan


def retrieve(service):
    receipt=json.loads((ROOT/'job.json').read_text())
    job=service.job(receipt['job_id'])
    status=str(job.status())
    save('status.json',{'job_id':receipt['job_id'],'status':status})
    print('Job status:',status)
    if status!='DONE': return
    result=job.result()
    manifest=json.loads(Path('results/hardware/pilot_manifest.json').read_text())
    rows=[]
    for pub,point in zip(result,manifest['points']):
        counts=pub.data.meas.get_counts()
        assert sum(counts.values())==SHOTS
        assert all(len(key)==2 and set(key)<=set('01') for key in counts)
        probabilities=np.array([counts.get(format(i,'02b'),0)/SHOTS for i in range(4)])
        prediction=z_outputs(probabilities)
        exact=np.array(point['exact_qcl']); target=np.array(point['numpy_target'])
        rows.append({'t':point['t'],'counts':counts,'hardware_prediction':prediction.tolist(),
                     'exact_qcl':exact.tolist(),'numpy_target':target.tolist(),
                     'hardware_minus_exact':(prediction-exact).tolist(),'hardware_minus_target':(prediction-target).tolist(),
                     'measurement_standard_error':np.sqrt((1-prediction**2)/(SHOTS-1)).tolist()})
    assert len(rows)==3
    metrics=job.metrics()
    save('results.json',{'job_id':receipt['job_id'],'backend':BACKEND,'status':status,'rows':rows,
                         'usage':metrics.get('usage',{}),'usage_after':numeric_usage(service.usage()),
                         'training_updates_on_hardware':0})
    print('Saved measured counts, output errors and usage for the three-input hardware pilot.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true')
    group.add_argument('--submit',action='store_true')
    group.add_argument('--retrieve',action='store_true')
    args=parser.parse_args()
    logging.disable(logging.CRITICAL)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            service=connect()
            if args.retrieve:
                retrieve(service); return
            if (ROOT/'job.json').exists() or (ROOT/'submission_started.json').exists():
                raise ValueError('Pilot already submitted or submission outcome needs investigation; no duplicate allowed')
            backend,circuits,plan=prepare(service)
            print('Backend:',BACKEND,'; estimated seconds:',round(plan['conservative_baseline_estimate_seconds'],3),'; cap:',CAP)
            if args.prepare: return
            # Create an exclusive marker BEFORE the network mutation to prevent duplicate retries.
            with (ROOT/'submission_started.json').open('x') as stream:
                json.dump({'backend':BACKEND,'circuits':3,'shots':SHOTS},stream)
            sampler=SamplerV2(mode=backend,options=plan['options'])
            job=sampler.run(circuits,shots=SHOTS)
            save('job.json',{'job_id':job.job_id(),'backend':BACKEND,'circuits':3,'shots_per_circuit':SHOTS,
                             'submitted_at_utc':datetime.now(timezone.utc).isoformat(),
                             'status_at_submission':str(job.status()),'checkpoint_sha256':plan['checkpoint_sha256']})
            print('Submitted one real-hardware pilot. Job ID:',job.job_id())
    except Exception as exc:
        # Never print provider messages: they may embed account/request details.
        code=getattr(exc,'code',None) or getattr(exc,'status_code',None)
        safe_code=str(code) if isinstance(code,int) else 'unavailable'
        raise SystemExit(f'Pilot stopped: {type(exc).__name__}, HTTP code {safe_code}. Check saved receipt before any retry.') from None


if __name__=='__main__':
    main()
