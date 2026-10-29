"""Read-only IBM access check. Never saves credentials or submits a quantum job."""
import argparse
import getpass
import json
import logging
from pathlib import Path
import sys
import warnings


def numeric_usage(value, prefix=''):
    """Keep numeric quota/usage evidence without account identifiers or strings."""
    result={}
    if isinstance(value,dict):
        for key,item in value.items():
            # Only documented-looking usage fields; never copy arbitrary identifiers.
            if any(word in str(key).lower() for word in ('usage','limit','remaining','allocated','seconds','minutes','quota','consumed')):
                if isinstance(item,(int,float)) and not isinstance(item,bool):
                    result[prefix+str(key)]=item
            if isinstance(item,(dict,list)):
                result.update(numeric_usage(item,prefix+str(key)+'.' if isinstance(key,str) and not key.startswith('crn:') else prefix))
    elif isinstance(value,list):
        for item in value:
            result.update(numeric_usage(item,prefix))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',help='An explicitly chosen backend; omission lists accessible backends only')
    args=parser.parse_args()
    if not sys.stdin.isatty():
        raise SystemExit('Run this script in your own interactive terminal for hidden credential entry.')
    from qiskit_ibm_runtime import QiskitRuntimeService
    # Explicit hidden entry avoids copying credentials into command history or files.
    token=getpass.getpass('New IBM API secret (hidden; not an API key identifier): ')
    instance=getpass.getpass('Instance CRN (hidden): ')
    if token.startswith('ApiKey-') or not token.strip() or not instance.startswith('crn:'):
        raise SystemExit('Enter the API secret value and instance CRN from your IBM account. Nothing saved.')
    logging.getLogger('qiskit_ibm_runtime').setLevel(logging.CRITICAL)
    logging.getLogger('ibm_cloud_sdk_core').setLevel(logging.CRITICAL)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            service=QiskitRuntimeService(channel='ibm_quantum_platform',token=token,instance=instance)
            usage=numeric_usage(service.usage())
            backends=service.backends(name=args.backend) if args.backend else service.backends()
            available=[{'name':b.name,'qubits':b.num_qubits,'operational':b.status().operational} for b in backends]
            report={'qpu_jobs_submitted':0,'usage_numeric_fields':usage,'backends':available,
                    'quota_approved':False,'submission_authorized':False,
                    'note':'Read-only access check. Confirm your plan, remaining allowance and the specific pilot before submission.'}
            if args.backend:
                if len(backends)!=1:
                    raise ValueError('Backend not uniquely accessible')
                from qiskit import qpy,transpile
                with Path('results/hardware/circuits.qpy').open('rb') as stream:
                    circuits=qpy.load(stream)
                compiled=transpile(circuits,backend=backends[0],optimization_level=3,seed_transpiler=2026)
                report['target_compilation']=[{'depth':c.depth(),'operations':dict(c.count_ops()),
                                              'physical_layout':c.layout.final_index_layout()} for c in compiled]
                private=Path('.ibm-private'); private.mkdir(exist_ok=True)
                with (private/'target_circuits.qpy').open('wb') as stream:
                    qpy.dump(compiled,stream)
            private=Path('.ibm-private'); private.mkdir(exist_ok=True)
            (private/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(report,indent=2))
            print('Saved local read-only report in .ibm-private/preflight.json. No credentials saved and no jobs submitted.')
    except Exception:
        # Provider exceptions may contain request/account details. Do not echo them.
        raise SystemExit('Read-only preflight failed. Check the new secret, CRN, access and backend locally. No jobs submitted.') from None
    finally:
        token=None
        instance=None


if __name__=='__main__':
    main()
