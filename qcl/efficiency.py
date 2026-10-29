"""Controlled CPU timing: scalar versus batched circuit forward and gradient passes."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import time
import numpy as np
import torch
from .checkpoints import load_checkpoint
from .boundary_report import load_model
from .mirror_study import datasets
from .training import regression_loss


def main():
    torch.set_num_threads(1)
    registry=json.loads(Path('results/current_model.json').read_text())
    path=Path(registry['run'])/'best.pt'
    model=load_model(load_checkpoint(path,weights_only=True))
    original=model.weights.detach().clone()
    data=datasets()
    x=torch.tensor(np.concatenate((data['mirrored_t'],data['anchor_t'])),dtype=torch.float64)
    y=torch.tensor(np.concatenate((data['mirrored_y'],data['anchor_y'])),dtype=torch.float64)
    def compute(batched):
        model.zero_grad(set_to_none=True)
        prediction=model(x) if batched else torch.stack([torch.stack(model.circuit(t,model.weights)) for t in x])
        loss=.75*regression_loss(prediction[:96],y[:96])+.25*regression_loss(prediction[96:],y[96:])
        loss.backward()
        return prediction.detach().clone(),float(loss.detach()),model.weights.grad.detach().clone()
    scalar=compute(False); batched=compute(True)
    torch.testing.assert_close(scalar[0],batched[0],atol=1e-12,rtol=0)
    torch.testing.assert_close(scalar[2],batched[2],atol=1e-12,rtol=0)
    assert abs(scalar[1]-batched[1])<1e-12
    timings={'scalar':[],'batched':[]}
    for repeat in range(5):
        for name in (('scalar','batched') if repeat%2==0 else ('batched','scalar')):
            started=time.perf_counter(); compute(name=='batched')
            timings[name].append(time.perf_counter()-started)
    torch.testing.assert_close(model.weights,original,atol=0,rtol=0)
    medians={key:statistics.median(values) for key,values in timings.items()}
    report={'scope':'CPU forward + backward on identical circuit/data; no optimizer steps or QPU execution',
            'inputs':len(x),'layers':5,'qubits':2,'threads':1,'repeats':5,'dtype':'float64',
            'seconds':timings,'median_seconds':medians,'observed_scalar_to_batch_ratio':medians['scalar']/medians['batched'],
            'maximum_prediction_difference':float((scalar[0]-batched[0]).abs().max()),
            'maximum_gradient_difference':float((scalar[2]-batched[2]).abs().max()),
            'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'versions':{name:importlib.metadata.version(name) for name in ('pennylane','torch','numpy')},
            'model_weights_unchanged':True,'training_updates':0,'qpu_jobs':0}
    root=Path('results/summary'); root.mkdir(exist_ok=True)
    (root/'efficiency.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
