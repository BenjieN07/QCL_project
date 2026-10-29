"""Read-only investigation of signed-error peaks in the active saved model."""
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
from .data import targets
from .model import QCLModel


def model_from(checkpoint):
    config = checkpoint['config']
    model = QCLModel(config['layers'], config['parameter_seed'], config.get('input_scale',1.))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def slopes(model, points):
    inputs = torch.tensor(points, dtype=torch.float64, requires_grad=True)
    prediction = model(inputs)
    derivatives = []
    for output in range(2):
        derivative, = torch.autograd.grad(prediction[:,output].sum(), inputs, retain_graph=output==0)
        derivatives.append(derivative.detach().numpy())
    actual = np.column_stack(derivatives)
    expected = np.column_stack((np.pi*np.cos(np.pi*points), -np.pi*np.sin(np.pi*points)))
    return actual, expected


def cosine(a,b):
    denominator = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/denominator) if denominator > 1e-20 else None


def investigate(output):
    torch.set_num_threads(1)
    registry_path = Path('results/current_model.json')
    registry_bytes = registry_path.read_bytes()
    registry = json.loads(registry_bytes)
    run = Path(registry['run'])
    path = run/'best.pt'
    checkpoint_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    checkpoint = torch.load(path, weights_only=True)
    model = model_from(checkpoint)
    initial_weights = model.weights.detach().clone()
    parent_path = Path(checkpoint['config']['source_run'])/'best.pt'
    parent_checkpoint = torch.load(parent_path, weights_only=True)
    parent = model_from(parent_checkpoint)
    with np.load('results/boundary/training_data.npz') as saved:
        data = {key:saved[key] for key in saved.files}
    grid = np.linspace(-1,1,10001)
    with torch.no_grad():
        prediction = model(torch.tensor(grid,dtype=torch.float64)).numpy()
        parent_prediction = parent(torch.tensor(grid,dtype=torch.float64)).numpy()
    expected = targets(grid)
    error = prediction-expected
    parent_error = parent_prediction-expected
    all_training = np.sort(np.concatenate((data['train_t'],data['anchor_t'])))
    gradients = {}

    def gradient(inputs, teacher, output_index=None):
        result = model(torch.tensor(inputs,dtype=torch.float64))
        truth = torch.tensor(teacher,dtype=torch.float64)
        residual = result-truth
        loss = residual.square().sum(dim=1).mean() if output_index is None else residual[:,output_index].square().mean()
        value, = torch.autograd.grad(loss,model.weights)
        return value.detach().numpy().ravel()

    g_original = gradient(data['train_t'],data['train_y'])
    g_anchor = gradient(data['anchor_t'],data['anchor_y'])
    g_total = .5*(g_original+g_anchor)
    g_sin = .5*(gradient(data['train_t'],data['train_y'],0)+gradient(data['anchor_t'],data['anchor_y'],0))
    g_cos = .5*(gradient(data['train_t'],data['train_y'],1)+gradient(data['anchor_t'],data['anchor_y'],1))
    np.testing.assert_allclose(g_total,g_sin+g_cos,atol=1e-12,rtol=0)
    gradients.update(original_loss_gradient_norm=float(np.linalg.norm(g_original)),
                     anchor_loss_gradient_norm=float(np.linalg.norm(g_anchor)),
                     weighted_training_gradient_norm=float(np.linalg.norm(g_total)),
                     original_vs_anchor_gradient_cosine=cosine(g_original,g_anchor),
                     sine_vs_cosine_gradient_cosine=cosine(g_sin,g_cos),
                     sine_loss_gradient_norm=float(np.linalg.norm(g_sin)),cosine_loss_gradient_norm=float(np.linalg.norm(g_cos)))
    peaks = []
    for output_index,name in enumerate(('sin','cos')):
        values = error[:,output_index]
        maxima = np.flatnonzero((values[1:-1]>values[:-2]) & (values[1:-1]>=values[2:]))+1
        minima = np.flatnonzero((values[1:-1]<values[:-2]) & (values[1:-1]<=values[2:]))+1
        indices = np.sort(np.concatenate((maxima,minima)))
        locations = grid[indices].copy()
        lower,upper = grid[indices-1],grid[indices+1]
        # Refine stationary points of signed residual, without modifying any model parameter.
        for _ in range(4):
            actual_slope,true_slope = slopes(model,locations)
            derivative = (actual_slope-true_slope)[:,output_index]
            plus_actual,plus_true = slopes(model,locations+1e-5)
            minus_actual,minus_true = slopes(model,locations-1e-5)
            second = ((plus_actual-plus_true)-(minus_actual-minus_true))[:,output_index]/2e-5
            locations = np.clip(locations-derivative/second,lower,upper)
        actual_slope,true_slope = slopes(model,locations)
        assert np.max(np.abs((actual_slope-true_slope)[:,output_index])) < 1e-6
        with torch.no_grad():
            values_at_peaks = model(torch.tensor(locations,dtype=torch.float64)).numpy()
            parent_at_peaks = parent(torch.tensor(locations,dtype=torch.float64)).numpy()
        teacher = targets(locations)
        for i,(location,index) in enumerate(zip(locations,indices)):
            closest = int(np.argmin(np.abs(data['train_t']-location)))
            near_t = data['train_t'][closest]
            with torch.no_grad():
                nearby_prediction = model(torch.tensor([near_t],dtype=torch.float64)).numpy()[0,output_index]
            point_gradient = gradient(np.array([location]),targets(np.array([location])),output_index)
            before_after_t = np.array([location-.025,location+.025])
            before_actual,before_true = slopes(model,before_after_t)
            nearby_original = np.abs(data['train_t']-location)<=.05
            nearby_anchor = np.abs(data['anchor_t']-location)<=.05
            left = all_training[all_training<location]
            right = all_training[all_training>location]
            nearest_augmented = all_training[np.argmin(np.abs(all_training-location))]
            nearest_validation = data['validation_t'][np.argmin(np.abs(data['validation_t']-location))]
            peaks.append({'output':name,'kind':'local maximum' if index in maxima else 'local minimum',
                          't':float(location),'prediction':float(values_at_peaks[i,output_index]),
                          'exact':float(teacher[i,output_index]),'signed_error':float(values_at_peaks[i,output_index]-teacher[i,output_index]),
                          'parent_error_at_same_t':float(parent_at_peaks[i,output_index]-teacher[i,output_index]),
                          'model_slope':float(actual_slope[i,output_index]),'target_slope':float(true_slope[i,output_index]),
                          'error_slope_left_and_right':(before_actual-before_true)[:,output_index].tolist(),
                          'nearest_original_training_t':float(near_t),'nearest_original_training_distance':float(abs(near_t-location)),
                          'error_at_nearest_training_t':float(nearby_prediction-data['train_y'][closest,output_index]),
                          'original_training_points_within_0_05':int(nearby_original.sum()),
                          'anchor_points_within_0_05':int(nearby_anchor.sum()),
                          'augmented_training_bracket':[float(left[-1]) if len(left) else None,float(right[0]) if len(right) else None],
                          'nearest_augmented_training_distance':float(abs(nearest_augmented-location)),
                          'nearest_original_validation_t':float(nearest_validation),
                          'point_gradient_norm':float(np.linalg.norm(point_gradient)),
                          'point_vs_original_gradient_cosine':cosine(point_gradient,g_original),
                          'point_vs_anchor_gradient_cosine':cosine(point_gradient,g_anchor),
                          'point_vs_other_output_gradient_cosine':cosine(point_gradient,g_cos if output_index==0 else g_sin)})
    principal=[]
    for name in ('sin','cos'):
        group=sorted([row for row in peaks if row['output']==name],key=lambda row:abs(row['signed_error']),reverse=True)[:3]
        for number,row in enumerate(group,1):
            row['label']=name[0].upper()+str(number)
            principal.append(row)
    with (run/'history.csv').open() as stream:
        history=list(csv.DictReader(stream))
    at_zero = model(torch.tensor([0.],dtype=torch.float64)).detach().numpy()[0]
    extrema = {}
    symmetry = {}
    for j,name in enumerate(('sin','cos')):
        high,low = int(np.argmax(prediction[:,j])),int(np.argmin(prediction[:,j]))
        extrema[name]={'predicted_maximum':float(prediction[high,j]),'maximum_t':float(grid[high]),
                       'predicted_minimum':float(prediction[low,j]),'minimum_t':float(grid[low])}
        defect = prediction[:,j]+prediction[::-1,j] if j==0 else prediction[:,j]-prediction[::-1,j]
        index = int(np.argmax(np.abs(defect)))
        symmetry[name]={'maximum_parity_defect':float(abs(defect[index])),'at_t':float(grid[index])}
    largest_cosine = max([row for row in principal if row['output']=='cos'],key=lambda row:abs(row['signed_error']))
    mirror_t = -largest_cosine['t']
    with torch.no_grad():
        mirror_prediction = model(torch.tensor([mirror_t],dtype=torch.float64)).numpy()[0,1]
    mirror_comparison={'peak_t':largest_cosine['t'],'peak_error':largest_cosine['signed_error'],'mirror_t':mirror_t,
                       'mirror_error':float(mirror_prediction-targets(np.array([mirror_t]))[0,1]),
                       'original_training_points_within_0_05_at_peak':largest_cosine['original_training_points_within_0_05'],
                       'original_training_points_within_0_05_at_mirror':int((np.abs(data['train_t']-mirror_t)<=.05).sum())}
    # Independent finite differences of the residual verify the stationary locations.
    max_fd_derivative=0.
    for row in principal:
        j=0 if row['output']=='sin' else 1
        pair=np.array([row['t']-1e-5,row['t']+1e-5])
        with torch.no_grad():
            residual=model(torch.tensor(pair,dtype=torch.float64)).numpy()-targets(pair)
        derivative=(residual[1,j]-residual[0,j])/2e-5
        max_fd_derivative=max(max_fd_derivative,abs(float(derivative)))
    assert max_fd_derivative<1e-6
    report={'run':registry['run'],'selected_step':checkpoint['step'],'parent_run':str(parent_path.parent).replace('\\','/'),
            'checkpoint_sha256':checkpoint_hash,'graph_is_signed_error_not_error_rate':True,'grid_points':len(grid),
            'principal_peaks':principal,'all_stationary_error_points':peaks,'gradient_diagnostics':gradients,
            'model_curve_extrema_on_grid':extrema,'largest_finite_difference_error_derivative_at_principal_peaks':max_fd_derivative,
            'symmetry_diagnostics':symmetry,'largest_cosine_peak_mirror_comparison':mirror_comparison,
            'all_six_highlighted_errors_shrank_from_parent':all(abs(row['signed_error'])<abs(row['parent_error_at_same_t']) for row in principal),
            'at_t_zero':{'prediction':at_zero.tolist(),'exact':[0.,1.],
                         'cosine_readout_probability_of_one':float((1-at_zero[1])/2)},
            'validation_start_and_end':{key:[float(history[0][key]),float(history[-1][key])] for key in ('validation_original_loss','validation_collar_loss','selection_score')},
            'model_parameters_unchanged':True,'new_training_performed':False,'new_checkpoint_selection_performed':False}
    assert torch.equal(model.weights.detach(),initial_weights)
    assert model.weights.grad is None
    assert hashlib.sha256(path.read_bytes()).hexdigest()==checkpoint_hash and registry_path.read_bytes()==registry_bytes
    output.mkdir(parents=True,exist_ok=True)
    (output/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez(output/'curves.npz',inputs=grid,target=expected,prediction=prediction,signed_error=error,parent_error=parent_error)
    fig,axes=plt.subplots(2,2,figsize=(13,8),gridspec_kw={'height_ratios':[3,2]})
    for j,name in enumerate(('sin','cos')):
        ax=axes[0,j]
        ax.plot(grid,parent_error[:,j],color='#888888',linestyle='--',label='Update 1,500')
        ax.plot(grid,error[:,j],color='#b54a16',label='Update 3,000')
        ax.axhline(0,color='black',linewidth=.8)
        for row in principal:
            if row['output']==name:
                ax.scatter(row['t'],row['signed_error'],color='#b54a16',zorder=3)
                ax.annotate(row['label']+f' ({row["signed_error"]:+.4f})',(row['t'],row['signed_error']),
                            xytext=(5,8 if row['signed_error']>=0 else -17),textcoords='offset points',fontsize=9)
        ax.plot(data['train_t'],np.full_like(data['train_t'],-.12),linestyle='None',marker='|',color='#1668bb',label='Original training inputs')
        ax.plot(data['anchor_t'],np.full_like(data['anchor_t'],-.13),linestyle='None',marker='|',color='#268251',label='Boundary anchors')
        ax.set(title=name+': error peaks and training coverage',xlabel='t',ylabel='Prediction − exact',ylim=(-.14,.12),xlim=(-1,1))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8,loc='upper right')
        axes[1,j].plot(grid,expected[:,j],color='black',linestyle='--',label='Exact target')
        axes[1,j].plot(grid,prediction[:,j],color='#1668bb',label='Current model')
        axes[1,j].set(xlabel='t',ylabel='Function value',title=name+' prediction versus target')
        axes[1,j].grid(alpha=.2)
        axes[1,j].legend(fontsize=8)
    fig.suptitle('Read-only error-peak investigation: fixed saved model, no new training')
    fig.tight_layout()
    fig.savefig(output/'error_peaks.png',dpi=150)
    plt.close(fig)
    lines=['# Read-only error-peak investigation','',f'Checkpoint: `{registry["run"]}/best.pt`, update {checkpoint["step"]}. SHA-256: `{checkpoint_hash}`.', '',
           'The curve is signed residual = model prediction minus exact target, evaluated on 10,001 plotting inputs. These plotting inputs are not a new independent test set. No optimizer updates or checkpoint selection occurred. The checkpoint bytes, active registry, and in-memory weights were verified unchanged.', '',
           '| Label | Output | t | Prediction | Exact target | Signed error | Parent error at the same t | Original training points within ±0.05 |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in principal:
        lines.append(f'| {row["label"]} | {row["output"]} | {row["t"]:.6f} | {row["prediction"]:.6f} | {row["exact"]:.6f} | {row["signed_error"]:+.6f} | {row["parent_error_at_same_t"]:+.6f} | {row["original_training_points_within_0_05"]} |')
    largest_cos=max([r for r in principal if r['output']=='cos'],key=lambda row:abs(row['signed_error']))
    sine=max([r for r in principal if r['output']=='sin'],key=lambda row:abs(row['signed_error']))
    cos_dip=min([r for r in principal if r['output']=='cos'],key=lambda row:row['signed_error'])
    lines += ['', '## What causes a peak mathematically', '',
              'The residual increases wherever the prediction slope exceeds the target slope and decreases where it is smaller. At each interior peak/trough the slopes match, while the values remain separated. Both autodifferentiated slopes and independent central finite differences verified these stationary locations. The location of an error peak is not necessarily a maximum of the target function.', '',
              '## Coverage and fitting evidence', '',
              f'The largest cosine residual ({largest_cos["label"]}) lies at t={largest_cos["t"]:.6f}. Its bracketing augmented training inputs are {largest_cos["augmented_training_bracket"]}; no original or anchor training inputs lie within ±0.05. Validation has a nearby point at {largest_cos["nearest_original_validation_t"]:.6f}, which measures the error but does not supply optimizer updates. This is direct evidence of a training-coverage gap, not proof that data coverage is the sole cause.', '',
              f'In contrast, the largest sine error ({sine["label"]}) has {sine["original_training_points_within_0_05"]} original training inputs within ±0.05. Its closest training input is {sine["nearest_original_training_t"]:.6f}, with error {sine["error_at_nearest_training_t"]:+.6f}. '
              f'The cosine dip ({cos_dip["label"]}) is only {cos_dip["nearest_original_training_distance"]:.6f} from an original training input, whose error is {cos_dip["error_at_nearest_training_t"]:+.6f}. These are also training-fit errors, so simply adding examples cannot be assumed to remove them.', '',
              f'The predicted cosine maximum is {extrema["cos"]["predicted_maximum"]:.6f} near t={extrema["cos"]["maximum_t"]:.4f}, while the true maximum is 1 at zero. The central negative cosine residual therefore reflects an under-height peak plus a small shape/position mismatch. Sine extrema likewise have small amplitude/position mismatches.', '',
              '## Local optimization trade-offs', '',
              f'Cosine similarity between original-data and anchor-loss gradients: {gradients["original_vs_anchor_gradient_cosine"]:.6f}. Between sine and cosine component gradients: {gradients["sine_vs_cosine_gradient_cosine"]:.6f}. Values near -1 indicate almost opposite parameter-update preferences at these exact weights. Their summed gradient norm is {gradients["weighted_training_gradient_norm"]:.6g}, compared with original/anchor component norms {gradients["original_loss_gradient_norm"]:.6g} and {gradients["anchor_loss_gradient_norm"]:.6g}.', '',
              'This is measured local competition, not a proof of an unavoidable Pareto frontier, a global optimum, a barren plateau, or insufficient circuit capacity. The location and sign of per-peak gradient conflicts differ; reducing one point error can increase a different output or region loss. Full per-peak diagnostics are in analysis.json. Hardware/shot noise is not an explanation for this deterministic exact-simulator residual.', '',
              '## Is error increasing with more training?', '',
              'All six highlighted residual magnitudes are smaller at update 3,000 than at update 1,500 when compared at the same t. This comparison uses identical architecture, encoding and data, so these peaks are shrinking with the observed continuation, although not vanishing. It does not establish monotonic improvement at every input or guarantee future improvement.', '',
              '## Candidate next tests — not executed', '',
              'First, test added stratified interior training coverage in the negative-input gap while preserving all validation/test separation and boundary anchors. Separately, compare loss weighting under per-output and boundary guards: interior errors already present at training examples indicate that coverage alone is not a complete explanation. Keep a same-model continuation as a reference if running new experiments. If conflicts persist, compare initialization seeds before attributing them to an architectural limit. Any future choices should use declared validation criteria and a newly reserved final holdout, not the now-exposed diagnostic curves.', '',
              'No new training, parameter perturbations, model replacement, or registry changes were performed. Recommendations await the user’s next-step decision.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    symmetry_text = (
        '\n## Asymmetric data and missing symmetry constraints\n\n'
        'Sine is odd and cosine is even, but this circuit/training objective does not explicitly enforce those identities. '
        f'The largest cosine peak has error {mirror_comparison["peak_error"]:+.6f} at t={mirror_comparison["peak_t"]:.6f}; '
        f'at its reflected input t={mirror_t:.6f}, error is only {mirror_comparison["mirror_error"]:+.6f}. '
        f'Original training counts within ±0.05 are {mirror_comparison["original_training_points_within_0_05_at_peak"]} '
        f'on the peak side and {mirror_comparison["original_training_points_within_0_05_at_mirror"]} on the mirrored side. '
        'This measured asymmetry is consistent with the coverage problem, without proving a unique cause.\n\n'
        f'Maximum sampled odd-symmetry defect for sine is {symmetry["sin"]["maximum_parity_defect"]:.6f}; '
        f'even-symmetry defect for cosine is {symmetry["cos"]["maximum_parity_defect"]:.6f}. '
        'A first controlled next experiment could mirror the training inputs with their correct sine/cosine labels, '
        'while retaining boundary anchors and fixed validation guards. That is data augmentation within our existing '
        'method, not an analytic output patch. Test loss rebalancing separately because the dense-region peaks '
        'show local fitting conflicts too. Neither experiment has been run.\n')
    with (output/'REPORT.md').open('a',encoding='utf-8') as stream:
        stream.write(symmetry_text)
    print(json.dumps({'principal_peaks':principal,'gradient_diagnostics':gradients,'at_t_zero':report['at_t_zero'],
                      'validation_start_and_end':report['validation_start_and_end'],'no_training_or_model_changes':True},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/error_peaks'))
    investigate(parser.parse_args().output)
