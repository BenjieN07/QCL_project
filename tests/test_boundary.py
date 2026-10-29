from qcl.checkpoints import load_checkpoint
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from qcl.boundary_study import make_datasets, restore_start
from qcl.model import QCLModel


class BoundaryTests(unittest.TestCase):
    def test_dataset_roles_are_disjoint(self):
        spec = json.loads(Path('configs/boundary_study.json').read_text())
        source = load_checkpoint(Path(spec['source_run']) / 'best.pt', weights_only=True)
        data = make_datasets(spec, source['config'])
        self.assertEqual(len(data['anchor_t']), 12)
        self.assertEqual(len(data['collar_t']), 10)
        np.testing.assert_array_equal(data['anchor_t'][[0,-1]], [-1.,1.])
        self.assertFalse(np.isin(data['anchor_t'], data['collar_t']).any())

    def test_resume_restores_weights_moments_and_counter(self):
        source = load_checkpoint('results/depth_study/runs/depth3_seed41/best.pt', weights_only=True)
        config = dict(source['config'], input_scale=1.)
        model, optimizer = restore_start(config, source)
        state = optimizer.state[model.weights]
        self.assertEqual(int(state['step'].item()), 250)
        saved_state = next(iter(source['optimizer_state_dict']['state'].values()))
        torch.testing.assert_close(state['exp_avg'], saved_state['exp_avg'], atol=0, rtol=0)
        torch.testing.assert_close(state['exp_avg_sq'], saved_state['exp_avg_sq'], atol=0, rtol=0)

    def test_smooth_encoding_has_finite_endpoint_input_gradients(self):
        model = QCLModel(3, 41, input_scale=.9)
        inputs = torch.tensor([-1., 1.], dtype=torch.float64, requires_grad=True)
        prediction = model(inputs)
        derivative, = torch.autograd.grad(prediction.sum(), inputs)
        self.assertTrue(torch.isfinite(derivative).all())
        self.assertTrue(torch.isfinite(prediction).all())
        with self.assertRaises(ValueError):
            QCLModel(input_scale=1.1)

    def test_fresh_capacity_trials_do_not_load_old_weights_or_moments(self):
        source = load_checkpoint('results/depth_study/runs/depth3_seed41/best.pt', weights_only=True)
        config = dict(source['config'], layers=5, start_mode='fresh', input_scale=.9)
        model, optimizer = restore_start(config, source)
        self.assertEqual(tuple(model.weights.shape), (5, 2, 3))
        self.assertEqual(len(optimizer.state), 0)
        expected = QCLModel(5, 41, input_scale=.9)
        torch.testing.assert_close(model.weights, expected.weights, atol=0, rtol=0)

    def test_refinement_resumes_five_layer_parent_exactly(self):
        source = load_checkpoint('results/boundary/runs/capacity_smooth/best.pt', weights_only=True)
        config = dict(source['config'], start_mode='resume')
        model, optimizer = restore_start(config, source)
        self.assertEqual(tuple(model.weights.shape), (5,2,3))
        self.assertEqual(int(optimizer.state[model.weights]['step'].item()), 1500)
        torch.testing.assert_close(model.weights, source['model_state_dict']['weights'], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
