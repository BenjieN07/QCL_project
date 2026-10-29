from qcl.checkpoints import load_checkpoint
import unittest
import numpy as np
import torch
from qcl.mirror_study import datasets,evaluate_values
from qcl.model import QCLModel


class MirrorTests(unittest.TestCase):
    def test_mirrored_data_are_symmetric_and_disjoint(self):
        data=datasets()
        np.testing.assert_allclose(data['mirrored_t'],-data['mirrored_t'][::-1],atol=0,rtol=0)
        np.testing.assert_allclose(data['mirrored_y'][:,0],-data['mirrored_y'][::-1,0],atol=1e-14)
        np.testing.assert_allclose(data['mirrored_y'][:,1],data['mirrored_y'][::-1,1],atol=1e-14)
        self.assertEqual(len(data['mirrored_t']),96); self.assertEqual(len(data['symmetric_validation_t']),32)

    def test_initial_source_passes_per_output_and_boundary_guards(self):
        checkpoint=load_checkpoint('results/boundary/runs/refine_constant/best.pt',weights_only=True)
        model=QCLModel(5,41,.9); model.load_state_dict(checkpoint['model_state_dict'])
        losses,mses,collar,endpoint=evaluate_values(model,datasets(),'train')
        self.assertAlmostEqual(sum(mses),checkpoint['validation_loss'],places=12)
        self.assertLessEqual(max(collar,endpoint),.05)
