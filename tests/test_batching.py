import unittest
import torch
from qcl.model import QCLModel
from qcl.training import regression_loss


class BatchingTests(unittest.TestCase):
    def test_predictions_and_gradients_match_scalar_circuits_at_all_study_depths(self):
        inputs = torch.tensor([-1., -.7, -.1, .35, .8, 1.], dtype=torch.float64)
        targets = torch.stack((torch.sin(torch.pi * inputs), torch.cos(torch.pi * inputs)), dim=1)
        for depth in (1, 2, 3, 5):
            with self.subTest(depth=depth):
                model = QCLModel(layers=depth, seed=7)
                batch = model(inputs)
                scalar = torch.stack([torch.stack(model.circuit(t, model.weights)) for t in inputs])
                torch.testing.assert_close(batch, scalar, atol=1e-12, rtol=0)
                batch_gradient, = torch.autograd.grad(regression_loss(batch, targets), model.weights)
                scalar_gradient, = torch.autograd.grad(regression_loss(scalar, targets), model.weights)
                torch.testing.assert_close(batch_gradient, scalar_gradient, atol=1e-12, rtol=0)
                self.assertEqual(tuple(model(inputs[:1]).shape), (1, 2))


if __name__ == '__main__':
    unittest.main()
