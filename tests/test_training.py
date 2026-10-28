import unittest
import torch
from qcl.model import QCLModel
from qcl.training import regression_loss, checked_backward


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.model = QCLModel()
        self.inputs = torch.tensor([-0.8, -0.2, 0.3, 0.9], dtype=torch.float64)
        self.targets = torch.stack((torch.sin(torch.pi * self.inputs),
                                    torch.cos(torch.pi * self.inputs)), dim=1)

    def test_loss_is_sum_over_outputs_mean_over_batch(self):
        self.assertEqual(regression_loss(torch.zeros(2, 2, dtype=torch.float64),
                                         torch.ones(2, 2, dtype=torch.float64)).item(), 2)

    def test_gradient_matches_finite_difference(self):
        loss = regression_loss(self.model(self.inputs), self.targets)
        self.assertGreater(checked_backward(loss, self.model), 1e-6)
        actual = self.model.weights.grad.clone()
        epsilon = 1e-6
        for wire in range(2):
            for angle in range(3):
                index = (0, wire, angle)
                original = self.model.weights[index].item()
                values = []
                for offset in (epsilon, -epsilon):
                    with torch.no_grad():
                        self.model.weights[index] = original + offset
                        values.append(regression_loss(self.model(self.inputs), self.targets).item())
                with torch.no_grad():
                    self.model.weights[index] = original
                self.assertAlmostEqual(actual[index].item(), (values[0] - values[1]) / (2 * epsilon), places=7)

    def test_optimizer_reduces_loss(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.05)
        initial = regression_loss(self.model(self.inputs), self.targets).item()
        for _ in range(5):
            optimizer.zero_grad()
            loss = regression_loss(self.model(self.inputs), self.targets)
            checked_backward(loss, self.model)
            optimizer.step()
        self.assertLess(regression_loss(self.model(self.inputs), self.targets).item(), initial)

    def test_errors_are_actionable(self):
        with self.assertRaisesRegex(TypeError, 'float64'):
            self.model(self.inputs.float())
        with self.assertRaisesRegex(ValueError, 'NaN'):
            regression_loss(torch.full((4, 2), float('nan'), dtype=torch.float64), self.targets)
        with self.assertRaisesRegex(ValueError, 'shapes must match'):
            regression_loss(self.targets[:1], self.targets)
        with self.assertRaisesRegex(RuntimeError, 'detached'):
            checked_backward(regression_loss(self.targets, self.targets), self.model)


if __name__ == '__main__':
    unittest.main()
