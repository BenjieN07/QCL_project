import unittest
import numpy as np
import torch
from qcl.paper_model import PaperInspiredQCL
from qcl.training import regression_loss


class PaperModelTests(unittest.TestCase):
    def test_unitarity_batch_and_gradients(self):
        model = PaperInspiredQCL(qubits=3, layers=2)
        np.testing.assert_allclose((model.entangler.conj().T @ model.entangler).numpy(), np.eye(8), atol=1e-12)
        inputs = torch.tensor([-1., -0.3, 0.4, 1.], dtype=torch.float64)
        batched = model(inputs)
        scalar = torch.cat([model(t.reshape(1)) for t in inputs], dim=0)
        torch.testing.assert_close(batched, scalar, atol=1e-13, rtol=0)
        targets = torch.stack((torch.sin(torch.pi * inputs), torch.cos(torch.pi * inputs)), dim=1)
        regression_loss(batched, targets).backward()
        for parameter in model.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
        for parameter, index in ((model.weights, (0, 1, 1)), (model.scales, (0,))):
            analytic = parameter.grad[index].item()
            original = parameter[index].item()
            values = []
            for delta in (1e-6, -1e-6):
                with torch.no_grad():
                    parameter[index] = original + delta
                    values.append(regression_loss(model(inputs), targets).item())
            with torch.no_grad():
                parameter[index] = original
            self.assertAlmostEqual(analytic, (values[0]-values[1])/2e-6, places=7)

    def test_six_qubit_shape_and_parameter_count(self):
        model = PaperInspiredQCL()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 110)
        result = model(torch.tensor([-.5, 0., .5], dtype=torch.float64))
        self.assertEqual(tuple(result.shape), (3, 2))
        self.assertTrue(torch.isfinite(result).all())


if __name__ == '__main__':
    unittest.main()
