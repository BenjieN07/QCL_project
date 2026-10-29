import unittest
import numpy as np
from qcl.evaluate import metrics, analytic_reference, polynomial_features


class EvaluationTests(unittest.TestCase):
    def test_metrics_known_errors(self):
        scores = metrics(np.array([[1., -2.], [3., -4.]]), np.zeros((2, 2)))
        self.assertEqual(scores['sin'], {'mse': 5., 'mae': 2., 'max_absolute_error': 3.})
        self.assertEqual(scores['cos'], {'mse': 10., 'mae': 3., 'max_absolute_error': 4.})
        with self.assertRaises(ValueError):
            metrics(np.zeros((2, 2)), np.zeros((1, 2)))
        with self.assertRaises(ValueError):
            metrics(np.full((2, 2), np.nan), np.zeros((2, 2)))

    def test_quantum_analytic_reference_including_boundaries(self):
        inputs = np.array([-1., -0.7, 0., 0.3, 1.])
        expected = np.column_stack((np.sin(np.pi * inputs), np.cos(np.pi * inputs)))
        np.testing.assert_allclose(analytic_reference(inputs), expected, atol=1e-14, rtol=0)

    def test_polynomial_fit_recovers_known_coefficients(self):
        inputs = np.linspace(-1, 1, 12)
        coefficients = np.array([[1., 2.], [3., -1.], [-2., 0.5]])
        features = polynomial_features(inputs, 2)
        recovered = np.linalg.lstsq(features, features @ coefficients, rcond=None)[0]
        np.testing.assert_allclose(recovered, coefficients, atol=1e-13, rtol=0)


if __name__ == '__main__':
    unittest.main()
