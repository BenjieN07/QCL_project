import unittest
import numpy as np
import torch
try:
    from qiskit.quantum_info import Statevector
    from qcl.hardware import export_circuit,z_outputs
except ModuleNotFoundError as exc:
    if exc.name != 'qiskit':
        raise
    raise unittest.SkipTest('Install requirements-hardware.txt for optional export checks')
from qcl.model import QCLModel


class HardwareTests(unittest.TestCase):
    def test_bit_mapping(self):
        np.testing.assert_array_equal(z_outputs(np.eye(4)),[[1,1],[-1,1],[1,-1],[-1,-1]])

    def test_export_matches_original_at_multiple_depths(self):
        ts=np.array([-1,-.812,-.338,0,.027,.65,1])
        for depth in (1,3,5):
            model=QCLModel(depth,19,.9)
            # Wide angles catch ordering errors that near-zero initialization can hide.
            with torch.no_grad():
                model.weights.mul_(15)
                expected=model(torch.tensor(ts,dtype=torch.float64)).numpy()
            actual=np.array([z_outputs(Statevector.from_instruction(export_circuit(model.weights.detach().numpy(),t)).probabilities()) for t in ts])
            np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
