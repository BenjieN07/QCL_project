"""Paper-inspired Ising circuit; two-output extension, not an exact reproduction."""
import numpy as np
import pennylane as qml
from scipy.linalg import expm
import torch


def ising_unitary(qubits, seed, evolution_time):
    rng = np.random.default_rng(seed)
    fields = rng.uniform(-1, 1, qubits)
    pairs = [(i, j) for i in range(qubits) for j in range(i)]
    couplings = rng.uniform(-1, 1, len(pairs))
    identity = np.eye(2)
    x = np.array([[0., 1.], [1., 0.]])
    z = np.diag([1., -1.])

    def product(operators):
        result = np.ones((1, 1))
        for wire in range(qubits):
            result = np.kron(result, operators.get(wire, identity))
        return result

    hamiltonian = sum(a * product({wire: x}) for wire, a in enumerate(fields))
    for (i, j), coupling in zip(pairs, couplings):
        hamiltonian += coupling * product({i: z, j: z})
    unitary = expm(-1j * evolution_time * hamiltonian)
    if not np.allclose(unitary.conj().T @ unitary, np.eye(2 ** qubits), atol=1e-12, rtol=0):
        raise RuntimeError('Ising evolution failed unitarity check')
    return fields, couplings, unitary


class PaperInspiredQCL(torch.nn.Module):
    def __init__(self, qubits=6, layers=6, seed=7, hamiltonian_seed=2026, evolution_time=10.):
        super().__init__()
        if qubits < 2 or layers < 1:
            raise ValueError('Need at least two qubits and one layer')
        self.qubits, self.layers = qubits, layers
        generator = torch.Generator().manual_seed(seed)
        self.weights = torch.nn.Parameter(2 * torch.pi * torch.rand(layers, qubits, 3, generator=generator, dtype=torch.float64))
        self.scales = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
        fields, couplings, unitary = ising_unitary(qubits, hamiltonian_seed, evolution_time)
        self.register_buffer('fields', torch.tensor(fields, dtype=torch.float64))
        self.register_buffer('couplings', torch.tensor(couplings, dtype=torch.float64))
        self.register_buffer('entangler', torch.tensor(unitary, dtype=torch.complex128))

        @qml.qnode(qml.device('default.qubit', wires=qubits, shots=None), interface='torch', diff_method='backprop')
        def circuit(inputs, weights):
            # Encode once, as opposed to the original baseline's per-layer re-uploading.
            for wire in range(qubits):
                qml.RY(torch.asin(inputs), wires=wire)
                qml.RZ(torch.acos(inputs * inputs), wires=wire)
            for layer in weights:
                for wire in range(qubits):
                    qml.RX(layer[wire, 0], wires=wire)
                    qml.RZ(layer[wire, 1], wires=wire)
                    qml.RX(layer[wire, 2], wires=wire)
                qml.QubitUnitary(self.entangler, wires=range(qubits))
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        self.circuit = circuit

    def forward(self, inputs):
        if not isinstance(inputs, torch.Tensor) or inputs.dtype != torch.float64 or inputs.device.type != 'cpu':
            raise TypeError('inputs must be CPU torch.float64 tensors')
        if inputs.ndim != 1 or inputs.numel() == 0 or not torch.isfinite(inputs).all() or (inputs.abs() > 1).any():
            raise ValueError('inputs must be a nonempty finite vector in [-1, 1]')
        return torch.stack(self.circuit(inputs, self.weights), dim=1) * self.scales


def build_model(config):
    if config.get('model', 'original') == 'original':
        from .model import QCLModel
        return QCLModel(config['layers'], config['parameter_seed'])
    if config['model'] != 'paper_inspired':
        raise ValueError('Unknown model family')
    return PaperInspiredQCL(config['qubits'], config['layers'], config['parameter_seed'],
                            config['hamiltonian_seed'], config['evolution_time'])
