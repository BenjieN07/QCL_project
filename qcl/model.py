"""Two-qubit feature map; encoding is fixed, rotations are trainable."""
import pennylane as qml
import torch


class QCLModel(torch.nn.Module):
    def __init__(self, layers=1, seed=7, input_scale=1.0):
        super().__init__()
        if not isinstance(layers, int) or layers < 1:
            raise ValueError('layers must be a positive integer')
        if not 0 < input_scale <= 1:
            raise ValueError('input_scale must be in (0, 1]')
        self.input_scale = float(input_scale)
        generator = torch.Generator().manual_seed(seed)
        self.weights = torch.nn.Parameter(
            0.1 * torch.randn(layers, 2, 3, generator=generator, dtype=torch.float64))
        device = qml.device('default.qubit', wires=2, shots=None)

        @qml.qnode(device, interface='torch', diff_method='backprop')
        def circuit(t, weights):
            encoded = self.input_scale * t
            for layer in weights:
                for wire in range(2):
                    qml.RY(torch.asin(encoded), wires=wire)
                    qml.RZ(torch.acos(encoded * encoded), wires=wire)
                    qml.Rot(*layer[wire], wires=wire)
                qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        self.circuit = circuit

    def forward(self, inputs):
        if not isinstance(inputs, torch.Tensor):
            raise TypeError('inputs must be a PyTorch tensor')
        if inputs.dtype != torch.float64 or inputs.device.type != 'cpu':
            raise TypeError('inputs must be CPU torch.float64 tensors')
        if inputs.ndim != 1 or inputs.numel() == 0:
            raise ValueError('inputs must be a nonempty vector')
        if not torch.isfinite(inputs).all() or (inputs.abs() > 1).any():
            raise ValueError('inputs must be finite and in [-1, 1]')
        # Broadcast inputs through the same gates; no change to the circuit or loss.
        return torch.stack(self.circuit(inputs, self.weights), dim=1)
