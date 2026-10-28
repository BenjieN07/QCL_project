"""Validated two-output loss and gradient checks for local training."""
import torch


def regression_loss(predictions, targets):
    for name, value in (('predictions', predictions), ('targets', targets)):
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float64:
            raise TypeError(f'{name} must be a torch.float64 tensor')
        if value.ndim != 2 or value.shape[1] != 2 or value.shape[0] == 0:
            raise ValueError(f'{name} must have nonempty shape (batch, 2)')
        if not torch.isfinite(value).all():
            raise ValueError(f'{name} contains NaN or infinity')
    if predictions.shape != targets.shape:
        raise ValueError('prediction and target shapes must match')
    return ((predictions - targets) ** 2).sum(dim=1).mean()


def checked_backward(loss, model):
    if not loss.requires_grad:
        raise RuntimeError('Loss is detached: gradient flow is broken')
    loss.backward()
    gradients = []
    for parameter in model.parameters():
        if parameter.grad is None:
            raise RuntimeError('A trainable parameter has no gradient')
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError('A parameter gradient contains NaN or infinity')
        gradients.append(parameter.grad.flatten())
    norm = torch.cat(gradients).norm().item()
    if norm <= 1e-12:
        raise RuntimeError('Gradient norm is effectively zero; inspect circuit and initialization')
    return norm
