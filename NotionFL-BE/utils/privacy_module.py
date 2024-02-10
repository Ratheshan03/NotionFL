import numpy as np
import torch

def apply_differential_privacy(model_parameters, clip_threshold, noise_multiplier, device):
    """
    Apply gradient clipping and add Gaussian noise for differential privacy.

    :param model_parameters: A list of parameter tensors from the model.
    :param clip_threshold: The maximum L2 norm of the gradients.
    :param noise_multiplier: The amount of noise to add (related to the privacy budget).
    :param device: The device on which to perform the calculations.
    """
    for p in model_parameters:
        if p.grad is not None:
            # Compute the L2 norm of the gradient.
            grad_norm = p.grad.norm(2)

            # Clip the gradient to the threshold.
            clip_coef = min(1, clip_threshold / (grad_norm + 1e-6))
            p.grad.data = p.grad.data * clip_coef

            # Add Gaussian noise.
            noise = torch.normal(0, noise_multiplier * clip_threshold, p.grad.data.size()).to(device)
            p.grad.data += noise
