import numpy as np
import torch
import time

def apply_differential_privacy(model_parameters, clip_threshold, noise_multiplier, device):
    """
    Apply gradient clipping and add Gaussian noise for differential privacy.

    :param model_parameters: A list of parameter tensors from the model.
    :param clip_threshold: The maximum L2 norm of the gradients.
    :param noise_multiplier: The amount of noise to add (related to the privacy budget).
    :param device: The device on which to perform the calculations.
    :return: Dictionary containing noise statistics and computation time.
    """
    start_time = time.time()
    noise_stats = []

    for p in model_parameters:
        if p.grad is not None:
            grad_norm = p.grad.norm(2)
            clip_coef = min(1, clip_threshold / (grad_norm + 1e-6))
            p.grad.data = p.grad.data * clip_coef

            noise = torch.normal(0, noise_multiplier * clip_threshold, p.grad.data.size()).to(device)
            p.grad.data += noise
            noise_stats.append({'mean': 0, 'std': noise_multiplier * clip_threshold, 'variance': (noise_multiplier * clip_threshold)**2})

    computation_time = time.time() - start_time
    return {'noise_stats': noise_stats, 'computation_time': computation_time}
