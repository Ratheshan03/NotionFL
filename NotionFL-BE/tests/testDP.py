import os
import sys
import unittest
import torch

# Add the project directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.privacy_module import apply_differential_privacy

class TestDifferentialPrivacy(unittest.TestCase):
    def test_dp_noise_application(self):
        model_param = torch.nn.Parameter(torch.ones(5, 5))
        model_param.grad = torch.ones_like(model_param)

        noise_multiplier = 0.5
        clip_threshold = 1.0
        device = 'cpu'

        dp_result = apply_differential_privacy([model_param], clip_threshold, noise_multiplier, device)

        self.assertIn('noise_stats', dp_result)
        self.assertIn('computation_time', dp_result)
        self.assertTrue(model_param.grad.abs().max() <= clip_threshold + noise_multiplier * clip_threshold)

if __name__ == '__main__':
    unittest.main()
