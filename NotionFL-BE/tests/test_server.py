import unittest
from FL_Core.server import FLServer
from models.model import MNISTModel
import torch

class TestFLServer(unittest.TestCase):

    def setUp(self):
        self.global_model = MNISTModel()
        self.server = FLServer(self.global_model)

    def test_aggregate_client_updates(self):
        # Mock client updates - Simplified example
        client_updates = [
            {'weights': torch.tensor([1.0, 2.0])},
            {'weights': torch.tensor([3.0, 4.0])}
        ]
        self.server.aggregate_client_updates(client_updates)
        for param in self.global_model.state_dict().values():
            self.assertTrue(torch.allclose(param, torch.tensor([2.0, 3.0])), "Aggregation failed")

    def test_evaluate_global_model(self):
        # You should use a small mock dataset for testing
        # Assume test_loader is a DataLoader with test data
        # test_loader = create_mock_test_loader()
        # accuracy = self.server.evaluate_global_model(test_loader)
        # self.assertIsInstance(accuracy, float, "Evaluation should return a float")
        pass

    # Add more tests for other server functionalities

if __name__ == '__main__':
    unittest.main()
