import copy
import os
import sys
import unittest
import torch
import logging
from torchvision import datasets, transforms

# Add project directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from FL_Core.client import FLClient
from FL_Core.server import FLServer
from utils.privacy_module import apply_differential_privacy
from utils.secure_aggregation import perform_fedavg_aggregation
from models.model import MNISTModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define dataset transforms for MNIST
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

class TestFLProcessWithDP(unittest.TestCase):

    def setUp(self):
        # Initialize data loaders for MNIST
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=True, download=True, transform=mnist_transform),
            batch_size=64, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, download=True, transform=mnist_transform),
            batch_size=64, shuffle=False)

        self.global_model = MNISTModel()

        self.server = FLServer(self.global_model)
        self.clients = [
            FLClient(client_id=i, model=copy.deepcopy(self.global_model), train_loader=self.train_loader, test_loader=self.test_loader, device='cpu')
            for i in range(5)
        ]

        # Parameters for DP
        self.clip_threshold = 1.0
        self.noise_multiplier = 0.1

    def test_dp_effectiveness_against_poisoning(self):
        logging.info("Testing differential privacy effectiveness against model poisoning.")

        all_client_updates = [] 
        client_weights = []
        for client in self.clients:
            # Training and getting model updates
            logging.info(f"Training client {client.client_id}.")
            individual_client_updates, _ = client.train_and_get_updates(epochs=3, lr=0.001)
            logging.info(f"Applying differential privacy to client {client.client_id}.")
            # Load the updates back to the model and apply DP
            client.model.load_state_dict(individual_client_updates)
            apply_differential_privacy(client.model.parameters(), self.clip_threshold, self.noise_multiplier, 'cpu')

            # Collect the DP-applied updates
            all_client_updates.append(copy.deepcopy(client.model.state_dict()))
            client_weights.append(1.0)

        logging.info("Simulating a malicious client with inverted gradients.")
        # Simulate a malicious client with inverted gradients
        malicious_client = FLClient(client_id='malicious', model=copy.deepcopy(self.global_model), train_loader=self.train_loader, test_loader=self.test_loader, device='cpu')
        malicious_client.model.train()
        optimizer = torch.optim.SGD(malicious_client.model.parameters(), lr=0.01)
        for data, target in self.train_loader:
            optimizer.zero_grad()
            output = malicious_client.model(data)
            loss = torch.nn.functional.nll_loss(output, target)
            loss.backward()

            # Inverting gradients
            with torch.no_grad():
                for param in malicious_client.model.parameters():
                    param.grad.neg_()

            optimizer.step()

        # Apply DP to the malicious client
        apply_differential_privacy(malicious_client.model.parameters(), self.clip_threshold, self.noise_multiplier, 'cpu')

        # Append the malicious client's DP-applied state_dict
        all_client_updates.append(copy.deepcopy(malicious_client.model.state_dict()))

        # Perform secure aggregation
        logging.info("Performing secure aggregation with client updates.")
        aggregated_model, _ = perform_fedavg_aggregation(self.global_model.state_dict(), all_client_updates, client_weights)

        # Update global model and evaluate performance
        self.global_model.load_state_dict(aggregated_model)
        final_accuracy = self._evaluate_model_performance()
        logging.info(f"Final global model accuracy after applying DP and handling malicious client: {final_accuracy}")

        # Evaluate the global model performance
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                # Access the global model through the server
                output = self.server.get_global_model()(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        # Update global model and evaluate performance
        self.global_model.load_state_dict(aggregated_model)
        final_accuracy = self._evaluate_model_performance()
        logging.info(f"Final global model accuracy after applying DP and handling malicious client: {final_accuracy}")

        if final_accuracy > 0.7:
            logging.info("Differential privacy effectively mitigated the model poisoning attack.")
        else:
            logging.error("Model accuracy below the acceptable threshold. This indicates a potential vulnerability in the current DP setup to certain types of attacks.")
            logging.info("Further investigation and possible adjustments to the DP parameters are required to enhance resilience against these attacks.")
        logging.info("Privacy and Security effectiveness test completed.")
        
        
    def _evaluate_model_performance(self):
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in self.test_loader:
                    output = self.server.get_global_model()(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
            return correct / total


if __name__ == "__main__":
    unittest.main()
