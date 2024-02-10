# FL_Core/client.py

import torch
from .training import train_model, evaluate_model

class FLClient:
    def __init__(self, client_id, model, train_loader, test_loader, device):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    def train(self, epochs, lr):
        print(f"Training client {self.client_id} model...")
        self.model = train_model(self.model, self.train_loader, epochs, lr, self.device)
        return self.model.state_dict()

    def evaluate(self):
        print(f"Evaluating client {self.client_id} model...")
        test_loss, accuracy = evaluate_model(self.model, self.test_loader, self.device)
        return test_loss, accuracy

    def update_model(self, global_model_state_dict):
        """
        Update local model parameters with the global model's state_dict.
        """
        self.model.load_state_dict(global_model_state_dict)
        self.model.to(self.device)

