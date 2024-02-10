# FL_Core/client.py

import json
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
        print(f"\nTraining client {self.client_id} model...")
        self.model = train_model(self.model, self.train_loader, epochs, lr, self.device)
        return self.model.state_dict()

    def evaluate(self):
        print(f"\nEvaluating client {self.client_id} model...")
        metrics = evaluate_model(self.model, self.test_loader, self.device)
        test_loss, accuracy, precision, recall, f1, conf_matrix = metrics
        print(f"Client {self.client_id} Evaluation - Loss: {test_loss}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
        print(f"Client {self.client_id} Confusion Matrix:\n{conf_matrix}")
        return metrics
    

    def update_model(self, global_model_state_dict):
        """
        Update local model parameters with the global model's state_dict.
        """
        self.model.load_state_dict(global_model_state_dict)
        self.model.to(self.device)

