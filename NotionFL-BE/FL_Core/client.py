# FL_Core/client.py

import copy
import json
import torch
from .training import train_model, evaluate_model


class FLClient:
    def __init__(self, client_id, model, train_loader, test_loader, device, data_collector=None):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.data_collector = data_collector

    def train(self, epochs, lr):
        print(f"\nTraining client {self.client_id} model...")
        self.model, training_logs = train_model(self.model, self.train_loader, epochs, lr, self.device)
        if self.data_collector:
            self.data_collector.collect_client_training_logs(self.client_id, training_logs)
        return self.model.state_dict()
    
    def train_and_get_updates(self, epochs, lr):
        initial_state = copy.deepcopy(self.model.state_dict())
        self.train(epochs, lr)
        final_state = self.model.state_dict()

        # Calculate updates (deltas)
        updates = {key: final_state[key] - initial_state[key] for key in final_state}
        return updates

    def evaluate(self, round):
        print(f"\nEvaluating client {self.client_id} model...")
        metrics = evaluate_model(self.model, self.test_loader, self.device)
        test_loss, accuracy, precision, recall, f1, conf_matrix = metrics

        # Collect evaluation logs
        evaluation_logs = {
            'loss': test_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix.tolist()  # Convert numpy array to list for JSON serialization
        }

        if self.data_collector:
            self.data_collector.collect_client_evaluation_logs(self.client_id, evaluation_logs, round)

        return metrics
    

    def update_model(self, global_model_state_dict):
        """
        Update local model parameters with the global model's state_dict.
        """
        self.model.load_state_dict(global_model_state_dict)
        self.model.to(self.device)

