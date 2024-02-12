# FL_Core/server.py

import json
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import copy
import torch.nn as nn
from models.model import MNISTModel

class FLServer:
    def __init__(self, global_model):
        self.global_model = global_model
        
    def set_global_model_state(self, state_dict):
        """Temporarily sets the global model to the given state_dict."""
        self.global_model.load_state_dict(state_dict)

    def aggregate_client_updates(self, client_updates, aggregation_method='average'):
        """
        Aggregate the updates from the clients to update the global model.
        Currently, it supports simple averaging of the updates.
        """
        # Assume all client updates are of the same structure
        global_dict = self.global_model.state_dict()

        # Initialize a dictionary to store the aggregated update
        aggregated_updates = {k: torch.zeros_like(v) for k, v in global_dict.items()}

        # Aggregate updates
        for update in client_updates:
            for k, v in update.items():
                aggregated_updates[k] += v / len(client_updates)
        
        # Update global model
        self.global_model.load_state_dict(aggregated_updates)
        
        return self.global_model.state_dict()

    def evaluate_global_model(self, test_loader, device):
        """
        Evaluate the global model's performance on a test dataset.
        """
        self.global_model.to(device)
        self.global_model.eval()
        
        test_loss = 0
        correct = 0
        all_targets = []
        all_predictions = []
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = self.global_model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

                # Append actual and predicted labels for further metrics
                all_targets.extend(target.view_as(pred).cpu().numpy())
                all_predictions.extend(pred.cpu().numpy())

        test_loss /= len(test_loader.dataset)
        accuracy = correct / len(test_loader.dataset)

        # Calculate additional metrics
        precision = precision_score(all_targets, all_predictions, average='weighted')
        recall = recall_score(all_targets, all_predictions, average='weighted')
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        conf_matrix = confusion_matrix(all_targets, all_predictions)

        # Print all metrics
        print(f'Global Model Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
        print(f'Confusion Matrix:\n{conf_matrix}')
        
        return test_loss, accuracy, precision, recall, f1, conf_matrix
    
    
    def evaluate_model_state(self, model_state, test_loader, device, model_template):
        """
        Evaluate a model state using the test data loader.

        Args:
        model_state (OrderedDict): State dictionary of the model to be evaluated.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to perform the computation on.
        model_template (torch.nn.Module): Template model to load the state into.

        Returns:
        float: The evaluation metric (e.g., accuracy) of the model.
        """
        model = copy.deepcopy(model_template)
        model.load_state_dict(model_state)
        model.to(device)
        model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = correct / total
        return accuracy
    
    
    def evaluate_model_state_dict(self, model_state_dict, test_loader, device):
        """
        Evaluate a model's performance given its state dictionary.

        Args:
            model_state_dict (OrderedDict): The state dictionary of the model to evaluate.
            test_loader (DataLoader): The test dataset loader.
            device (torch.device): The device to perform evaluation on.

        Returns:
            float: The evaluation metric (e.g., accuracy).
        """
        # Create a new instance of the model and load the state dict
       
        
        # Load state dict into the global model and evaluate
        self.global_model.load_state_dict(model_state_dict)
        self.global_model.to(device)
        return self.evaluate_global_model(test_loader, device)
    
    
    def evaluate_model(self, model, test_loader, device):
        """
        Evaluate a model's performance given the model instance.

        Args:
            model (torch.nn.Module): The model to evaluate.
            test_loader (DataLoader): The test dataset loader.
            device (torch.device): The device to perform evaluation on.

        Returns:
            float: The evaluation metric (e.g., accuracy).
        """
        model.to(device)
        model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = correct / total
        return accuracy
    
    def evaluate_models(self, models, test_loader, device):
        """Evaluates a list of models and returns the average metric."""
        metrics = [self.evaluate_model(model, test_loader, device) for model in models]
        return sum(metrics) / len(metrics)
    
    
    def fedavg_aggregate(self, model_states):
        """
        Aggregate the model states by averaging the parameters.

        Args:
            model_states (list): A list of state_dicts of the models to be averaged.

        Returns:
            OrderedDict: The averaged state_dict.
        """
        # Initialize a dictionary to store the aggregated weights
        avg_weights = {}

        # Iterate through each parameter
        for key in model_states[0].keys():
            # Sum the weights of each model for this parameter
            sum_weights = sum([model_state[key] for model_state in model_states])
            # Calculate the average weight
            avg_weights[key] = sum_weights / len(model_states)

        return avg_weights
