# FL_Core/server.py

import torch

class FLServer:
    def __init__(self, global_model):
        self.global_model = global_model

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
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = self.global_model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = correct / len(test_loader.dataset)

        print(f'Global Model Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')
        
        return test_loss, accuracy
