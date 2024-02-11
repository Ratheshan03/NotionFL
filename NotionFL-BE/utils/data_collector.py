import json
import os
import numpy as np

import torch
class DataCollector:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Initialize storage structures for various metrics and logs
        self.client_training_logs = {}
        self.client_models = {}
        self.client_updates = {}
        self.global_model_metrics = {}
        self.contribution_eval_metrics = {}
        self.secure_aggregation_logs = {}
        self.differential_privacy_logs = {}

    def collect_client_training_logs(self, client_id, training_logs):
        """
        Collect and store training logs for each client.

        Args:
            client_id (int): The unique identifier of the client.
            training_logs (dict): A dictionary containing training logs and metrics.
        """
        client_training_logs_dir = os.path.join(self.output_dir, 'client', 'training')
        file_path = os.path.join(client_training_logs_dir, f"client_{client_id}_training_logs.json")
        os.makedirs(client_training_logs_dir, exist_ok=True)
        
        with open(file_path, 'a') as file:
            json.dump(training_logs, file)
            file.write("\n")
        
        
    def collect_client_evaluation_logs(self, client_id, evaluation_logs, round):
        client_evaluation_logs_dir = os.path.join(self.output_dir, 'client', 'evaluation')
        file_path = os.path.join(client_evaluation_logs_dir, f"client_{client_id}_evaluation_logs_round_{round + 1}.json")
        os.makedirs(client_evaluation_logs_dir, exist_ok=True)
        
        with open(file_path, 'a') as file:  # Append to the file if it exists
            json.dump(evaluation_logs, file)
            file.write("\n")
            

    def collect_client_model(self, client_id, model, round_num):
        """
        Collect and store client models for each round.

        Args:
            client_id (int): The unique identifier of the client.
            model (nn.Module): The PyTorch model of the client.
            round_num (int): The current round number in federated learning.
        """
        client_models_dir = os.path.join(self.output_dir, 'client', 'localModels')
        model_path = os.path.join(client_models_dir, f"client_{client_id}_model_round_{round_num}.pt")
        os.makedirs(client_models_dir, exist_ok=True)
        
        torch.save(model, model_path)

    def collect_client_updates(self, client_id, model_update):
        """
        Collect and store model updates sent to the server.

        Args:
            client_id (int): The unique identifier of the client.
            model_update (OrderedDict): The state_dict representing the model update.
        """
        client_update_dir = os.path.join(self.output_dir, 'client', 'modelUpdates')
        update_path = os.path.join(client_update_dir, f"client_{client_id}_update.pt")
        os.makedirs(client_update_dir, exist_ok=True)
        
        torch.save(model_update, update_path)


    def collect_global_model_metrics(self, round_num, model_metrics):
        # Convert the metrics tuple into a dictionary
        metrics_dict = {
            'test_loss': model_metrics[0],
            'accuracy': model_metrics[1],
            'precision': model_metrics[2],
            'recall': model_metrics[3],
            'f1': model_metrics[4],
            'conf_matrix': model_metrics[5].tolist() if isinstance(model_metrics[5], np.ndarray) else model_metrics[5]
        }

        # Save global model metrics
        global_metrics_dir = os.path.join(self.output_dir, 'global', 'evaluation')
        global_metrics_path = os.path.join(global_metrics_dir, f'global_model_round_{round_num + 1}.json')
        os.makedirs(global_metrics_dir, exist_ok=True)
        
        with open(global_metrics_path, 'w') as file:
            json.dump(metrics_dict, file)
            file.write("\n")

            
            
    def collect_global_model(self, global_model_state, round_num):
        """
        Collect and store the global model state after each round.

        Args:
            global_model_state (OrderedDict): The state dictionary of the global model.
            round_num (int): The current round number in the Federated Learning process.
        """
        global_model_dir = os.path.join(self.output_dir, 'global', 'models')
        file_path = os.path.join(global_model_dir, f'global_model_round_{round_num}.pt')
        os.makedirs(global_model_dir, exist_ok=True)
        
        torch.save(global_model_state, file_path)
            

    def collect_contribution_eval_metrics(self, round_num, contribution_metrics):
        contribution_metrics_dir = os.path.join(self.output_dir, 'client', 'contribution')
        shapley_values_path = os.path.join(contribution_metrics_dir, f"client_shapley_values_round_{round_num + 1}.json")
        os.makedirs(contribution_metrics_dir, exist_ok=True)
        with open(shapley_values_path, 'w') as file:
            json.dump(contribution_metrics, file)
            file.write("\n")

    def collect_secure_aggregation_logs(self, round_num, aggregation_metrics, time_overheads):
        """
        Collect and store logs related to secure aggregation.

        Args:
            round_num (int): The current round number in the Federated Learning process.
            aggregation_metrics (dict): Metrics demonstrating the effectiveness of the aggregation method.
            time_overheads (dict): Time taken and computational resources used for secure aggregation.
        """
        secure_aggregation_data = {
            'round': round_num,
            'aggregation_metrics': aggregation_metrics,
            'time_overheads': time_overheads
        }
        aggregation_dir = os.path.join(self.output_dir, 'aggregation')
        file_path = os.path.join(aggregation_dir, f'secure_aggregation_round_{round_num}.json')
        os.makedirs(aggregation_dir, exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(secure_aggregation_data, file, indent=4)
            

    def collect_differential_privacy_logs(self, round_num, dp_metrics):
        """
        Collect and store logs related to differential privacy.

        Args:
            round_num (int): The current round number.
            dp_metrics (dict): Dictionary containing differential privacy metrics including noise statistics and computation overheads.
        """
        privacy_dir = os.path.join(self.output_dir, 'privacy')
        file_path = os.path.join(privacy_dir, f"differential_privacy_round_{round_num}.json")
        os.makedirs(privacy_dir, exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(dp_metrics, file, indent=4)

    
    

    # Additional methods for saving, loading, and processing the collected data can be added as needed
