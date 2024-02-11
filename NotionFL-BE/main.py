# main.py
import copy
import os
import yaml
import torch
from FL_Core.data_manager import get_mnist_data_loaders, split_data_for_clients
from FL_Core.client import FLClient
from FL_Core.server import FLServer
from models.model import MNISTModel
from utils.secure_aggregation import fedavg_aggregate, average_model_states, calculate_variance, calculate_aggregation_time_and_resources
from utils.privacy_module import apply_differential_privacy
from utils.contribution_evaluation import calculate_shapley_values
from utils.data_collector import DataCollector

def main():
    # Load configurations from a YAML file
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

    # Initialize DataCollector
    data_collector = DataCollector(output_dir='output/data_collector')
    
    # Configuration parameters
    num_clients = config['num_clients']
    epochs = config['epochs']
    batch_size = config['batch_size']
    lr = config['learning_rate']
    
    # Load data
    train_loader, test_loader = get_mnist_data_loaders(batch_size=batch_size)
    
    # Split data among clients
    client_data_loaders = split_data_for_clients(train_loader.dataset, num_clients=num_clients, batch_size=batch_size)

    # Initialize the global model
    global_model = MNISTModel().to(config['device'])
    
    # Initialize clients with DataCollector
    clients = [
        FLClient(client_id=i, model=global_model, train_loader=client_data_loaders[f'client_{i}'], test_loader=test_loader, device=config['device'], data_collector=data_collector)
        for i in range(num_clients)
    ]
    
    # Initialize the server
    server = FLServer(global_model)

    # Federated Learning Process
    for round in range(config['fl_rounds']):
        print(f"\nFL Training Round {round + 1}/{config['fl_rounds']}")

        client_updates = []
        client_states_before_aggregation = {}
        dp_metrics = {'privacy_params': [], 'model_accuracy': [], 'noise_distribution': [], 'computation_overheads': []}
        
        for client in clients:
            # Each client trains on their data
            model_updates = client.train_and_get_updates(epochs, lr)

            # Evaluate client performance and save metrics
            client.evaluate(round)
            
            # Collect the client's model state
            client_model_state = client.model.cpu().state_dict()
            data_collector.collect_client_model(client.client_id, client_model_state, round)

            # Collect the model updates sent to the server
            data_collector.collect_client_updates(client.client_id, model_updates)
              

            # Apply differential privacy to the model parameters
            dp_info = apply_differential_privacy(
                [param for param in client.model.parameters() if param.requires_grad],
                config['clip_threshold'],
                config['noise_multiplier'],
                config['device']
            )
            
            # Store differential privacy metrics
            dp_metrics['noise_distribution'] += dp_info['noise_stats']
            dp_metrics['computation_overheads'].append(dp_info['computation_time'])

            # Collect the model parameters for aggregation
            client_updates.append(client_model_state)

            # Keep track of client states before aggregation for Shapley Value calculation
            client_states_before_aggregation[client.client_id] = client_model_state

        # Collect differential privacy metrics
        data_collector.collect_differential_privacy_logs(round, dp_metrics)
        
        # Calculate variance before aggregation
        variance_before = calculate_variance(client_updates)

        # Perform FedAvg aggregation
        pre_aggregated_state_dict = copy.deepcopy(global_model.state_dict())
        aggregated_state_dict, time_overheads = calculate_aggregation_time_and_resources(global_model.state_dict(), client_updates)
        
        # Calculate variance after aggregation
        variance_after = calculate_variance([aggregated_state_dict])

        # Evaluate performance difference
        pre_aggregation_accuracy = server.evaluate_global_model(test_loader, config['device'])[1]
        global_model.load_state_dict(aggregated_state_dict)
        post_aggregation_accuracy = server.evaluate_global_model(test_loader, config['device'])[1]

        performance_difference = post_aggregation_accuracy - pre_aggregation_accuracy

        # Collect aggregation metrics
        aggregation_metrics = {
            'variance_before_aggregation': variance_before,
            'variance_after_aggregation': variance_after,
            'performance_difference': performance_difference
        }
        data_collector.collect_secure_aggregation_logs(round, aggregation_metrics, time_overheads)

        # Update each client model with the new global state
        for client in clients:
            client.update_model(global_model.state_dict())

        # Evaluate global model performance after each round
        if round % config['eval_every_n_rounds'] == 0 or round == config['fl_rounds'] - 1:
            global_metrics = server.evaluate_global_model(test_loader, config['device'])
            # Optionally save global model metrics
            data_collector.collect_global_model_metrics(round, global_metrics)

        # Calculate Shapley Values
        model_evaluation_func = lambda model_state: server.evaluate_model_state_dict(model_state, test_loader, config['device'])[1] # Accuracy
        shapley_values = calculate_shapley_values(client_states_before_aggregation, model_evaluation_func, average_model_states, global_model, config['device'])
        print(f"Shapley Values for round {round + 1}: {shapley_values}")

        # Optionally save Shapley values
        data_collector.collect_contribution_eval_metrics(round, shapley_values)

    # Save the global model after all rounds of training
    data_collector.collect_global_model(global_model.cpu().state_dict(), round)
    

if __name__ == "__main__":
    main()
