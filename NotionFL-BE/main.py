# main.py
import copy
import os
import shap
import yaml
import torch
import logging
from FL_Core.data_manager import get_data_loaders, split_client_data
from FL_Core.client import FLClient
from FL_Core.server import FLServer
from models.model import MNISTModel, CIFAR10Model
from utils.secure_aggregation import fedavg_aggregate, average_model_states, calculate_variance, perform_fedavg_aggregation
from utils.privacy_module import apply_differential_privacy
from utils.contribution_evaluation import calculate_shapley_values
from utils.data_collector import DataCollector
from utils.federated_xai import FederatedXAI
from utils.allocate_incentive import allocate_and_save_incentives

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Load training configurations from a YAML file
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

    # Configuration parameters
    dataset_name = config['dataset']
    num_clients = config['num_clients']
    epochs = config['epochs']
    batch_size = config['batch_size']
    lr = config['learning_rate']
    fl_rounds = config['fl_rounds']
    
    # Loading and splitting data based on dataset selection
    train_loader, test_loader = get_data_loaders(dataset_name, batch_size=batch_size)
    client_data_loaders = split_client_data(train_loader.dataset, num_clients=num_clients, batch_size=batch_size)

    # Initialize the global model based on dataset selection
    if dataset_name == 'MNIST':
        global_model = MNISTModel().to(config['device'])
    elif dataset_name == 'CIFAR10':
        global_model = CIFAR10Model().to(config['device'])
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    
    # Initialize DataCollector and FederatedXAI
    data_collector = DataCollector(output_dir='output/data_collector')
    federated_xai = FederatedXAI(data_collector_path=data_collector.output_dir, device=config['device'], global_model=global_model)
    
    # Initializing FL clients
    clients = [
        FLClient(client_id=i, model=global_model, train_loader=client_data_loaders[f'client_{i}'], test_loader=test_loader, device=config['device'], data_collector=data_collector)
        for i in range(num_clients)
    ]
    
    # Initializing the FL server
    server = FLServer(global_model)

    # Federated Learning Process
    for round in range(fl_rounds):
        print(f"\nFL Training Round {round + 1}/{fl_rounds}")

        client_updates = []
        client_states_before_aggregation = {}
        dp_metrics = {'privacy_params': [], 'model_accuracy': [], 'noise_distribution': [], 'computation_overheads': []}
        
        for client in clients:
            # local training and getting model updates
            model_updates = client.train_and_get_updates(epochs, lr)

            # Evaluate client performance and save metrics
            client.evaluate(round)
            
            # Store the client's model state
            client_model_state = client.model.cpu().state_dict()
            data_collector.collect_client_model(client.client_id, client_model_state, round)

            # Store the client model updates
            data_collector.collect_client_updates(client.client_id, model_updates)
              
            # Explain client model each round
            shap_plot, (shap_numpy, test_numpy) = federated_xai.explain_client_model(client.client_id, round, test_loader)
            data_collector.save_shap_explanation_plot(shap_plot, f'client_{client.client_id}', round)

            # Apply differential privacy to the model parameters
            logging.info(f"\nAdding differential privacy for client_{client.client_id}'s round {round} model")
            dp_info = apply_differential_privacy(
                [param for param in client.model.parameters() if param.requires_grad],
                config['clip_threshold'],
                config['noise_multiplier'],
                config['device']
            )
            
            # Store differential privacy metrics
            dp_metrics['noise_distribution'] += dp_info['noise_stats']
            dp_metrics['computation_overheads'].append(dp_info['computation_time'])
            
            # Storing the client model with DP 
            private_model_state = client.model.cpu().state_dict()
            data_collector.collect_client_model(client.client_id, private_model_state, round, suffix='private')
             
            # Explain the impact of differential privacy on this client's model
            privacy_explanation_text, privacy_plot_buffer = federated_xai.explain_privacy_impact(
                client.client_id, round, test_loader, config['noise_multiplier']
            )

            # Save the privacy explanation and plot using the DataCollector
            data_collector.save_privacy_explanations(
                privacy_explanation_text, privacy_plot_buffer, client.client_id, round
            )
            
            # Collect the model parameters for aggregation
            client_updates.append(client_model_state)

            # Keep track of client states before aggregation for Shapley Value calculation
            client_states_before_aggregation[client.client_id] = client_model_state

        # Collect differential privacy metrics
        data_collector.collect_differential_privacy_logs(round, dp_metrics)
        
        # Calculate variance before aggregation
        variance_before = calculate_variance(client_updates)

        # Perform FedAvg aggregation
        logging.info(f'Performing secured aggregation for round: {round}')
        pre_aggregated_state_dict = copy.deepcopy(global_model.state_dict())
        aggregated_state_dict, time_overheads = perform_fedavg_aggregation(global_model.state_dict(), client_updates)

        # Calculate variance after aggregation
        variance_after = calculate_variance([aggregated_state_dict])
        
        # Explain aggregation
        aggregated_explanation = federated_xai.explain_aggregation(pre_aggregated_state_dict, aggregated_state_dict, test_loader, round)
        data_collector.save_aggregation_explanation(aggregated_explanation, round)
        
        # Evaluate performance difference
        pre_aggregation_accuracy = server.evaluate_model_state(pre_aggregated_state_dict, test_loader, config['device'])
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

        # Evaluate global model performance conditionally
        if (round + 1) % config['eval_every_n_rounds'] == 0 or round == config['fl_rounds'] - 1:
            print(f"Evaluating global model performance at round {round + 1}")
            global_metrics = server.evaluate_global_model(test_loader, config['device'])

            # Save global model metrics
            data_collector.collect_global_model_metrics(round, global_metrics)
            
        # Save the global model after all rounds of training
        data_collector.collect_global_model(global_model.cpu().state_dict(), round)


    # Define the model evaluation function
    model_evaluation_func = lambda model_state: server.evaluate_model_state_dict(model_state, test_loader, config['device'], dataset_name)

    # Define the averaging function
    averaging_func = lambda model_states: server.fedavg_aggregate(model_states)

    # Calculate Contribution evaluation Shapley Values
    shapley_values = calculate_shapley_values(
            round_num=round,
            num_clients=num_clients,
            data_collector_path=data_collector.output_dir,
            model_evaluation_func=model_evaluation_func,
            averaging_func=averaging_func,
            device=config['device']
    )
    print(f"Shapley Values for round {round + 1}: {shapley_values}")

    # Optionally save Shapley values
    data_collector.collect_contribution_eval_metrics(round, shapley_values)
        
    # Allocate incentives based on Shapley values
    allocate_and_save_incentives(round+1)
    # Call the function to generate and save the explanation
    federated_xai.generate_incentive_explanation(round+1)
    
    # Explain the global model using SHAP
    shap_numpy, test_numpy = federated_xai.explain_global_model(round, test_loader)

    # Save the SHAP explanation plot for the global model
    data_collector.save_shap_explanation_plot(shap_numpy, test_numpy, 'global', round+1)
    
    # After explaining all client and global models
    comparison_plot = federated_xai.compare_models(round+1, num_clients)

    data_collector.save_comparison_plot(comparison_plot, round+1)

    explanations = federated_xai.compare_model_shap_values(round, num_clients, test_loader)
    for client_id, explanation in explanations.items():
        data_collector.save_evaluation_plot(explanation['comparison_plot'], client_id, round)
 

if __name__ == "__main__":
    main()
