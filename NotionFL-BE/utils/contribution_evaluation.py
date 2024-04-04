import itertools
import math
import os
import torch
import matplotlib.pyplot as plt


def calculate_shapley_values(total_rounds, num_clients, data_collector_path, model_evaluation_func, averaging_func, device):
    shapley_values = {client_id: 0 for client_id in range(num_clients)}

    for round_num in range(total_rounds):
        for client_id in range(num_clients):
            for subset_size in range(1, num_clients + 1):
                for subset in itertools.combinations(range(num_clients), subset_size):
                    if client_id not in subset:
                        continue

                    # Load models for each round in the current subset
                    subset_models = []
                    for other_client_id in subset:
                        model_path = os.path.join(data_collector_path, 'client', 'localModels', f'client_{other_client_id}_model_round_{round_num}.pt')
                        if os.path.exists(model_path):
                            model_state = torch.load(model_path)
                            subset_models.append(model_state)

                    if not subset_models:
                        continue

                    # Calculate the contribution of the client
                    value_with = model_evaluation_func(averaging_func(subset_models))

                    # Evaluate without the current client
                    if len(subset) > 1:
                        subset_models_without_client = [s for i, s in enumerate(subset_models) if subset[i] != client_id]
                        value_without = model_evaluation_func(averaging_func(subset_models_without_client))
                    else:
                        base_model_path = os.path.join(data_collector_path, 'global', 'models', f'global_model_round_{round_num}.pt')
                        base_model = torch.load(base_model_path)
                        value_without = model_evaluation_func(base_model)

                    # Update Shapley values
                    weight = (math.factorial(subset_size - 1) * math.factorial(num_clients - subset_size)) / math.factorial(num_clients)
                    shapley_values[client_id] += weight * (value_with - value_without)

    # Normalize Shapley values across all rounds
    for client_id in shapley_values:
        shapley_values[client_id] /= total_rounds

    shapley_plot = create_shapley_value_plot(shapley_values)
    return shapley_values, shapley_plot


def calculate_shapley_values_every_round(round_num, num_clients, data_collector_path, model_evaluation_func, averaging_func, device):
    """
    Calculate Shapley values for each client by loading their models.

    Args:
        round_num (int): The current round number.
        num_clients (int): Total number of clients.
        data_collector_path (str): Path to the directory where client models are stored.
        model_evaluation_func (func): Function to evaluate model update.
        averaging_func (func): Function used to average the model updates.
        device (str): Device to move models to for evaluation.

    Returns:
        dict: Shapley values for each client.
    """
    shapley_values = {client_id: 0 for client_id in range(num_clients)}

    for client_id in range(num_clients):
        for subset_size in range(1, num_clients + 1):
            for subset in itertools.combinations(range(num_clients), subset_size):
                if client_id not in subset:
                    continue

                # Load models for the current subset
                subset_models = []
                for other_client_id in subset:
                    model_path = os.path.join(data_collector_path, 'client', 'localModels', f'client_{other_client_id}_model_round_{round_num}.pt')
                    model_state = torch.load(model_path)
                    subset_models.append(model_state)

                # Calculate the contribution of the client
                value_with = model_evaluation_func(averaging_func(subset_models))

                # Handle case where subset becomes empty after excluding client_id
                if len(subset) > 1:
                    subset_models_without_client = [s for i, s in enumerate(subset_models) if subset[i] != client_id]
                    value_without = model_evaluation_func(averaging_func(subset_models_without_client))
                else:
                    # Use the base model value when the subset contains only the client_id
                    base_model_path = os.path.join(data_collector_path, 'global', 'models', f'global_model_round_{round_num}.pt')
                    base_model = torch.load(base_model_path)
                    value_without = model_evaluation_func(base_model)

                # Calculate marginal contribution
                weight = (math.factorial(subset_size - 1) * math.factorial(num_clients - subset_size)) / math.factorial(num_clients)
                shapley_values[client_id] += weight * (value_with - value_without)

    # Calculate and return the Shapley values and the plot
    shapley_plot = create_shapley_value_plot(shapley_values)
    return shapley_values, shapley_plot


def create_shapley_value_plot(shapley_values):
    clients = list(shapley_values.keys())
    values = list(shapley_values.values())

    fig = plt.figure(figsize=(10, 6))
    plt.bar(clients, values, color='blue')
    plt.xlabel('Client ID')
    plt.ylabel('Shapley Value')
    plt.title(f'Clients Contribution Evaluation Analysis)')
    plt.xticks(clients, [f'Client {client}' for client in clients])

    return fig


