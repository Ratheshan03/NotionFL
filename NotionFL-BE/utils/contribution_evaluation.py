import itertools
import copy
import math
import os
import torch
import matplotlib.pyplot as plt


# def calculate_shapley_values(total_rounds, num_clients, data_collector_path, model_evaluation_func, averaging_func, device):
#     shapley_values = {client_id: 0 for client_id in range(num_clients)}

#     for round_num in range(total_rounds):
#         for client_id in range(num_clients):
#             for subset_size in range(1, num_clients + 1):
#                 for subset in itertools.combinations(range(num_clients), subset_size):
#                     if client_id not in subset:
#                         continue

#                     # Load models for each round in the current subset
#                     subset_models = []
#                     for other_client_id in subset:
#                         model_path = os.path.join(data_collector_path, 'client', 'localModels', f'client_{other_client_id}_model_round_{round_num}.pt')
#                         if os.path.exists(model_path):
#                             model_state = torch.load(model_path)
#                             subset_models.append(model_state)

#                     if not subset_models:
#                         continue

#                     # Calculate the contribution of the client
#                     value_with = model_evaluation_func(averaging_func(subset_models))

#                     # Evaluate without the current client
#                     if len(subset) > 1:
#                         subset_models_without_client = [s for i, s in enumerate(subset_models) if subset[i] != client_id]
#                         value_without = model_evaluation_func(averaging_func(subset_models_without_client))
#                     else:
#                         base_model_path = os.path.join(data_collector_path, 'global', 'models', f'global_model_round_{round_num}.pt')
#                         base_model = torch.load(base_model_path)
#                         value_without = model_evaluation_func(base_model)

#                     # Update Shapley values
#                     weight = (math.factorial(subset_size - 1) * math.factorial(num_clients - subset_size)) / math.factorial(num_clients)
#                     shapley_values[client_id] += weight * (value_with - value_without)

#     # Normalize Shapley values across all rounds
#     for client_id in shapley_values:
#         shapley_values[client_id] /= total_rounds

#     shapley_plot = create_shapley_value_plot(shapley_values)
#     return shapley_values, shapley_plot


def calculate_shapley_values(total_rounds, num_clients, client_models, global_models, model_evaluation_func, averaging_func, device):
    shapley_values = {client_id: 0 for client_id in range(num_clients)}

    for round_num in range(total_rounds):
        for client_id in range(num_clients):
            for subset_size in range(1, num_clients + 1):
                for subset in itertools.combinations(range(num_clients), subset_size):
                    if client_id not in subset:
                        continue

                    # Retrieve models for each round in the current subset
                    subset_models = [client_models[other_client_id][round_num] for other_client_id in subset if other_client_id in client_models and round_num in client_models[other_client_id]]

                    if not subset_models:
                        continue

                    # Calculate the contribution of the client
                    value_with = model_evaluation_func(averaging_func(subset_models))

                    # Evaluate without the current client
                    if len(subset) > 1:
                        subset_models_without_client = [s for i, s in enumerate(subset_models) if subset[i] != client_id]
                        value_without = model_evaluation_func(averaging_func(subset_models_without_client))
                    else:
                        base_model = global_models[round_num]
                        value_without = model_evaluation_func(base_model)

                    # Update Shapley values
                    weight = (math.factorial(subset_size - 1) * math.factorial(num_clients - subset_size)) / math.factorial(num_clients)
                    shapley_values[client_id] += weight * (value_with - value_without)

    # Normalize Shapley values across all rounds
    for client_id in shapley_values:
        shapley_values[client_id] /= total_rounds

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


def calculate_federated_shapley_values(client_models, model_evaluation_func, averaging_func, global_models=None, cache=None):
    """
    Calculates Shapley values for clients in federated learning.

    :param client_models: Dict of client model states for each round {client_id: {round_num: model_state}}.
    :param model_evaluation_func: Function to evaluate a model's performance.
    :param averaging_func: Function to average model states.
    :param global_models: (Optional) Dict of global model states for each round {round_num: model_state}.
    :param cache: (Optional) Cache to store intermediate computations.
    :return: Dict of calculated Shapley values for each client.
    """

    if cache is None:
        cache = {}

    sv = {client_id: 0 for client_id in client_models.keys()}

    # Iterate over all possible combinations of clients
    for subset in itertools.chain.from_iterable(itertools.combinations(client_models.keys(), r) for r in range(1, len(client_models) + 1)):
        for client_id in subset:
            without_client = tuple(sorted(set(subset) - {client_id}))
            with_client = tuple(sorted(set(subset)))

            # Check cache to avoid redundant computation
            if with_client in cache:
                value_with = cache[with_client]
            else:
                combined_model = averaging_func([copy.deepcopy(client_models[other_client_id]) for other_client_id in with_client])
                value_with = model_evaluation_func(combined_model)
                cache[with_client] = value_with

            if without_client in cache:
                value_without = cache[without_client]
            else:
                if without_client:
                    combined_model = averaging_func([copy.deepcopy(client_models[other_client_id]) for other_client_id in without_client])
                    value_without = model_evaluation_func(combined_model)
                else:
                    # Use global model if available and no other clients are in subset
                    value_without = model_evaluation_func(global_models) if global_models else 0
                cache[without_client] = value_without

            # Compute the marginal contribution
            marginal_contribution = value_with - value_without

            # Calculate Shapley Value
            sv[client_id] += marginal_contribution / len(subset)

    # Average the Shapley Values over all permutations
    total_permutations = len(list(itertools.permutations(client_models.keys())))
    for client_id in sv:
        sv[client_id] /= total_permutations

    return sv
