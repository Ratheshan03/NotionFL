import os
import time
import torch
import copy

def fedavg_aggregate(global_state_dict, client_state_dicts):
    """
    Aggregate client models' state_dicts into a global model's state_dict using Federated Averaging.

    Args:
        global_state_dict (OrderedDict): The state_dict of the global model to be updated.
        client_state_dicts (list of OrderedDict): List of state_dicts from client models.

    Returns:
        OrderedDict: Updated state_dict for the global model.
    """
    # Ensure there is at least one client state_dict provided for aggregation
    if not client_state_dicts:
        raise ValueError("No client model state_dicts provided for aggregation.")

    num_clients = len(client_state_dicts)

    # Aggregate each parameter
    for key in global_state_dict.keys():
        # Sum the parameter values from each client state_dict
        avg_param = sum(client_state_dict[key] for client_state_dict in client_state_dicts) / num_clients
        global_state_dict[key] = avg_param

    return global_state_dict


def average_model_states(base_model, model_states):
    """
    Average the state dictionaries of a list of models.

    Args:
        base_model (torch.nn.Module): The model whose structure we'll use for averaging.
        model_states (list): List of state_dicts from different models.

    Returns:
        OrderedDict: Averaged state dictionary.
    """
    averaged_state_dict = copy.deepcopy(base_model.state_dict())
    num_models = len(model_states)

    # Ensure there are models to average
    if num_models == 0:
        raise ValueError("No model states provided for averaging.")

    for key in averaged_state_dict.keys():
        # Sum the parameter values from each model state
        avg_param = sum(state_dict[key] for state_dict in model_states) / num_models
        averaged_state_dict[key] = avg_param

    return averaged_state_dict


def calculate_variance(models):
    """ Calculate the variance among a list of model state_dicts. """
    sum_state_dict = None
    for model in models:
        if sum_state_dict is None:
            sum_state_dict = {k: v.clone().detach() for k, v in model.items()}
        else:
            for k, v in model.items():
                sum_state_dict[k] += v
    mean_state_dict = {k: v / len(models) for k, v in sum_state_dict.items()}

    variance = 0
    for model in models:
        for k, v in model.items():
            variance += ((v - mean_state_dict[k]) ** 2).sum().item()
    return variance / len(models)


def calculate_aggregation_time_and_resources(global_state_dict, client_state_dicts):
    """
    Calculate the time taken for aggregation and estimate computational resources used.

    Args:
        global_state_dict (OrderedDict): The state_dict of the global model to be updated.
        client_state_dicts (list of OrderedDict): List of state_dicts from client models.

    Returns:
        dict: Aggregation time and estimated computational resources.
    """
    start_time = time.time()
    aggregated_state_dict = fedavg_aggregate(global_state_dict, client_state_dicts)
    end_time = time.time()

    time_overheads = {
        'aggregation_time': end_time - start_time,
        'computational_resources': {
            'cpu_usage': os.cpu_count(),  # Example metric, you can add more complex metrics here
            # 'memory_usage': ...,  # Additional resource metrics can be added
            # 'gpu_usage': ...      # If GPUs are involved
        }
    }
    
    return aggregated_state_dict, time_overheads

