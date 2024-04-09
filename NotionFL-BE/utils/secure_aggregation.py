import os
import time
import torch
import copy
import psutil

def perform_fedavg_aggregation(global_state_dict, client_state_dicts, client_weights):
    """
    Perform Federated Averaging aggregation using weights.

    Args:
        global_state_dict (OrderedDict): The state_dict of the global model to be updated.
        client_state_dicts (list of OrderedDict): List of state_dicts from client models.
        client_weights (list): List of weights for each client model based on the data size.

    Returns:
        OrderedDict: Updated state_dict for the global model.
        dict: Aggregation time and estimated computational resources.
    """
    if not client_state_dicts:
        raise ValueError("No client model state_dicts provided for aggregation.")

    start_time = time.time()
    start_memory = psutil.virtual_memory()

    aggregated_state_dict = copy.deepcopy(global_state_dict)

    # Aggregate each parameter
    for key in aggregated_state_dict.keys():
        aggregated_state_dict[key] = sum(client_state_dict[key] * weight for client_state_dict, weight in zip(client_state_dicts, client_weights)) / sum(client_weights)

    end_time = time.time()
    end_memory = psutil.virtual_memory()
    memory_used = start_memory.used - end_memory.used

    time_overheads = {
        'aggregation_time': end_time - start_time,
        'computational_resources': {
            'cpu_usage': os.cpu_count(),
            'memory_usage': memory_used,
        }
    }

    return aggregated_state_dict, time_overheads


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
    """ 
    Calculate the variance among a list of model state_dicts. 
    provides insights into how diverse the client models are.
    
    """
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

