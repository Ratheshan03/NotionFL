import torch

def fedavg_aggregate(global_model, client_models):
    """
    Aggregate client models into a global model using Federated Averaging.

    Args:
        global_model (nn.Module): The global model to be updated.
        client_models (list of nn.Module): List of client models.

    Returns:
        nn.Module: Updated global model.
    """
    # Ensure there is at least one client model
    if not client_models:
        raise ValueError("No client models provided for aggregation.")

    # Initialize a dictionary to store the averaged parameters
    global_state_dict = global_model.state_dict()
    num_clients = len(client_models)

    # Aggregate each parameter
    for key in global_state_dict.keys():
        # Sum the parameter values from each client model
        avg_param = sum(client_model.state_dict()[key] for client_model in client_models) / num_clients
        global_state_dict[key] = avg_param

    # Update the global model
    global_model.load_state_dict(global_state_dict)
    return global_model
