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
