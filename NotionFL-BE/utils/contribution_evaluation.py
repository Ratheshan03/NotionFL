import copy
import itertools
import torch


def calculate_shapley_values(models, model_evaluation_func, averaging_func, base_model, device):
    """
    Computes the Shapley Value for clients.

    Parameters:
    models (dict): Key-value pair of client identifiers and model updates.
    model_evaluation_func (func): Function to evaluate model update.
    averaging_func (func): Function used to average the model updates.
    base_model (torch.nn.Module): Base model to use for averaging.
    device (str): Device to move models to for evaluation.

    Returns:
    dict: Key-value pair of client identifiers and the computed Shapley values.
    """

    all_perms = list(itertools.permutations(models.keys()))
    marginal_contributions = []
    history = {}

    for perm in all_perms:
        perm_values = {}
        local_models = []

        for i, client_id in enumerate(perm):
            model_state = copy.deepcopy(models[client_id])
            local_models.append(model_state)

            perm_key = ','.join(str(cid) for cid in perm[:i+1])

            if perm_key in history:
                current_value = history[perm_key]
            else:
                # Average the model states
                averaged_state = averaging_func(base_model, local_models)
                # Load averaged state into a copy of the base model
                model_copy = copy.deepcopy(base_model)
                model_copy.load_state_dict(averaged_state)
                model_copy.to(device)
                # Evaluate the model
                model_copy_state_dict = model_copy.state_dict() if isinstance(model_copy, torch.nn.Module) else model_copy
                current_value = model_evaluation_func(model_copy_state_dict)
                history[perm_key] = current_value

            perm_values[client_id] = max(0, current_value - sum(perm_values.values()))

        marginal_contributions.append(perm_values)

    sv = {client_id: sum(perm[client_id] for perm in marginal_contributions) / len(all_perms) for client_id in models.keys()}
    return sv
