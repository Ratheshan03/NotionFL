# main.py
import os
import yaml
import torch
from FL_Core.data_manager import get_mnist_data_loaders, split_data_for_clients
from FL_Core.client import FLClient
from FL_Core.server import FLServer
from models.model import MNISTModel
from utils.secure_aggregation import fedavg_aggregate
from utils.privacy_module import apply_differential_privacy

def main():
    # Load configurations from a YAML file
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

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
    
    # Initialize clients
    clients = [FLClient(client_id=i, model=global_model, train_loader=client_data_loaders[f'client_{i}'], test_loader=test_loader, device=config['device'])
               for i in range(num_clients)]

    # Initialize the server
    server = FLServer(global_model)

    # Federated Learning Process
    for round in range(config['fl_rounds']):
        print(f"\nFL Training Round {round + 1}/{config['fl_rounds']}")

        client_updates = []
        for client in clients:
            # Each client trains on their data
            client.train(epochs, lr)
            
            # Evaluate client performance and save metrics
            client_metrics = client.evaluate()
            client_metrics_path = os.path.join('output/metrics/client', f'client_{client.client_id}_round_{round + 1}.txt')
            torch.save(client_metrics, client_metrics_path)
            
            # Apply differential privacy to the model parameters
            for param in client.model.parameters():
                if param.requires_grad:  # Apply only if the parameter requires gradients
                    apply_differential_privacy([param], config['clip_threshold'], config['noise_multiplier'], config['device'])

            # Collect the model parameters for aggregation
            client_state_dict = client.model.cpu().state_dict()
            client_updates.append(client_state_dict)
            
            # Save the client's model state before aggregation
            client_state_path = os.path.join('output/modelupdates', f'client_{client.client_id}_round_{round + 1}.pt')
            torch.save(client_state_dict, client_state_path)

        # Perform FedAvg aggregation
        client_state_dicts = [client.model.state_dict() for client in clients]
        aggregated_state_dict = fedavg_aggregate(global_model.state_dict(), client_state_dicts)
        
        # Load the aggregated state_dict into global_model
        global_model.load_state_dict(aggregated_state_dict)
        global_model.to(config['device'])

        # Update each client model with the new global state
        for client in clients:
            client.update_model(global_model.state_dict())

        # Evaluate global model performance after each round
        if round % config['eval_every_n_rounds'] == 0 or round == config['fl_rounds'] - 1:
            global_metrics = server.evaluate_global_model(test_loader, config['device'])

            # Save global model metrics
            global_metrics_path = os.path.join('output/metrics/global', f'global_model_round_{round + 1}.txt')
            torch.save(global_metrics, global_metrics_path)

    # Save the global model after all rounds of training
    global_model_path = os.path.join('output/globalmodel', 'final_global_model.pt')
    torch.save(global_model.cpu().state_dict(), global_model_path)

if __name__ == "__main__":
    main()
