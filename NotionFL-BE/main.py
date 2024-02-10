# main.py
import os
import yaml
import torch
from FL_Core.data_manager import get_mnist_data_loaders, split_data_for_clients
from FL_Core.client import FLClient
from FL_Core.server import FLServer
from models.model import MNISTModel
from utils.secure_aggregation import fedavg_aggregate

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
    global_model = MNISTModel()
    
    # Initialize clients
    clients = [FLClient(client_id=i, model=global_model, train_loader=client_data_loaders[f'client_{i}'], test_loader=test_loader, device=config['device'])
               for i in range(num_clients)]

    # Initialize the server
    server = FLServer(global_model)
    
    # Directories for metrics
    client_metrics_dir = "output/metrics/client"
    global_metrics_dir = "output/metrics/global"

    # Create directories if they don't exist
    os.makedirs(client_metrics_dir, exist_ok=True)
    os.makedirs(global_metrics_dir, exist_ok=True)

    # Federated Learning Process
    for round in range(config['fl_rounds']):
        print(f"\nFL Training Round {round + 1}/{config['fl_rounds']}")
        
        client_models = []
        client_states_before_aggregation = {}
        
        for client in clients:
            # Each client trains on their data
            client_state_dict = client.train(epochs, lr)
            metrics = client.evaluate()
            
            # Save client metrics
            client_metrics_file = os.path.join(client_metrics_dir, f"client_{client.client_id}_round_{round+1}.txt")
            with open(client_metrics_file, 'w') as file:
                file.write(str(metrics))
            
            client_models.append(client.model)
            
            # Save the client's model state before aggregation
            client_states_before_aggregation[f'client_{client.client_id}'] = client_state_dict
            torch.save(client_state_dict, f'output/modelupdates/client_{client.client_id}_round_{round + 1}.pth')

        # Perform FedAvg aggregation
        global_model = fedavg_aggregate(global_model, client_models)
        
        # Update clients with the new global model state
        global_model_state = global_model.state_dict()
        for client in clients:
            client.update_model(global_model_state)
        
        # Optionally evaluate the global model performance after each round
        if round % config['eval_every_n_rounds'] == 0 or round == config['fl_rounds'] - 1:
            global_metrics = server.evaluate_global_model(test_loader, config['device'])

            # Save global model metrics
            global_metrics_file = os.path.join(global_metrics_dir, f"global_model_round_{round+1}.txt")
            with open(global_metrics_file, 'w') as file:
                file.write(str(global_metrics))

    # Save the global model after all rounds of training
    torch.save(global_model.state_dict(), 'output/globalmodel/final_global_model.pth')

if __name__ == "__main__":
    main()
