# main.py
import yaml
import torch
from FL_Core.data_manager import get_mnist_data_loaders, split_data_for_clients
from FL_Core.client import FLClient
from FL_Core.server import FLServer
from models.model import MNISTModel

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

    # Federated Learning Process
    for round in range(config['fl_rounds']):
        client_updates = []
        for client in clients:
            # Each client trains on their data
            client_update = client.train(epochs, lr)
            client_updates.append(client_update)

        # Server aggregates the updates
        global_model_state = server.aggregate_client_updates(client_updates)
        
        # Clients update their local models
        for client in clients:
            client.update_model(global_model_state)
        
        # Optionally evaluate the global model performance after each round
        if round % config['eval_every_n_rounds'] == 0 or round == config['fl_rounds'] - 1:
            server.evaluate_global_model(test_loader, config['device'])

    # Save the global model after all rounds of training
    torch.save(global_model.state_dict(), 'output/global_model.pth')

if __name__ == "__main__":
    main()
