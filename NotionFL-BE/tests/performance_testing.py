import time
import requests
import json

def send_training_request(config):
    API_ENDPOINT = 'http://localhost:5000/start_training'
    headers = {'Content-Type': 'application/json'}
    response = requests.post(API_ENDPOINT, json=config, headers=headers)

    if response.status_code == 202:
        response_data = response.json()
        if response_data.get("status") == "Training started":
            return response_data.get("training_id")
        else:
            raise Exception("Training did not start successfully.")
    else:
        raise Exception(f"Failed to start training: {response.status_code} - {response.text}")


def check_training_status(training_id):
    API_STATUS_ENDPOINT = f'http://localhost:5000/training_status/{training_id}'
    while True:
        response = requests.get(API_STATUS_ENDPOINT)
        if response.status_code != 200:
            raise Exception(f"Failed to get training status: {response.text}")
        
        status = response.json()['status']
        if status in ['Completed', 'Failed']:
            return status
        time.sleep(60)  # Wait for 60 seconds before checking again

def performance_test():
    configurations = [
        {'num_clients': 2, 'epochs': 2, 'fl_rounds': 1, 'eval_every_n_rounds': 1, 'device': 'cpu', 'batch_size': 64, 'learning_rate': 0.01, 'clip_threshold': 1.0, 'noise_multiplier': 0.1},
        {'num_clients': 3, 'epochs': 5, 'fl_rounds': 1, 'eval_every_n_rounds': 1, 'device': 'cpu', 'batch_size': 64, 'learning_rate': 0.01, 'clip_threshold': 1.0, 'noise_multiplier': 0.1},
        {'num_clients': 4, 'epochs': 10, 'fl_rounds': 2, 'eval_every_n_rounds': 1, 'device': 'cpu', 'batch_size': 64, 'learning_rate': 0.01, 'clip_threshold': 1.0, 'noise_multiplier': 0.1},
        # Add more configurations as needed
    ]

    performance_results = []

    for config in configurations:
        start_time = time.time()
        training_id = send_training_request(config)
        status = check_training_status(training_id)
        end_time = time.time()

        duration = end_time - start_time
        performance_results.append({
            'config': config,
            'duration': duration,
            'status': status
        })
        print(f"Config: {config}, Duration: {duration} seconds, Status: {status}")


    with open('performance_results.json', 'w') as file:
        json.dump(performance_results, file, indent=4)

if __name__ == "__main__":
    performance_test()
