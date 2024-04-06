import json
import threading
import time
import os
import yaml
from datetime import datetime, timezone
from subprocess import Popen
from flask import current_app as app
from ..schemas.user_schema import User, TrainingSession
from ..schemas.training_schema import TrainingModel

def get_training_configuration():
    config_path = app.config.get('TRAINING_CONFIG_PATH', './scripts/training_configuration.json')
    try:
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
            return config
    except FileNotFoundError:
        app.logger.error('Training configuration file not found.')
        return {}
    except json.JSONDecodeError:
        app.logger.error('Error decoding JSON from the training configuration file.')
        return {}

def update_training_logs(training_id, log_file_path, update_interval=30):
    try:
        while True:
            with open(log_file_path, 'r') as log_file:
                logs = log_file.read()

            training_session = TrainingModel.objects(training_id=training_id).first()
            if training_session:
                training_session.update(set__logs=logs)

            if training_session and training_session.status in ['Completed', 'Failed']:
                break

            time.sleep(update_interval)
    except Exception as e:
        print(f"An error occurred while updating training logs: {e}")


def start_fl_training(training_id, training_data, user_id):
    try:
        # Initiate training process and save config data
        initiating_training(training_id, training_data, user_id)
        
        # Assign client IDs to registered clients for this training session
        assign_client_ids_to_training(training_id, training_data['num_clients'])
        
        log_dir = 'training_logs'
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, f"{training_id}_logs.txt")

        # Start the training process
        with open(log_file_path, 'w') as log_file:
            process = Popen(["python", "main.py", training_id], stdout=log_file, stderr=log_file)

        # Start a thread to update logs in real-time
        log_thread = threading.Thread(target=update_training_logs, args=(training_id, log_file_path))
        log_thread.start()

        process.wait()
        update_training_status(training_id, 'Completed' if process.returncode == 0 else 'Failed')
    
    except Exception as e:
        print(f"Error during training: {e}")
        
        
def initiating_training(training_id, training_data, user_id):
    user = User.objects(id=user_id).first()
    if not user:
        raise ValueError(f"User with ID {user_id} not found")

    training_object = TrainingModel(
        training_id=training_id,
        initiator={
            'user_id': str(user.id),
            'username': user.username
        },
        config=training_data,
        status='Started',
        start_time=datetime.now(timezone.utc),
    )

    training_object.save()

    return training_object


def get_user_training_sessions(userId):
    try:
        training_sessions = TrainingModel.objects(initiator__user_id=userId)
        return [session.to_json() for session in training_sessions]
    except Exception as e:
        raise e


def update_training_status(training_id, new_status):
    try:
        training_session = TrainingModel.objects(training_id=training_id).first()
        if training_session:
            training_session.update(set__status=new_status)
            print(f"Training session {training_id} status updated to {new_status}")
        else:
            print(f"No training session found with ID {training_id}")
    except Exception as e:
        print(f"An error occurred while updating training status: {e}")


def assign_client_ids_to_training(training_id, num_clients):
    registered_clients = User.objects(role='client', training_sessions__0__exists=False)
    registered_clients = sorted(registered_clients, key=lambda x: x.id.generation_time)[:num_clients]

    for i, client in enumerate(registered_clients):
        client_number = i 
        new_session = TrainingSession(training_id=training_id, client_number=client_number)
        client.update(push__training_sessions=new_session)
    