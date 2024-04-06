import threading
import uuid
from flask import Blueprint, jsonify, request
from Database.controllers.training import get_training_configuration
from Database.controllers.training import start_fl_training, get_user_training_sessions

training_bp = Blueprint('training_bp', __name__)
training_sessions = {} 

@training_bp.route('/get_training_config', methods=['GET'])
def get_training_config():
    config = get_training_configuration()
    return jsonify(config), 200


@training_bp.route('/start_training', methods=['POST'])
def start_training():
    data = request.json

    try:
        training_data = {
            'batch_size': int(data.get('batch_size', 64)),
            'epochs': int(data.get('epochs', 5)),
            'fl_rounds': int(data.get('fl_rounds', 1)),
            'eval_every_n_rounds': int(data.get('eval_every_n_rounds', 1)),
            'num_clients': int(data.get('num_clients', 4)),
            'learning_rate': float(data.get('learning_rate', 0.01)),
            'noise_multiplier': float(data.get('noise_multiplier', 0.1)),
            'clip_threshold': float(data.get('clip_threshold', 1.0)),
            'device': data.get('device', 'cpu'),
            'dataset': data.get('dataset'),
            'model': data.get('model')
        }
        user_id = data.get('user_id')

        if training_data['dataset'] not in ['MNIST', 'CIFAR10']: 
            raise ValueError(f"Unsupported dataset: {data['dataset']}")
        if training_data['model'] not in ['MNISTModel', 'CIFAR10Model']: 
            raise ValueError(f"Unsupported model: {data['model']}")
        
        # start training in a new thread
        training_id = str(uuid.uuid4())
        print('-------->',training_data)
        training_sessions[training_id] = {'status': 'Started', 'logs': ''}
        thread = threading.Thread(target=start_fl_training, args=(training_id, training_data, user_id))
        thread.start()

        return jsonify({"status": "Training started", "training_id": training_id}), 202
    
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid data format: {e}"}), 400


@training_bp.route('/training_sessions/<userId>', methods=['GET'])
def user_training_sessions(userId):
    try:
        training_sessions = get_user_training_sessions(userId)
        return jsonify(training_sessions), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    


