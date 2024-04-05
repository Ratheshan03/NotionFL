# Training related methods
training_sessions = {}
training_status = {} 

def run_training_process(training_id, config_data):
    log_dir = 'training_logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize training session
    training_sessions[training_id] = {
        'status': 'Started',
        'start_time': datetime.utcnow().isoformat() + 'Z',
        'logs': '',
        'end_time': None
    }

    # Save received config data to config.yml
    with open('config.yml', 'w') as file:
        yaml.dump(config_data, file, default_flow_style=False)
    
    log_file_path = os.path.join(log_dir, f"{training_id}_logs.txt")
    with open(log_file_path, 'w') as log_file:
        process = Popen(["python", "main.py"], stdout=log_file, stderr=log_file)
    
    process.wait()
    
    # Update the status with end time and status
    training_sessions[training_id]['end_time'] = datetime.utcnow().isoformat() + 'Z'
    training_sessions[training_id]['status'] = 'Completed' if process.returncode == 0 else 'Failed'
    
    # Read the log file and store its contents in the session
    with open(log_file_path, 'r') as log_file:
        training_sessions[training_id]['logs'] = log_file.read()
                

@app.route('/start_training', methods=['POST'])
def start_training():
    data = request.json  # Get data from POST request
    # training_id = "training_" + str(len(training_status) + 1)  # Generate simple training ID

    # Convert string values to correct types
    try:
        data['batch_size'] = int(data.get('batch_size', 64))
        data['epochs'] = int(data.get('epochs', 5))
        data['fl_rounds'] = int(data.get('fl_rounds', 1))
        data['eval_every_n_rounds'] = int(data.get('eval_every_n_rounds', 1))
        data['num_clients'] = int(data.get('num_clients', 4))
        data['learning_rate'] = float(data.get('learning_rate', 0.01))
        data['noise_multiplier'] = float(data.get('noise_multiplier', 0.1))
        data['clip_threshold'] = float(data.get('clip_threshold', 1.0))
        dataset = data.get('dataset')
        model = data.get('model')
        if dataset not in ['MNIST', 'CIFAR10']:  # Extend with other datasets as needed
            raise ValueError(f"Unsupported dataset: {dataset}")
        if model not in ['MNISTModel', 'CIFAR10Model']:  # Extend with other models as needed
            raise ValueError(f"Unsupported model: {model}")
    except (ValueError, KeyError) as e:
        return jsonify({"error": f"Invalid data format: {e}"}), 400

    # Initialize training status and start training in a new thread
    training_id = str(uuid.uuid4())
    training_sessions[training_id] = {'status': 'Started', 'logs': ''}
    thread = threading.Thread(target=run_training_process, args=(training_id, data))
    thread.start()
    
    return jsonify({"status": "Training started", "training_id": training_id}), 202


@app.route('/training_status', methods=['GET'])
def get_all_training_status():
    # Return the status of all training sessions
    return jsonify(training_sessions)


@app.route('/training_status/<training_id>', methods=['GET'])
def check_training_status(training_id):
    status = training_status.get(training_id, "No such training process found")
    log_file = os.path.join('training_logs', f"{training_id}_logs.txt")
    
    # Read logs from the file
    try:
        with open(log_file, 'r') as file:
            logs = file.read()
    except FileNotFoundError:
        logs = "Logs not found"
    
    print({"training_id": training_id, "status": status})

    return jsonify({"training_id": training_id, "status": status, "logs": logs})

