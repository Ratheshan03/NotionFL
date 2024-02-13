from flask import Flask, jsonify, request, send_file, send_from_directory, url_for
from flask_cors import CORS
import subprocess
import threading
import os
import yaml
import json

app = Flask(__name__)
CORS(app)

# Assuming your logs and models are stored in the following directory structure:
DATA_COLLECTOR_DIR = os.path.join('output', 'data_collector')
TRAINING_LOGS_DIR = os.path.join(DATA_COLLECTOR_DIR, 'client', 'training')
EVALUATION_DIR = os.path.join(DATA_COLLECTOR_DIR, 'client', 'evaluation')
GLOBAL_EVALUATION_DIR = os.path.join(DATA_COLLECTOR_DIR, 'global', 'evaluation')
GLOBAL_MODEL_DIR = os.path.join(DATA_COLLECTOR_DIR, 'global', 'models')
FEDXAI_DIR = os.path.join(DATA_COLLECTOR_DIR, 'FedXAIEvaluation')
AGGREGATION_EXPLANATION_DIR = os.path.join(DATA_COLLECTOR_DIR, 'FedXAIEvaluation', 'aggregation_explanation')
COMPARISON_PLOTS_DIR = os.path.join(DATA_COLLECTOR_DIR, 'FedXAIEvaluation', 'comparison')
GLOBAL_SHAP_DIR = os.path.join(DATA_COLLECTOR_DIR, 'FedXAIEvaluation', 'global')





@app.route('/get_global_evaluation/<round_number>', methods=['GET'])
def get_global_evaluation(round_number):
    json_filename = f'global_model_round_{round_number}.json'
    shap_image_filename = f'shap_explanation_round_{round_number}.png'

    # Construct the file paths
    json_file_path = os.path.join(GLOBAL_EVALUATION_DIR, json_filename)
    shap_image_path = os.path.join(GLOBAL_SHAP_DIR, shap_image_filename)

    # Check if both files exist
    if not os.path.exists(json_file_path) or not os.path.exists(shap_image_path):
        return jsonify({'error': 'One or more files not found'}), 404

    # Create full URLs for both files
    json_url = request.host_url.rstrip('/') + url_for('static', filename=json_file_path)
    shap_image_url = request.host_url.rstrip('/') + url_for('static', filename=shap_image_path)

    # Return URLs in JSON response
    return jsonify({
        'json_url': json_url,
        'shap_image_url': shap_image_url
    }), 200

@app.route('/comparison_plot/<round_number>', methods=['GET'])
def get_comparison_plot(round_number):
    """
    Endpoint to send the client and global model comparison plot for a given round.
    """
    filename = f'comparison_plot_round_{round_number}.png'
    file_path = os.path.join(COMPARISON_PLOTS_DIR, filename)

    # Check if file exists
    if not os.path.exists(file_path):
        return jsonify({'error': 'Comparison plot not found'}), 404

    # Create full URL for the file
    file_url = request.host_url.rstrip('/') + url_for('static', filename=file_path)

    # Return URL in JSON response
    return jsonify({
        'comparison_plot_url': file_url
    }), 200


@app.route('/training_logs/<int:client_id>', methods=['GET'])
def get_training_logs(client_id):
    try:
        log_files = [f for f in os.listdir(TRAINING_LOGS_DIR) if f.startswith(f'client_{client_id}_')]
        logs = [json.load(open(os.path.join(TRAINING_LOGS_DIR, file), 'r')) for file in log_files]
        return jsonify(logs)
    except FileNotFoundError:
        return jsonify({"error": "Training logs not found"}), 404

@app.route('/evaluation_logs/<int:client_id>/<int:round_number>', methods=['GET'])
def get_evaluation_logs(client_id, round_number):
    try:
        filename = f'client_{client_id}_evaluation_logs_round_{round_number}.json'
        file_path = os.path.join(EVALUATION_DIR, filename)
        return send_file(file_path)
    except FileNotFoundError:
        return jsonify({"error": "Evaluation logs not found"}), 404


@app.route('/get_final_global_model', methods=['GET'])
def get_final_global_model():
    try:
        files = [f for f in os.listdir(GLOBAL_MODEL_DIR) if f.endswith('.pt')]
        files.sort(reverse=True)  # Sort to get the latest file
        latest_model = files[0]
        return send_file(os.path.join(GLOBAL_MODEL_DIR, latest_model))
    except (FileNotFoundError, IndexError):
        return jsonify({"error": "Final global model not found"}), 404
    

@app.route('/get_explanation/<client_id>/<round_number>', methods=['GET'])
def get_explanation(client_id, round_number):
    try:
        image_file = f'impact_visualization_client_{client_id}_round_{round_number}.png'
        text_file = f'interpretation_client_{client_id}_round_{round_number}.txt'
        image_path = os.path.join(FEDXAI_DIR,'privacy_explanations', 'visualizations', image_file)
        text_path = os.path.join(FEDXAI_DIR, client_id, text_file)
        image_url = request.host_url.rstrip('/') + image_path
        text_url = request.host_url.rstrip('/') + text_path
        return jsonify({'image_url': image_url, 'text_url': text_url})
    except FileNotFoundError:
        return jsonify({"error": "Explanation files not found"}), 404
    

@app.route('/start_training', methods=['POST'])
def start_training():
    data = request.form.to_dict()
    # Convert form data to appropriate types and save to config.yml
    with open('config.yml', 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
    # Start training in a new thread
    threading.Thread(target=lambda: subprocess.run(["python", "main.py"])).start()
    return jsonify({"status": "Training started"}), 200

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
