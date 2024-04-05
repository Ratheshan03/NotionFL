#Assuming your logs and models are stored in the following directory structure:
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
        'jso': json_url,
        'shap_image_url': shap_image_url
    }), 200
    
    
@app.route('/get_global_evaluation/<round_number>', methods=['GET'])
def get_client_evaluation_logs(round):
    try:
        filename = f'global_model_round_{round}.json'
        file_path = os.path.join(DATA_COLLECTOR_DIR, 'global', 'evaluation', filename)
        with open(file_path, "r") as file:
            return file.read()
    except FileNotFoundError:
        return jsonify({"error": "Evaluation logs not found"}), 404
    

@app.route('/get_client_shap_plot/<client_number>/<round_number>', methods=['GET'])
def get_client_shap_plot(client_id, round):
    try:
        filename = f'shap_explanation_round_{round}.png'
        file_path = os.path.join(FEDXAI_DIR, f'client_{client_id}', filename)
        return image_to_base64(file_path)
    except FileNotFoundError:
        return jsonify({"error": "Evaluation logs not found"}), 404
    

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
        log_files = [f for f in os.listdir(TRAINING_LOGS_DIR) if f.startswith(f'client_{client_id}')]
        logs = [json.load(open(os.path.join(TRAINING_LOGS_DIR, file), 'r')) for file in log_files]
        return jsonify(logs)
    except FileNotFoundError:
        return jsonify({"error": "Training logs not found"}), 404


@app.route('/client_incentives/<int:round_num>', methods=['GET'])
def get_incentive_data(round_num):
    data = {
            'shapleyValues': get_incentive_shapley_values(round_num),
            'contributionPlot': get_incentive_contribution_plot(round_num), 
            'incentives': get_incentives(round_num),
            'incentivesPlot': get_shapley_incentive_plot(round_num),
            'incentivesExplanation': get_text_explanation(round_num),
    }
    return jsonify(data)


def get_incentive_shapley_values(round_num):
    try:
        filename = f'client_shapley_values_round_{round_num}.json'
        file_path = os.path.join(DATA_COLLECTOR_DIR, 'client', 'contribution', filename)
        with open(file_path, "r") as file:
            return file.read()
    except FileNotFoundError:
        return jsonify({"error": "Shapley values not found"}), 404
    
def get_incentive_contribution_plot(round_num):
    try:
        filename = f'client_contribution_plot_round_{round_num}.png'
        file_path = os.path.join(DATA_COLLECTOR_DIR, 'client', 'contribution', filename)
        return image_to_base64(file_path)
    except FileNotFoundError:
        return jsonify({"error": "Contribution plot not found"}), 404
    
    
def get_incentives(round_num):
    try:
        filename = f'client_incentives_round_{round_num}.json'
        file_path = os.path.join(DATA_COLLECTOR_DIR, 'client', 'contribution', filename)
        with open(file_path, "r") as file:
            return file.read()
    except FileNotFoundError:
        return jsonify({"error": "Incentives not found"}), 404
    
    
def get_shapley_incentive_plot(round_num):
    try:
        filename = f'incentive_plot_round_{round_num}.png'
        file_path = os.path.join(DATA_COLLECTOR_DIR, 'client', 'contribution', filename)
        return image_to_base64(file_path)
    except FileNotFoundError:
        return jsonify({"error": "Incentive plot not found"}), 404
    

def get_text_explanation(round_num):
    try:
        filename = f'incentive_explanation.txt'
        file_path = os.path.join(DATA_COLLECTOR_DIR, 'client', 'contribution', filename)
        with open(file_path, "r") as file:
            return file.read()
    except FileNotFoundError:
        return jsonify({"error": "Incentive explanation not found"}), 404




@app.route('/client_full_data/<int:client_id>', methods=['GET'])
def get_client_full_data(client_id):
    data = {
            'trainingLogs': get_client_training_logs(client_id),
            'trainingStatus': check_client_training_status(client_id), 
    }
    return data


@app.route('/client_training_logs/<int:client_id>', methods=['GET'])
def get_client_training_logs(client_id):
    try:
        filename = f'client_{client_id}_training_logs.json'
        file_path = os.path.join(TRAINING_LOGS_DIR, filename)
        with open(file_path, "r") as file:
            return file.read()
    except FileNotFoundError:
        return jsonify({"error": "Evaluation logs not found"}), 404
    
@app.route('/training_status/<int:client_id>', methods=['GET'])
def check_client_training_status(client_id):
    # Assuming training_status is a dictionary mapping client_ids to their training status
    status = training_status.get(client_id, "No such training process found")

    return status




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
    
    
    
    
@app.route('/global_model_data/<int:round_number>', methods=['GET'])
def get_global_model_data(round_number):
    setGlobalData = {
          'evaluationResults': getEvaluations(round_number),
          'shapValuesPlot': getShapPlot(round_number),
          'modelComparisonPlot': getComparisonPlot(round_number),
          'globalModelUrl': getGlobalModel(round_number),
    }
    
    return jsonify(setGlobalData)


def getEvaluations(round_number):
    try:
        filename = f'global_model_round_{round_number}.json'
        file_path = os.path.join(GLOBAL_EVALUATION_DIR, filename)
        with open(file_path, "r") as file:
            return file.read()
    except FileNotFoundError:
        return jsonify({"error": "Evaluation logs not found"}), 404
    
    
def getShapPlot(round_number):
    try:
        filename = f'shap_explanation_round_{round_number}.png'
        file_path = os.path.join(GLOBAL_SHAP_DIR, filename)
        return image_to_base64(file_path)
    except FileNotFoundError:
        return jsonify({"error": "Shap explanation plot not found"}), 404
    
    
def getComparisonPlot(round_number):
    try:
        filename = f'comparison_plot_round_{round_number}.png'
        file_path = os.path.join(COMPARISON_PLOTS_DIR, filename)
        return image_to_base64(file_path)
    except FileNotFoundError:
        return jsonify({"error": "Comparison plot not found"}), 404
    

def getGlobalModel(round_number):
    try:
        filename = f'global_model_round_{round_number}.pt'
        file_path = os.path.join(GLOBAL_EVALUATION_DIR, filename)
        return file_path
    except (FileNotFoundError, IndexError):
        return jsonify({"error": "Final global model not found"}), 404
    

    
@app.route('/privacy_data/<int:client_id>/<int:round_number>', methods=['GET'])
def get_privacy_data(round_number, client_id):
    setPrivacyData = {
          'dpExplanation': getDPExplanations(round_number, client_id),
          'dpUsageImage': getDPUsageImage(round_number, client_id),
          'secureAggregationPlot': getSecureAggregationPlot(round_number),
    }
    
    return jsonify(setPrivacyData)
   
    
def getDPExplanations(round_number, client_id):
    try:
        filename = f'interpretation_client_{client_id}_round_{round_number}.txt'
        file_path = os.path.join(FEDXAI_DIR, 'privacy_explanations', filename)
        with open(file_path, "r") as file:
            return file.read()
    except FileNotFoundError:
        return jsonify({"error": "DP explanation text file not found"}), 404 
    
    
def getDPUsageImage(round_number, client_id):
    try:
        filename = f'impact_visualization_client_{client_id}_round_{round_number}.png'
        file_path = os.path.join(FEDXAI_DIR, 'privacy_explanations', 'visualizations', filename)
        return image_to_base64(file_path)
    except FileNotFoundError:
        return jsonify({"error": "DP explanation plot not found"}), 404
    

def getSecureAggregationPlot(round_number):
    try:
        filename = f'comparison_plot_round_{round_number}.png'
        file_path = os.path.join(FEDXAI_DIR, 'aggregation_explanation', filename)
        return image_to_base64(file_path)
    except FileNotFoundError:
        return jsonify({"error": "Secure aggregation plot not found"}), 404
    


# Helper function to get base64 encoded string for an image file
def image_to_base64(image_path):
    """
    Read an image file and convert it to a base64 encoded string.
    """
    with open(image_path, 'rb') as image_file:
        buffered = BytesIO(image_file.read())

    image_base64 = base64.b64encode(buffered.getvalue())
    encoded_string = image_base64.decode('utf-8')
    
    return encoded_string
    

@app.route('/client_data/<int:client_id>/<int:round_number>', methods=['GET'])
def get_client_data(client_id, round_number):
    # Here you would gather all the data for the client and round
    # For example:
    data = {
        'evaluationLogs': get_evaluation_logs(client_id, round_number),
        'modelEvaluation': get_model_evaluation(client_id, round_number),
        'globalModelComparison': get_global_model_comparison(round_number),
        'contributionShapleyValues': get_contribution_shapley_values(client_id, round_number),
        'contributionPlot': get_contribution_plot(client_id, round_number),
    }
    return jsonify(data)

def get_evaluation_logs(client_id, round_number):
    try:
        filename = f'client_{client_id}_evaluation_logs_round_{round_number}.json'
        file_path = os.path.join(EVALUATION_DIR, filename)
        with open(file_path, "r") as file:
            return file.read()
    except FileNotFoundError:
        return jsonify({"error": "Evaluation logs not found"}), 404
    
def get_model_evaluation(client_id, round_number):
    try:
        filename = f'shap_explanation_round_{round_number}.png'
        file_path = os.path.join(FEDXAI_DIR, f'client_{client_id}', filename)
        return image_to_base64(file_path)
    except FileNotFoundError:
        return jsonify({"error": "Shap explanation plot not found"}), 404
    
    
def get_global_model_comparison(round_number):
    try:
        filename = f'comparison_plot_round_{round_number}.png'
        file_path = os.path.join(FEDXAI_DIR, 'comparison', filename)
        return image_to_base64(file_path)
    except FileNotFoundError:
        return jsonify({"error": "Globalexplanation plot not found"}), 404
    
    
def get_contribution_shapley_values(client_id, round_number):
    try:
        filename = f'client_shapley_values_round_{round_number}.json'
        file_path = os.path.join(DATA_COLLECTOR_DIR, 'client', 'contribution', filename)
        with open(file_path, "r") as file:
            return file.read()
    except FileNotFoundError:
        return jsonify({"error": "Shap values not found"}), 404


def get_contribution_plot(client_id, round_number):
    try:
        filename = f'client_contribution_plot_round_{round_number}.png'
        file_path = os.path.join(DATA_COLLECTOR_DIR, 'client', 'contribution', filename)
        return image_to_base64(file_path)
    except FileNotFoundError:
        return jsonify({"error": "Shap explanation plot not found"}), 404
    
    