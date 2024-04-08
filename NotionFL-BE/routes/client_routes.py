import threading
import uuid
from flask import Blueprint, jsonify, request
from Database.controllers.results import  get_client_specific_contribution_data,get_client_specific_evaluation_data, get_client_specific_aggregation_data, get_client_specific_privacy_data, get_client_specific_training_data
from Database.controllers.training import get_client_training_sessions

client_bp = Blueprint('client_bp', __name__)
client_sessions = {} 

@client_bp.route('/get_training_sessions/<user_id>', methods=['GET'])
def get_client_sessions(user_id):
    try:
        training_sessions = get_client_training_sessions(user_id)
        return jsonify(training_sessions), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@client_bp.route('/get_training_data/<training_id>/<client_id>/<round>', methods=['GET'])
def get_client_data_route(training_id, client_id, round):
    try:
        training_data = get_client_specific_training_data(training_id, client_id, round)
        return jsonify(training_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
@client_bp.route('/get_privacy_data/<training_id>/<client_id>/<round>', methods=['GET'])
def get_client_privacyData_route(training_id, client_id, round):
    try:
        training_data = get_client_specific_privacy_data(training_id, client_id, round)
        return jsonify(training_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@client_bp.route('/get_aggregation_data/<training_id>/<client_id>/<round>', methods=['GET'])
def get_client_aggregationData_route(training_id, client_id, round):
    try:
        training_data = get_client_specific_aggregation_data(training_id, client_id, round)
        return jsonify(training_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@client_bp.route('/get_model_evaluation_data/<training_id>/<client_id>/<round>', methods=['GET'])
def get_client_evalautionData_route(training_id, client_id, round):
    try:
        training_data = get_client_specific_evaluation_data(training_id, client_id, round)
        return jsonify(training_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
  
@client_bp.route('/get_incentives_data/<training_id>/<client_id>', methods=['GET'])
def get_client_contribution_data_route(training_id, client_id):
    try:
        training_data = get_client_specific_contribution_data(training_id, client_id)
        return jsonify(training_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400 
  
  

    
  