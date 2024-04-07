import threading
import uuid
from flask import Blueprint, jsonify, request
from Database.controllers.results import get_global_data,get_client_training_data, get_privacy_and_security_data

server_bp = Blueprint('server_bp', __name__)
server_sessions = {} 

@server_bp.route('/get_client_data/<training_id>/<client_id>/<round>', methods=['GET'])
def get_client_data_route(training_id, client_id, round):
    try:
        training_data = get_client_training_data(training_id, client_id, round)
        return jsonify(training_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
    
@server_bp.route('/privacy_data/<training_id>', methods=['GET'])
def get_privacy_data_route(training_id):
    try:
        training_data = get_privacy_and_security_data(training_id)
        return jsonify(training_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400  
    
    
@server_bp.route('/global_model_data/<training_id>', methods=['GET'])
def get_global_data_route(training_id):
    try:
        training_data = get_global_data(training_id)
        return jsonify(training_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400 