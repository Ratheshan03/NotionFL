import base64
from utils.file_handler import FileHandler
import os
import re
import yaml
from datetime import datetime, timezone
from subprocess import Popen
from flask import current_app as app
from ..schemas.user_schema import User
from ..schemas.training_schema import TrainingModel
from firebase_admin import storage

def get_client_training_data(training_id, client_id, round):
    file_handler = FileHandler()
    
    # paths in the cloud storage
    data_types = {
        "eval_logs": f"client/evaluation/client_{client_id}_evaluation_logs_round_{round}.json",
        "training_logs": f"client/training/client_{client_id}_training_logs.json",
        "eval_plot": f"FedXAIEvaluation/clients/client_{client_id}/evaluation/shap_plot.png",
        "eval_text": f"FedXAIEvaluation/clients/client_{client_id}/evaluation/evaluation.txt",
    }

    training_data = {}
    for data_type, cloud_path in data_types.items():
        file_content = file_handler.retrieve_file(training_id, cloud_path)
        if file_content is not None:
            if cloud_path.endswith(".png"):
                # If it's an image, convert to base64 for JSON serialization
                training_data[data_type] = base64.b64encode(file_content).decode()
            else:
                training_data[data_type] = file_content
        else:
            training_data[data_type] = "Not available"

    return training_data


def get_privacy_and_security_data(training_id):
    file_handler = FileHandler()
    
    data_types = {
        "privacy_explanation": f"FedXAIEvaluation/privacy_explanations/interpretation_client_0_round_0.txt",
        "privacy_explanation_plot": f"FedXAIEvaluation/privacy_explanations/visualizations/impact_visualization_client_0_round_0.png",
        "dp_json": f"privacy/differential_privacy_round_0.json",
        "aggregation": f"aggregation/secure_aggregation_round_0.json"
    }
    
    privacy_data = {}
    for data_type, cloud_path in data_types.items():
        file_content = file_handler.retrieve_file(training_id, cloud_path)
        if file_content is not None:
            if cloud_path.endswith(".png"):
                privacy_data[data_type] = base64.b64encode(file_content).decode('utf-8')
            else:

                privacy_data[data_type] = file_content
        else:
            privacy_data[data_type] = "Not available"

    return privacy_data


def get_global_data(training_id):
    file_handler = FileHandler()
    
    data_types = {
        "global_model_cmatrix": f"FedXAIEvaluation/globals/final_confusion_matrix.png",
        "global_model_eval": f"FedXAIEvaluation/globals/final_evaluation.txt",
        "global_model_shap_plot": f"FedXAIEvaluation/globals/final_shap_plot.png",
    }
    
    # Initially, you don't have the final global model path,
    global_model_prefix = "global/models/global_model_round_"
    global_model_suffix = ".pt"
    
    # Get a list of all 
    global_model_files = file_handler.list_files(f"{training_id}/global/models/")
    
    # Filter out the files
    round_numbers = [
        int(re.search(rf"{global_model_prefix}(\d+){global_model_suffix}", file_path).group(1))
        for file_path in global_model_files
        if re.match(rf"{global_model_prefix}\d+{global_model_suffix}", file_path)
    ]
    
    if round_numbers:
        highest_round_num = max(round_numbers)
        final_global_model_path = f"{global_model_prefix}{highest_round_num}{global_model_suffix}"
    else:
        final_global_model_path = "Not available"
    
    
    if final_global_model_path != "Not available":
        data_types["final_global_model"] = final_global_model_path
    
    privacy_data = {}
    for data_type, cloud_path in data_types.items():
        file_content = file_handler.retrieve_file(training_id, cloud_path)
        if file_content is not None:
            if cloud_path.endswith(".png"):
                privacy_data[data_type] = base64.b64encode(file_content).decode('utf-8')
            else:
                privacy_data[data_type] = file_content
        else:
            privacy_data[data_type] = "Not available"

    return privacy_data


