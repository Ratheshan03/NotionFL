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
        "eval_plot": f"FedXAIEvaluation/clients/client_{client_id}/evaluation/shap_plot_{round}.png",
        "eval_text": f"FedXAIEvaluation/clients/client_{client_id}/evaluation/evaluation_{round}.txt",
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
        "global_model_cmatrix": "FedXAIEvaluation/globals/final_confusion_matrix.png",
        "global_model_eval": "FedXAIEvaluation/globals/final_evaluation.txt",
        "global_model_shap_plot": "FedXAIEvaluation/globals/final_shap_plot.png",
        "final_global_model": "global/models/final_global_model.pt"
    }

    global_model_data = {}
    for data_type, cloud_path in data_types.items():
        file_content = file_handler.retrieve_file(training_id, cloud_path)
        if file_content is not None:
            if cloud_path.endswith(".png") or cloud_path.endswith(".pt"):
                global_model_data[data_type] = base64.b64encode(file_content).decode('utf-8')
            else:
                # Directly store other file types like .txt
                global_model_data[data_type] = file_content
        else:
            global_model_data[data_type] = "Not available"

    return global_model_data



#  Client Results Route Helper Functions


def get_client_specific_training_data(training_id, client_id, round):
    file_handler = FileHandler()
    
    # paths in the cloud storage
    data_types = {
        "eval_logs": f"client/evaluation/client_{client_id}_evaluation_logs_round_{round}.json",
        "training_logs": f"client/training/client_{client_id}_training_logs.json",
        "eval_plot": f"FedXAIEvaluation/clients/client_{client_id}/evaluation/shap_plot_{round}.png",
        "eval_text": f"FedXAIEvaluation/clients/client_{client_id}/evaluation/evaluation_{round}.txt",
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
    

def get_client_specific_privacy_data(training_id, client_id, round):
    file_handler = FileHandler()
    
    data_types = {
        "privacy_explanation": f"FedXAIEvaluation/privacy_explanations/interpretation_client_{client_id}_round_{round}.txt",
        "privacy_explanation_plot": f"FedXAIEvaluation/privacy_explanations/visualizations/impact_visualization_client_{client_id}_round_{round}.png",
        "dp_json": f"privacy/differential_privacy_round_{round}.json",
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



def get_client_specific_aggregation_data(training_id, client_id, round):
    file_handler = FileHandler()
    
    data_types = {
        "aggregation_plot": f"FedXAIEvaluation/aggregation_explanation/aggregation_plot_round_{round}.png",
        "aggregation_json": f"aggregation/secure_aggregation_round_{round}.json",
    }
    
    aggregation_data = {}
    for data_type, cloud_path in data_types.items():
        file_content = file_handler.retrieve_file(training_id, cloud_path)
        if file_content is not None:
            if cloud_path.endswith(".png"):
                aggregation_data[data_type] = base64.b64encode(file_content).decode('utf-8')
            else:

                aggregation_data[data_type] = file_content
        else:
            aggregation_data[data_type] = "Not available"

    return aggregation_data




def get_client_specific_evaluation_data(training_id, client_id, round):
    file_handler = FileHandler()
    
    data_types = {
        "eval_logs": f"client/evaluation/client_{client_id}_evaluation_logs_round_{round}.json",
        "eval_plot": f"FedXAIEvaluation/clients/client_{client_id}/evaluation/shap_plot_{round}.png",
        "eval_text": f"FedXAIEvaluation/clients/client_{client_id}/evaluation/evaluation_{round}.txt",
    }
    
    evaluation_data = {}
    for data_type, cloud_path in data_types.items():
        file_content = file_handler.retrieve_file(training_id, cloud_path)
        if file_content is not None:
            if cloud_path.endswith(".png"):
                evaluation_data[data_type] = base64.b64encode(file_content).decode('utf-8')
            else:

                evaluation_data[data_type] = file_content
        else:
            evaluation_data[data_type] = "Not available"

    return evaluation_data



def get_client_specific_contribution_data(training_id, client_id):
    file_handler = FileHandler()
    
    data_types = {
        "contribution_distribution": f"FedXAIEvaluation/contribution_explanations/contribution_distribution.png",
        "contribution_vs_incentives": f"FedXAIEvaluation/contribution_explanations/contributions_vs_incentives.png",
        "contribution_allocation_vs_shap_values": f"FedXAIEvaluation/contribution_explanations/incentive_allocation_vs_shapley_values.png",
        "incentive_explanation": f"FedXAIEvaluation/contribution_explanations/incentive_explanation.txt",
    }
    
    contribution_data = {}
    for data_type, cloud_path in data_types.items():
        file_content = file_handler.retrieve_file(training_id, cloud_path)
        if file_content is not None:
            if cloud_path.endswith(".png"):
                contribution_data[data_type] = base64.b64encode(file_content).decode('utf-8')
            else:

                contribution_data[data_type] = file_content
        else:
            contribution_data[data_type] = "Not available"

    return contribution_data