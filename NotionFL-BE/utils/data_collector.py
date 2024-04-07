from io import BytesIO
import io
import shutil
import bson
import matplotlib
matplotlib.use('Agg')
import json
import os
from matplotlib import pyplot as plt
import numpy as np
import shap
import logging
from Database.schemas.training_schema import AggregationPlot,TrainingModel, EvaluationPlot, ModelComparison, GlobalModelData, GlobalModelEvaluation, GlobalModel, ClientData, IncentivesData, IncentiveExplanation, FileField, ClientEvaluation, ClientModel, PrivacyExplanation, GlobalModel,GlobalModelData, ContributionMetric
import gzip
import torch
import pickle
import gridfs
from pymongo import MongoClient

class DataCollector:
    def __init__(self, output_dir, training_id):
        self.output_dir = output_dir
        self.training_id = training_id
        
        os.makedirs(self.output_dir, exist_ok=True)
        # Initialize storage structures for various metrics and logs
        self.client_training_logs = {}
        self.client_models = {}
        self.client_updates = {}
        self.global_model_metrics = {}
        self.contribution_eval_metrics = {}
        self.secure_aggregation_logs = {}
        self.differential_privacy_logs = {}

    def collect_client_training_logs(self, client_id, training_logs):
        training_session = TrainingModel.objects(training_id=self.training_id).first()
        if training_session:
            client_id_str = str(client_id)  # Convert client_id to string
            if 'client_training_logs' not in training_session:
                training_session.client_training_logs = {}
            training_session.client_training_logs[client_id_str] = json.dumps(training_logs)
            training_session.save()
            logging.info(f"Training logs saved successfully for client_{client_id_str} in DB.")
        else:
            logging.error(f"No training session found with ID {self.training_id}")

        
        
    def collect_client_evaluation_logs(self, client_id, evaluation_logs, round):
        training_session = TrainingModel.objects(training_id=self.training_id).first()
        if training_session:
            client_id_str = str(client_id)  # Convert client_id to string
            round_key = f"round_{str(round)}"  # Convert round to string

            if client_id_str not in training_session.client_evaluation_logs:
                training_session.client_evaluation_logs[client_id_str] = {}
            
            training_session.client_evaluation_logs[client_id_str][round_key] = json.dumps(evaluation_logs)
            
            # Save the updated document to the database
            training_session.save()
            logging.info(f"Evaluation logs saved successfully for client_{client_id} for round {round} in DB.")
        else:
            logging.error(f"No training session found with ID {self.training_id}")
      

    def collect_client_model(self, client_id, model_state, round_num):
        training_session = TrainingModel.objects(training_id=self.training_id).first()
        if training_session:
            # Create a binary representation of the model state dict
            model_state_binary = bson.Binary(pickle.dumps(model_state))

            # Find or initialize the client data within the training session
            client_data = next((c for c in training_session.clients if c.client_id == str(client_id)), None)
            if not client_data:
                client_data = ClientData(client_id=str(client_id))
                training_session.clients.append(client_data)

            # Append the model state to the client's models list for the given round
            client_data.models.append({'round': round_num, 'state_dict': model_state_binary})

            # Save the updated document to the database
            training_session.save()
            logging.info(f"Model for client_{client_id} round {round_num} saved in DB.")
        else:
            logging.error(f"No training session found with ID {self.training_id}")

        

    def collect_client_updates(self, client_id, model_update):
        training_session = TrainingModel.objects(training_id=self.training_id).first()
        if training_session:
            # Serialize the model update to binary
            model_update_binary = bson.Binary(pickle.dumps(model_update))

            # Find or initialize the client data within the training session
            client_data = next((c for c in training_session.clients if c.client_id == str(client_id)), None)
            if not client_data:
                client_data = ClientData(client_id=str(client_id))
                training_session.clients.append(client_data)

            # Append the model update binary data to the client's updates list
            client_data.updates.append(model_update_binary)

            # Save the updated document to the database
            training_session.save()
            logging.info(f"Update for client_{client_id} saved in DB.")
        else:
            logging.error(f"No training session found with ID {self.training_id}")
            
            
    def save_client_model_evaluation(self, client_id, evaluation_text, shap_plot_buf):
        training_session = TrainingModel.objects(training_id=self.training_id).first()
        if training_session:
            # Create a ClientEvaluation instance
            client_evaluation = ClientEvaluation(
                evaluation_text=evaluation_text,
                shap_plot=shap_plot_buf.getvalue()
            )
            
            # Find or initialize the client data within the training session
            client_data = next((c for c in training_session.clients if c.client_id == str(client_id)), None)
            if not client_data:
                client_data = ClientData(client_id=str(client_id))
                training_session.clients.append(client_data)
            
            # Append the new client evaluation to the client's evaluations list
            client_data.evaluations.append(client_evaluation)

            # Save the updated document to the database
            training_session.save()
            logging.info(f"Model evaluation saved for client_{client_id} in DB.")
        else:
            logging.error(f"No training session found with ID {self.training_id}")


    def compress_model_state(model_state_dict):
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode='wb') as f:
            torch.save(model_state_dict, f)
        return 
    
    
    def collect_client_model(self, client_id, model_state_dict, round_num, suffix=''):
        # Define a directory to save model states
        model_state_dir = 'output/model_states'
        os.makedirs(model_state_dir, exist_ok=True)

        # Create a file path for the model state
        file_name = f"client_{client_id}_model_round_{round_num}{('_' + suffix) if suffix else ''}.pt"
        model_file_path = os.path.join(model_state_dir, file_name)

        # Save the model state
        torch.save(model_state_dict, model_file_path)

        training_session = TrainingModel.objects(training_id=self.training_id).first()
        if not training_session:
            logging.error(f"No training session found with ID {self.training_id}")
            return

        client_model = ClientModel(
            round_num=str(round_num),
            model_state=model_file_path,  # Now storing file path as a string
            suffix=suffix
        )

        client_data = next((c for c in training_session.clients if c.client_id == str(client_id)), None)
        if not client_data:
            client_data = ClientData(client_id=str(client_id))
            training_session.clients.append(client_data)
        
        client_data.models.append(client_model)
        training_session.save()
        logging.info(f"Client model saved for client_{client_id} in DB.")

        
    def decompress_model_state(compressed_model_state):
        buffer = io.BytesIO(compressed_model_state)
        with gzip.GzipFile(fileobj=buffer, mode='rb') as f:
            return torch.load(f)
        
        
    def save_privacy_explanations(self, explanation_text, plot_buffer, client_id, round_num):
        training_session = TrainingModel.objects(training_id=self.training_id).first()
        if not training_session:
            logging.error(f"No training session found with ID {self.training_id}")
            return

        # Save the privacy explanation text and visualization plot in the database
        privacy_explanation = PrivacyExplanation(
            text=explanation_text,
            visualization=plot_buffer.getvalue()
        )

        # Append the new privacy explanation to the training session
        training_session.privacy_explanations.append(privacy_explanation)

        # Save the updated document to the database
        training_session.save()
        logging.info(f"Privacy explanations saved for client_{client_id} round {round_num} in DB.")
        
        
    def collect_differential_privacy_logs(self, round_num, dp_metrics):
        training_session = TrainingModel.objects(training_id=self.training_id).first()
        if not training_session:
            logging.error(f"No training session found with ID {self.training_id}")
            return

        # Add the differential privacy metrics to the session
        dp_log_entry = {'round': round_num, 'metrics': dp_metrics}
        training_session.differential_privacy_logs.append(dp_log_entry)

        # Save the updated document to the database
        training_session.save()
        logging.info(f"Differential privacy logs saved for round {round_num} in DB.")
        
        
    def save_aggregation_explanation(self, plot_buffer, round_num):
        training_session = TrainingModel.objects(training_id=self.training_id).first()
        if not training_session:
            logging.error(f"No training session found with ID {self.training_id}")
            return

        plot_buffer.seek(0)
        plot_instance = AggregationPlot()
        plot_instance.plot.replace(plot_buffer, filename=f'aggregation_plot_round_{round_num}.png')
        plot_instance.save()

        # Reference the plot instance in the TrainingModel
        training_session.aggregation_plots.append(plot_instance)
        training_session.save()

        print(f"Aggregation plot for round {round_num} saved successfully in DB.")

        
        
    def collect_secure_aggregation_logs(self, round_num, aggregation_metrics, time_overheads):
        training_session = TrainingModel.objects(training_id=self.training_id).first()
        if not training_session:
            logging.error(f"No training session found with ID {self.training_id}")
            return

        # Prepare the aggregation log data
        secure_aggregation_data = {
            'aggregation_metrics': aggregation_metrics,
            'time_overheads': time_overheads
        }

        # Update or initialize the secure_aggregation_logs
        if not training_session.secure_aggregation_logs:
            training_session.secure_aggregation_logs = {}
        
        training_session.secure_aggregation_logs[f'round_{round_num}'] = secure_aggregation_data

        # Save the updated document to the database
        training_session.save()

        print(f"Secure aggregation log for round {round_num} saved successfully in DB.")
        

    def collect_global_model_metrics(self, round_num, model_metrics):
        training_session = TrainingModel.objects(training_id=self.training_id).first()
        if not training_session:
            logging.error(f"No training session found with ID {self.training_id}")
            return

        # Convert the metrics tuple into a dictionary
        metrics_dict = {
            'round': round_num,
            'test_loss': model_metrics[0],
            'accuracy': model_metrics[1],
            'precision': model_metrics[2],
            'recall': model_metrics[3],
            'f1': model_metrics[4],
            'conf_matrix': model_metrics[5].tolist() if isinstance(model_metrics[5], np.ndarray) else model_metrics[5]
        }

        # Update the global model metrics in the database
        training_session.update(push__global_model_metrics=metrics_dict)

        print(f"Global model metrics for round {round_num} saved successfully in DB.")



    def collect_global_model(self, global_model_state, round_num):
        training_session = TrainingModel.objects(training_id=self.training_id).first()
        if not training_session:
            logging.error(f"No training session found with ID {self.training_id}")
            return

        # Convert the model state to binary data
        model_binary_data = pickle.dumps(global_model_state)

        # Connect to GridFS in the MongoDB database
        db_uri = os.environ.get("MONGODB_URI") # assuming you are getting your MongoDB URI from environment variable
        client = MongoClient(db_uri)
        db_name = os.environ.get("DB_NAME") # getting the database name
        db = client[db_name]
        fs = gridfs.GridFS(db)

        # Save model state to GridFS and get the file ID
        file_id = fs.put(model_binary_data, filename=f'global_model_round_{round_num}.pkl')

        # Convert the file ID to a string (if it's not already)
        file_id_str = str(file_id)

        # Create an instance of GlobalModel with the string file_id
        global_model_instance = GlobalModel(round=str(round_num), model_state=file_id_str)

        # Update the global data in the database
        if training_session.global_data:
            training_session.global_data.models.append(global_model_instance)
        else:
            training_session.global_data = GlobalModelData(models=[global_model_instance])

        training_session.save()
        print(f"Global model state for round {round_num} saved successfully in DB.")

        
          
    def collect_contribution_metrics(self, contribution_metrics, shapley_plot):
        training_session = TrainingModel.objects(training_id=self.training_id).first()
        if not training_session:
            logging.error(f"No training session found with ID {self.training_id}")
            return

        # Convert all keys in the contribution_metrics dictionary to strings
        stringified_contribution_metrics = {str(k): v for k, v in contribution_metrics.items()}

        # Prepare a ContributionMetric object
        contribution_data = ContributionMetric(shapley_values=stringified_contribution_metrics)

        # Convert the plot to a binary format and save it
        plot_buffer = io.BytesIO()
        shapley_plot.savefig(plot_buffer, format='png')
        plot_buffer.seek(0)  # Go to the start of the buffer
        contribution_data.shapley_plot.put(plot_buffer, content_type='image/png')

        # Update the database
        training_session.update(set__contribution_metrics=contribution_data)
        training_session.save()
        print(f"Contribution metrics and plot for training session {self.training_id} saved successfully in DB.")

            
                    
    def save_incentives(self, incentives_json, incentive_plot_buf):
        training_session = TrainingModel.objects(training_id=self.training_id).first()
        if not training_session:
            logging.error(f"No training session found with ID {self.training_id}")
            return

        # Prepare an IncentivesData object
        incentives_data = IncentivesData(incentives_json=incentives_json)

        # Ensure buffer's pointer is at the beginning
        incentive_plot_buf.seek(0)

        # Save the plot in the IncentivesData object
        incentives_data.incentive_plot.put(incentive_plot_buf, content_type='image/png')

        # Update the database
        training_session.update(set__incentives_data=incentives_data)
        training_session.save()
        print(f"Incentives data for training session {self.training_id} saved successfully in DB.")

            
    def save_incentive_explanation(self, explanation_text, plot_buffers):
        training_session = TrainingModel.objects(training_id=self.training_id).first()
        if not training_session:
            logging.error(f"No training session found with ID {self.training_id}")
            return
        
        # Connect to MongoDB
        db_uri = os.environ.get("MONGODB_URI")  # Getting MongoDB URI from environment variable
        client = MongoClient(db_uri)
        db_name = os.environ.get("DB_NAME")  # Getting the database name
        db = client[db_name]
        fs = gridfs.GridFS(db)

        # Prepare an IncentiveExplanation object
        incentive_explanation = IncentiveExplanation(explanation_text=explanation_text, plots=[])

        # Save plots to GridFS and append file IDs to plots list
        for plot_buffer in plot_buffers:
            plot_buffer.seek(0)  # Ensure the buffer's pointer is at the start
            file_id = fs.put(plot_buffer, content_type='image/png')  # Save the plot to GridFS
            incentive_explanation.plots.append(file_id)  # Append the file ID to plots

        # Update the database
        training_session.update(set__incentive_explanation=incentive_explanation)
        training_session.save()
        print(f"Incentive explanation for training session {self.training_id} saved successfully in DB.")


    def save_global_model_evaluation(self, evaluation_text, shap_plot_buf, cm_plot_buf):
        training_session = TrainingModel.objects(training_id=self.training_id).first()
        if not training_session:
            logging.error(f"No training session found with ID {self.training_id}")
            return

        # Initialize or update the global_evaluation field
        if not training_session.global_evaluation:
            training_session.global_evaluation = GlobalModelEvaluation()

        # Save evaluation text
        training_session.global_evaluation.evaluation_text = evaluation_text

        # Assume shap_plot_buf and cm_plot_buf are already BytesIO objects
        # Reset buffer pointers to the start
        shap_plot_buf.seek(0)
        cm_plot_buf.seek(0)

        # Replace existing plot data with new data
        training_session.global_evaluation.shap_plot.replace(shap_plot_buf, content_type='image/png')
        training_session.global_evaluation.cm_plot.replace(cm_plot_buf, content_type='image/png')

        # Save changes to the database
        training_session.save()

        

    def collect_contribution_eval_metrics(self, round_num, contribution_metrics, shapley_plot):
        contribution_metrics_dir = os.path.join(self.output_dir, 'client', 'contribution')
        os.makedirs(contribution_metrics_dir, exist_ok=True)

        # Save the Shapley values in JSON format
        shapley_values_path = os.path.join(contribution_metrics_dir, f"client_shapley_values_round_{round_num}.json")
        with open(shapley_values_path, 'w') as file:
            json.dump(contribution_metrics, file, indent=4)

        # Save the Shapley values plot
        plot_path = os.path.join(contribution_metrics_dir, f"client_contribution_plot_round_{round_num}.png")
        shapley_plot.savefig(plot_path)
        plt.close(shapley_plot)
        
    
    def save_model_comparisons(self, explanations, plot_buffers):
        training_session = TrainingModel.objects(training_id=self.training_id).first()
        if not training_session:
            logging.error(f"No training session found with ID {self.training_id}")
            return

        for client_id, explanation in explanations.items():
            # Find or create the client data
            client_data = next((c for c in training_session.clients if c.client_id == client_id), None)
            if not client_data:
                client_data = ClientData(client_id=client_id)
                training_session.clients.append(client_data)

            # Create a new ModelComparison object
            model_comparison = ModelComparison(explanation=explanation)

            # Save comparison plot
            plot_buffer = io.BytesIO()
            plot_buffers[client_id].save(plot_buffer, format='png')
            plot_buffer.seek(0)
            model_comparison.comparison_plot.replace(plot_buffer, content_type='image/png')

            # Add the model comparison to the client data
            client_data.model_comparisons.append(model_comparison)

        # Save the updated document to the database
        training_session.save()
        
    
    def save_evaluation_plot(self, plot_path, client_id, round_num):
            training_session = TrainingModel.objects(training_id=self.training_id).first()
            if not training_session:
                logging.error(f"No training session found with ID {self.training_id}")
                return

            client_data = next((c for c in training_session.clients if c.client_id == client_id), None)
            if not client_data:
                client_data = ClientData(client_id=client_id)
                training_session.clients.append(client_data)

            # Create a new EvaluationPlot object
            evaluation_plot = EvaluationPlot(round=str(round_num))

            # Read the plot file
            with open(plot_path, 'rb') as file:
                plot_buffer = io.BytesIO(file.read())
                evaluation_plot.plot.replace(plot_buffer, content_type='image/png')

            # Add the evaluation plot to the client data
            client_data.evaluation_plots.append(evaluation_plot)

            # Save the updated document to the database
            training_session.save()

    # def save_comparison_plot(self, plot, round_num):
    #         # Create directory for storing comparison plots
    #         plot_save_dir = os.path.join(self.output_dir, 'FedXAIEvaluation', 'modelComparison')
    #         os.makedirs(plot_save_dir, exist_ok=True)
    #         file_path = os.path.join(plot_save_dir, f'comparison_plot_round_{round_num}.png')
        
    #         plot.savefig(file_path)
        