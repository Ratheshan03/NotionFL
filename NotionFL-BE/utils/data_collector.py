import io
import matplotlib
matplotlib.use('Agg')
import json
import os
from matplotlib import pyplot as plt
import numpy as np
import logging
from utils.file_handler import FileHandler
import torch
class DataCollector:
    def __init__(self, output_dir, training_id):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.training_id = training_id
        self.file_handler = FileHandler()

    def collect_client_training_logs(self, client_id, training_logs, performance_plot):
        # Serialize the training logs to a JSON string
        training_logs_json = json.dumps(training_logs)
        training_id = self.training_id
        
        cloud_file_path = f'{training_id}/client/training/client_{client_id}_training_logs.json'
        self.file_handler.store_file(training_logs_json, cloud_file_path)
        
       # Store performance plot
        if performance_plot:  # Check if plot object exists
            plot_buffer = io.BytesIO()
            performance_plot.savefig(plot_buffer, format='png')
            plot_buffer.seek(0)
            plot_file_path = f'{training_id}/client/training/client_{client_id}_performance_plot.png'
            self.file_handler.store_file(plot_buffer.getvalue(), plot_file_path)

            
        logging.info(f"Training logs,plot for client_{client_id} successfully saved in cloud storage under training ID {training_id}")

        
    def collect_client_evaluation_logs(self, client_id, evaluation_logs, round):
        # Serialize the evaluation logs to a JSON string
        evaluation_logs_json = json.dumps(evaluation_logs)

        training_id = self.training_id  
        cloud_file_path = f'{training_id}/client/evaluation/client_{client_id}_evaluation_logs_round_{round}.json'

        # Use FileHandler to upload the data directly to Firebase Cloud Storage
        self.file_handler.store_file(evaluation_logs_json, cloud_file_path)
        
        logging.info(f"Evaluation logs for client_{client_id} successfully saved in cloud storage under training ID {training_id}, round {round}")
            

    def collect_client_model(self, client_id, model, round_num):
        buffer = io.BytesIO()
        torch.save(model, buffer)
        buffer.seek(0) 

        training_id = self.training_id 
        cloud_file_path = f'{training_id}/client/localModels/client_{client_id}_model_round_{round_num}.pt'

        # Use FileHandler to upload the data directly to Firebase Cloud Storage
        self.file_handler.store_file(buffer.getvalue(), cloud_file_path)

        logging.info(f"Model for client_{client_id} successfully saved in cloud storage under training ID {training_id}, round {round_num}")
            

    def collect_client_updates(self, client_id, model_update):
        buffer = io.BytesIO()
        torch.save(model_update, buffer)
        buffer.seek(0) 

        training_id = self.training_id 
        cloud_file_path = f'{training_id}/client/modelUpdates/client_{client_id}_update.pt'

        # Use FileHandler to upload the data directly to Firebase Cloud Storage
        self.file_handler.store_file(buffer.getvalue(), cloud_file_path)

        logging.info(f"Model update for client_{client_id} successfully saved in cloud storage under training ID {training_id}")
            

    def collect_client_model(self, client_id, model, round_num, suffix=''):
        buffer = io.BytesIO()
        torch.save(model, buffer)
        buffer.seek(0) 

        model_filename = f"client_{client_id}_model_round_{round_num}{('_' + suffix) if suffix else ''}.pt"
        training_id = self.training_id
        cloud_file_path = f'{training_id}/client/localModels/{model_filename}'

        # Use FileHandler to upload the data directly to Firebase Cloud Storage
        self.file_handler.store_file(buffer.getvalue(), cloud_file_path)

        logging.info(f"Model for client_{client_id} successfully saved in cloud storage under training ID {training_id}, round {round_num}, with suffix '{suffix}'")
        
        
    def collect_final_client_model(self, client_id, model):
        buffer = io.BytesIO()
        torch.save(model, buffer)
        buffer.seek(0) 

        model_filename = f"client_{client_id}_final_model.pt"
        training_id = self.training_id
        cloud_file_path = f'{training_id}/client/localModels/{model_filename}'

        # Use FileHandler to upload the data directly to Firebase Cloud Storage
        self.file_handler.store_file(buffer.getvalue(), cloud_file_path)

        logging.info(f"Final model for client_{client_id} successfully saved in cloud storage under training ID {training_id}")


    def collect_global_model_metrics(self, round_num, model_metrics):
        # Convert the metrics tuple into a dictionary
        metrics_dict = {
            'test_loss': model_metrics[0],
            'accuracy': model_metrics[1],
            'precision': model_metrics[2],
            'recall': model_metrics[3],
            'f1': model_metrics[4],
            'conf_matrix': model_metrics[5].tolist() if isinstance(model_metrics[5], np.ndarray) else model_metrics[5]
        }

        # Serialize the metrics dictionary to a JSON string
        metrics_json = json.dumps(metrics_dict)

        # Construct the cloud storage file path
        training_id = self.training_id
        cloud_file_path = f'{training_id}/global/evaluation/global_model_round_{round_num}.json'

        self.file_handler.store_file(metrics_json, cloud_file_path)

        logging.info(f"Global model metrics for round {round_num} successfully saved in cloud storage under training ID {training_id}")

                    
    def collect_global_model(self, global_model_state, round_num):
        buffer = io.BytesIO()
        torch.save(global_model_state, buffer)
        buffer.seek(0)

        training_id = self.training_id
        cloud_file_path = f'{training_id}/global/models/global_model_round_{round_num}.pt'

        self.file_handler.store_file(buffer.getvalue(), cloud_file_path)

        logging.info(f"Global model for round {round_num} successfully saved in cloud storage under training ID {training_id}")
                
                
                
    def collect_final_global_model(self, global_model_state):
        buffer = io.BytesIO()
        torch.save(global_model_state, buffer)
        buffer.seek(0)

        training_id = self.training_id
        cloud_file_path = f'{training_id}/global/models/final_global_model.pt'

        self.file_handler.store_file(buffer.getvalue(), cloud_file_path)

        logging.info(f"Final Global model successfully saved in cloud storage under training ID {training_id}")
        
                    

    def collect_contribution_eval_metrics(self, round_num, contribution_metrics, shapley_plot):
        contribution_metrics_json = json.dumps(contribution_metrics, indent=4)
        training_id = self.training_id 
        metrics_cloud_file_path = f'{training_id}/client/contribution/client_shapley_values_round_{round_num}.json'

        # Use FileHandler to upload the metrics data directly to Firebase Cloud Storage
        self.file_handler.store_file(contribution_metrics_json, metrics_cloud_file_path)

        buffer = io.BytesIO()
        shapley_plot.savefig(buffer, format='png')
        buffer.seek(0) 

        plot_cloud_file_path = f'{training_id}/client/contribution/client_contribution_plot_round_{round_num}.png'

        # Use FileHandler to upload the plot directly to Firebase Cloud Storage
        self.file_handler.store_file(buffer.getvalue(), plot_cloud_file_path)

        plt.close(shapley_plot)

        logging.info(f"Contribution metrics and plot for round {round_num} successfully saved in cloud storage under training ID {training_id}")
        
    
    def collect_contribution_metrics(self, contribution_metrics, shapley_plot):
        contribution_metrics_json = json.dumps(contribution_metrics, indent=4)
        training_id = self.training_id
        cloud_metrics_path = f'{training_id}/client/contribution/clients_shapley_values.json'

        self.file_handler.store_file(contribution_metrics_json, cloud_metrics_path)
        
        buffer = io.BytesIO()
        shapley_plot.savefig(buffer, format='png')
        buffer.seek(0) 
        training_id = self.training_id
        cloud_plot_path = f'{training_id}/client/contribution/clients_contribution_plot.png'

        self.file_handler.store_file(buffer.getvalue(), cloud_plot_path)

        plt.close(shapley_plot)

        logging.info("Contribution metrics and plot successfully saved in cloud storage")
            
    
    def save_incentives(self, incentives_json, incentive_plot_buf):
        incentives_json_string = json.dumps(incentives_json, indent=4)
        training_id = self.training_id
        cloud_json_path = f'{training_id}/client/contribution/clients_incentives.json'
        self.file_handler.store_file(incentives_json_string, cloud_json_path)

        incentive_plot_buf.seek(0)
        
        cloud_plot_path = f'{training_id}/client/contribution/initial_incentive_plot.png'

        self.file_handler.store_file(incentive_plot_buf.getvalue(), cloud_plot_path)

        logging.info("Incentives JSON and plot successfully saved in cloud storage")


    def collect_secure_aggregation_logs(self, round_num, aggregation_metrics, time_overheads):
        secure_aggregation_data = {
            'round': round_num,
            'aggregation_metrics': aggregation_metrics,
            'time_overheads': time_overheads
        }

        secure_aggregation_data_json = json.dumps(secure_aggregation_data, indent=4)

        training_id = self.training_id
        cloud_file_path = f'{training_id}/aggregation/secure_aggregation_round_{round_num}.json'

        self.file_handler.store_file(secure_aggregation_data_json, cloud_file_path)

        logging.info(f"Secure aggregation logs for round {round_num} successfully saved in cloud storage under training ID {training_id}")
            

    def collect_differential_privacy_logs(self, round_num, dp_metrics):
        dp_metrics_json = json.dumps(dp_metrics, indent=4)

        training_id = self.training_id
        cloud_file_path = f'{training_id}/privacy/differential_privacy_round_{round_num}.json'

        self.file_handler.store_file(dp_metrics_json, cloud_file_path)

        logging.info(f"Differential privacy logs for round {round_num} successfully saved in cloud storage under training ID {training_id}")
        
    
    def save_client_model_evaluation(self, client_id, evaluation_text, shap_plot_buf, round):
        training_id = self.training_id 
        text_cloud_file_path = f'{training_id}/FedXAIEvaluation/clients/client_{client_id}/evaluation/evaluation_{round}.txt'
        self.file_handler.store_file(evaluation_text, text_cloud_file_path)
        shap_plot_buf.seek(0)

        plot_cloud_file_path = f'{training_id}/FedXAIEvaluation/clients/client_{client_id}/evaluation/shap_plot_{round}.png'

        self.file_handler.store_file(shap_plot_buf.getvalue(), plot_cloud_file_path)

        logging.info(f"Client model evaluation for client_{client_id} successfully saved in cloud storage under training ID {training_id}")
        
        
    def save_global_model_shapplot(self, shap_plot_buf, round):
        training_id = self.training_id 
        shap_plot_buf.seek(0)
        plot_cloud_file_path = f'{training_id}/FedXAIEvaluation/globals/shap_plot_round_{round}.png'

        self.file_handler.store_file(shap_plot_buf.getvalue(), plot_cloud_file_path)

        logging.info(f"Global model shap plot successfully saved in cloud storage under training ID {training_id}")
            
        
    def save_global_model_evaluation(self, evaluation_text, shap_plot_buf, cm_plot_buf):
        training_id = self.training_id 
        text_cloud_file_path = f'{training_id}/FedXAIEvaluation/globals/final_evaluation.txt'
        self.file_handler.store_file(evaluation_text, text_cloud_file_path)

        shap_plot_buf.seek(0)
        cm_plot_buf.seek(0)

        shap_plot_cloud_file_path = f'{training_id}/FedXAIEvaluation/globals/final_shap_plot.png'
        self.file_handler.store_file(shap_plot_buf.getvalue(), shap_plot_cloud_file_path)

        cm_plot_cloud_file_path = f'{training_id}/FedXAIEvaluation/globals/final_confusion_matrix.png'
        self.file_handler.store_file(cm_plot_buf.getvalue(), cm_plot_cloud_file_path)

        logging.info(f"Global model evaluation successfully saved in cloud storage under training ID {training_id}")


    def save_model_comparisons(self, explanations, plot_buffers):
        training_id = self.training_id 
        for client_id, explanation in explanations.items():
            explanation_json = json.dumps(explanation, indent=4)

            explanation_cloud_path = f'{training_id}/FedXAIEvaluation/modelComparison/client_{client_id}/comparison_details.json'
            self.file_handler.store_file(explanation_json, explanation_cloud_path)

            plot_buffers[client_id].seek(0)

            plot_cloud_path = f'{training_id}/FedXAIEvaluation/modelComparison/client_{client_id}_comparison_plot.png'
            self.file_handler.store_file(plot_buffers[client_id].getvalue(), plot_cloud_path)

        logging.info(f"Model comparison details and plots successfully saved in cloud storage under training ID {training_id}")

                
    def save_comparison_plot(self, plot, round_num):
        training_id = self.training_id 
        # Create directory for storing comparison plots
        plot_buffer = io.BytesIO()  # Create a buffer for the plot

        # Save the plot to the buffer
        plot.savefig(plot_buffer, format='png')
        plot_buffer.seek(0)
        
        file_path = f'{training_id}/FedXAIEvaluation/modelComparison/comparison_plot_round_{round_num}.png'
        self.file_handler.store_file(plot_buffer.getvalue(), file_path)
        plot_buffer.close() 
        
        
    def save_evaluation_plot(self, plot_path, client_id, round_num):
        with open(plot_path, 'rb') as file:
            plot_buffer = io.BytesIO(file.read())

        training_id = self.training_id 
        cloud_file_path = f'{training_id}/FedXAIEvaluation/clients/client_{client_id}/comparison_shap_values_round_{round_num}.png'
        self.file_handler.store_file(plot_buffer.getvalue(), cloud_file_path)

        os.remove(plot_path)

        logging.info(f"Evaluation plot for client_{client_id}, round {round_num}, successfully saved in cloud storage under training ID {training_id}")
        
        
    def save_aggregation_explanation(self, aggregation_plot, round_num):
        training_id = self.training_id
        aggregation_plot.seek(0)
        cloud_file_path = f'{training_id}/FedXAIEvaluation/aggregation_explanation/aggregation_plot_round_{round_num}.png'
        self.file_handler.store_file(aggregation_plot.getvalue(), cloud_file_path)
        logging.info(f"Aggregation explanation plot for round {round_num} successfully saved in cloud storage under training ID {training_id}")

    
    def save_privacy_explanations(self, explanation_text, plot_buffer, client_id, round_num):
        training_id = self.training_id 

        explanation_cloud_path = f'{training_id}/FedXAIEvaluation/privacy_explanations/interpretation_client_{client_id}_round_{round_num}.txt'
        self.file_handler.store_file(explanation_text, explanation_cloud_path)

        plot_buffer.seek(0)

        visualization_cloud_path = f'{training_id}/FedXAIEvaluation/privacy_explanations/visualizations/impact_visualization_client_{client_id}_round_{round_num}.png'
        self.file_handler.store_file(plot_buffer.getvalue(), visualization_cloud_path)

        logging.info(f"Privacy explanations for client {client_id}, round {round_num}, successfully saved in cloud storage under training ID {training_id}")
            
            
    def save_incentive_explanation(self, explanation_text, plot_buffers):
        training_id = self.training_id 
        explanation_text_cloud_path = f'{training_id}/FedXAIEvaluation/contribution_explanations/incentive_explanation.txt'
        self.file_handler.store_file(explanation_text, explanation_text_cloud_path)

        # Plot titles
        plot_titles = ['incentive_allocation_vs_shapley_values', 'contribution_distribution', 'contributions_vs_incentives']

        # Save each plot in separate files
        for i, plot_buf in enumerate(plot_buffers):
            plot_buf.seek(0)
            plot_cloud_file_path = f'{training_id}/FedXAIEvaluation/contribution_explanations/{plot_titles[i]}.png'
            self.file_handler.store_file(plot_buf.getvalue(), plot_cloud_file_path)

        logging.info(f"Incentive explanation text and plots successfully saved in cloud storage under training ID {training_id}")
