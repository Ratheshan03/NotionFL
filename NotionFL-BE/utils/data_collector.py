import shutil
import matplotlib
matplotlib.use('Agg')
import json
import os
from matplotlib import pyplot as plt
import numpy as np
import shap
import logging

import torch
class DataCollector:
    def __init__(self, output_dir):
        self.output_dir = output_dir
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
        client_training_logs_dir = os.path.join(self.output_dir, 'client', 'training')
        file_path = os.path.join(client_training_logs_dir, f"client_{client_id}_training_logs.json")
        os.makedirs(client_training_logs_dir, exist_ok=True)
        
        with open(file_path, 'a') as file:
            json.dump(training_logs, file)
            file.write("\n")
            
        logging.info(f"Training logs saved successfully for client_{client_id}")
        
        
    def collect_client_evaluation_logs(self, client_id, evaluation_logs, round):
        client_evaluation_logs_dir = os.path.join(self.output_dir, 'client', 'evaluation')
        file_path = os.path.join(client_evaluation_logs_dir, f"client_{client_id}_evaluation_logs_round_{round + 1}.json")
        os.makedirs(client_evaluation_logs_dir, exist_ok=True)
        
        with open(file_path, 'a') as file:  # Append to the file if it exists
            json.dump(evaluation_logs, file)
            file.write("\n")
            
        logging.info(f"Evalaution logs saved successfully for client_{client_id}")
            

    def collect_client_model(self, client_id, model, round_num):
        client_models_dir = os.path.join(self.output_dir, 'client', 'localModels')
        model_path = os.path.join(client_models_dir, f"client_{client_id}_model_round_{round_num}.pt")
        os.makedirs(client_models_dir, exist_ok=True)
        
        torch.save(model, model_path)
        

    def collect_client_updates(self, client_id, model_update):
        client_update_dir = os.path.join(self.output_dir, 'client', 'modelUpdates')
        update_path = os.path.join(client_update_dir, f"client_{client_id}_update.pt")
        os.makedirs(client_update_dir, exist_ok=True)
        
        torch.save(model_update, update_path)
        

    def collect_client_model(self, client_id, model, round_num, suffix=''):
        client_models_dir = os.path.join(self.output_dir, 'client', 'localModels')
        model_filename = f"client_{client_id}_model_round_{round_num}{('_' + suffix) if suffix else ''}.pt"
        model_path = os.path.join(client_models_dir, model_filename)
        os.makedirs(client_models_dir, exist_ok=True)
        
        torch.save(model, model_path)
        

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

        # Save global model metrics
        global_metrics_dir = os.path.join(self.output_dir, 'global', 'evaluation')
        global_metrics_path = os.path.join(global_metrics_dir, f'global_model_round_{round_num + 1}.json')
        os.makedirs(global_metrics_dir, exist_ok=True)
        
        with open(global_metrics_path, 'w') as file:
            json.dump(metrics_dict, file)
            file.write("\n")

                   
    def collect_global_model(self, global_model_state, round_num):
        global_model_dir = os.path.join(self.output_dir, 'global', 'models')
        file_path = os.path.join(global_model_dir, f'global_model_round_{round_num}.pt')
        os.makedirs(global_model_dir, exist_ok=True)
        
        torch.save(global_model_state, file_path)
            

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
        
    
    def collect_contribution_metrics(self, contribution_metrics, shapley_plot):
        contribution_metrics_dir = os.path.join(self.output_dir,'client', 'contribution')
        os.makedirs(contribution_metrics_dir, exist_ok=True)

        # Save the Shapley values in JSON format
        shapley_values_path = os.path.join(contribution_metrics_dir, "clients_shapley_values.json")
        with open(shapley_values_path, 'w') as file:
            json.dump(contribution_metrics, file, indent=4)

        # Save the Shapley values plot
        plot_path = os.path.join(contribution_metrics_dir, "clients_contribution_plot.png")
        shapley_plot.savefig(plot_path)
        plt.close(shapley_plot)
        
    
    def save_incentives(self, incentives_json, incentive_plot_buf):
        # Define the directory for storing incentives related data
        incentives_dir = os.path.join(self.output_dir,'client', 'contribution')
        os.makedirs(incentives_dir, exist_ok=True)

        # Path for the incentives JSON file
        incentives_json_path = os.path.join(incentives_dir, "clients_incentives.json")
        with open(incentives_json_path, 'w') as file:
            json.dump(incentives_json, file, indent=4)

        # Path for the incentives plot
        incentive_plot_path = os.path.join(incentives_dir, "initial_incentive_plot.png")
        with open(incentive_plot_path, 'wb') as file:
            file.write(incentive_plot_buf.getvalue())


    def collect_secure_aggregation_logs(self, round_num, aggregation_metrics, time_overheads):
        secure_aggregation_data = {
            'round': round_num,
            'aggregation_metrics': aggregation_metrics,
            'time_overheads': time_overheads
        }
        aggregation_dir = os.path.join(self.output_dir, 'aggregation')
        file_path = os.path.join(aggregation_dir, f'secure_aggregation_round_{round_num}.json')
        os.makedirs(aggregation_dir, exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(secure_aggregation_data, file, indent=4)
            

    def collect_differential_privacy_logs(self, round_num, dp_metrics):
        privacy_dir = os.path.join(self.output_dir, 'privacy')
        file_path = os.path.join(privacy_dir, f"differential_privacy_round_{round_num}.json")
        os.makedirs(privacy_dir, exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(dp_metrics, file, indent=4)
    
    
    def save_client_model_evaluation(self, client_id, evaluation_text, shap_plot_buf):
        # Create directory for storing client model evaluation
        client_eval_dir = os.path.join(self.output_dir, 'FedXAIEvaluation', 'clients', f'client_{client_id}', 'evaluation')
        os.makedirs(client_eval_dir, exist_ok=True)

        # Save evaluation text
        evaluation_file = os.path.join(client_eval_dir, 'evaluation.txt')
        with open(evaluation_file, 'w') as file:
            file.write(evaluation_text)

        # Save SHAP plot
        shap_plot_file = os.path.join(client_eval_dir, 'shap_plot.png')
        with open(shap_plot_file, 'wb') as file:
            file.write(shap_plot_buf.getvalue())
        
        
    def save_global_model_evaluation(self, evaluation_text, shap_plot_buf, cm_plot_buf):
        # Save evaluation text
        evaluation_file = os.path.join(self.output_dir, 'FedXAIEvaluation', 'globals', 'final_evaluation.txt')
        os.makedirs(os.path.dirname(evaluation_file), exist_ok=True)
        with open(evaluation_file, 'w') as file:
            file.write(evaluation_text)

        # Save SHAP plot
        shap_plot_file = os.path.join(self.output_dir, 'FedXAIEvaluation', 'globals', 'final_shap_plot.png')
        with open(shap_plot_file, 'wb') as file:
            file.write(shap_plot_buf.getvalue())

        # Save Confusion Matrix plot
        cm_plot_file = os.path.join(self.output_dir, 'FedXAIEvaluation', 'globals', 'final_confusion_matrix.png')
        with open(cm_plot_file, 'wb') as file:
            file.write(cm_plot_buf.getvalue())


    def save_model_comparisons(self, explanations, plot_buffers):
        for client_id, explanation in explanations.items():
            # Save SHAP comparison details
            explanation_file = os.path.join(self.output_dir, 'FedXAIEvaluation', 'clients', f'client_{client_id}', 'comparison_details.json')
            os.makedirs(os.path.dirname(explanation_file), exist_ok=True)
            with open(explanation_file, 'w') as file:
                json.dump(explanation, file, indent=4)

            # Save comparison plots
            plot_save_dir = os.path.join(self.output_dir, 'FedXAIEvaluation', 'modelComparison')
            os.makedirs(plot_save_dir, exist_ok=True)
            plot_path = os.path.join(plot_save_dir, f'client_{client_id}_comparison_plot.png')
            with open(plot_path, 'wb') as file:
                file.write(plot_buffers[client_id].getvalue())

                

    # def save_comparison_plot(self, plot, round_num):
    #         # Create directory for storing comparison plots
    #         plot_save_dir = os.path.join(self.output_dir, 'FedXAIEvaluation', 'modelComparison')
    #         os.makedirs(plot_save_dir, exist_ok=True)
    #         file_path = os.path.join(plot_save_dir, f'comparison_plot_round_{round_num}.png')
        
    #         plot.savefig(file_path)
        
        
    def save_evaluation_plot(self, plot_path, client_id, round_num):
        # Ensure the directory exists
        eval_plots_dir = os.path.join(self.output_dir, 'FedXAIEvaluation', 'clients', f'client_{client_id}')
        os.makedirs(eval_plots_dir, exist_ok=True)

        # Construct the full path to save the plot
        final_plot_path = os.path.join(eval_plots_dir, f'comparison_shap_values_round_{round_num}.png')

        # Check if the plot is already in the final location
        if os.path.abspath(plot_path) != os.path.abspath(final_plot_path):
            # Move the plot file to the final directory if it's not already there
            shutil.move(plot_path, final_plot_path)
    
        
    def save_aggregation_explanation(self, aggregation_plot, round_num):
        agg_explanation_dir = os.path.join(self.output_dir, 'FedXAIEvaluation', 'aggregation_explanation')
        os.makedirs(agg_explanation_dir, exist_ok=True)

        # Path for saving the plot
        plot_path = os.path.join(agg_explanation_dir, f'aggregation_plot_round_{round_num}.png')

        # Save the plot from the buffer
        with open(plot_path, 'wb') as file:
            file.write(aggregation_plot.getbuffer())

        print(f"Aggregation plot for round {round_num} saved successfully.")

    
    def save_privacy_explanations(self, explanation_text, plot_buffer, client_id, round_num):
        # Save textual explanation
        interpretation_dir = os.path.join(self.output_dir, 'FedXAIEvaluation', 'privacy_explanations')
        os.makedirs(interpretation_dir, exist_ok=True)
        interpretation_path = os.path.join(interpretation_dir, f'interpretation_client_{client_id}_round_{round_num}.txt')
        with open(interpretation_path, 'w') as file:
            file.write(explanation_text)

        # Save visualization plot
        visualization_dir = os.path.join(self.output_dir, 'FedXAIEvaluation', 'privacy_explanations', 'visualizations')
        os.makedirs(visualization_dir, exist_ok=True)
        visualization_path = os.path.join(visualization_dir, f'impact_visualization_client_{client_id}_round_{round_num}.png')
        with open(visualization_path, 'wb') as f:
            f.write(plot_buffer.getvalue())
            
            
    def save_incentive_explanation(self, explanation_text, plot_buffers):
        # Save explanation text
        explanation_file = os.path.join(self.output_dir,'FedXAIEvaluation', 'contribution_explanations', 'incentive_explanation.txt')
        os.makedirs(os.path.dirname(explanation_file), exist_ok=True)
        with open(explanation_file, 'w') as file:
            file.write(explanation_text)

        # Save each plot in separate files
        plot_titles = ['incentive_allocation_vs_shapley_values', 'contribution_distribution', 'contributions_vs_incentives']
        for i, plot_buf in enumerate(plot_buffers):
            plot_file = os.path.join(self.output_dir, 'FedXAIEvaluation', 'contribution_explanations', f'{plot_titles[i]}.png')
            with open(plot_file, 'wb') as file:
                file.write(plot_buf.getvalue())

