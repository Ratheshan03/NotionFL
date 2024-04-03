import copy
import io
import os
import shap
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models.model import MNISTModel

class FederatedXAI:
    def __init__(self, data_collector_path, device, global_model):
        self.data_collector_path = data_collector_path
        self.device = device
        self.global_model = global_model

    
    def explain_client_model(self, client_id, round_num, test_loader):
        model_path = self.data_collector_path + f"/client/localModels/client_{client_id}_model_round_{round_num}.pt"
        model_state_dict = torch.load(model_path)
        model = copy.deepcopy(self.global_model)  # Use a copy of the global model
        model.load_state_dict(model_state_dict)
        model.to(self.device).eval()

        # Get a batch of data
        batch = next(iter(test_loader))
        images, _ = batch

        # Use images for explanation
        background = images[:50].to(self.device)
        test_images = images[50:64].to(self.device)

        # Create the explainer
        e = shap.GradientExplainer(model, background)
        shap_values = e.shap_values(test_images)

        # Convert SHAP values to a format suitable for plotting
        shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
        test_numpy = np.swapaxes(np.swapaxes(test_images.cpu().numpy(), 1, -1), 1, 2)

        # Generate the SHAP plot
        shap_plot = plt.figure()
        shap.image_plot(shap_values, -test_images)

        return shap_plot, (shap_numpy, test_numpy)
    

    def explain_global_model(self, round_num, data_loader):
        # Load the global model from the stored path
        global_model_path = os.path.join(self.data_collector_path, 'global', 'models', f'global_model_round_{round_num}.pt')
        global_model_state_dict = torch.load(global_model_path, map_location=self.device)
        self.global_model.load_state_dict(global_model_state_dict)
        self.global_model.eval()

        # Get a batch of data
        batch = next(iter(data_loader))
        images, _ = batch

        # Use images for explanation
        background = images[:50].to(self.device)
        test_images = images[50:64].to(self.device)  # Taking 14 images for test

        # Create the explainer
        e = shap.GradientExplainer(self.global_model, background)
        shap_values = e.shap_values(test_images)

        # Convert SHAP values to a format suitable for plotting
        shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
        test_numpy = np.swapaxes(np.swapaxes(test_images.cpu().numpy(), 1, -1), 1, 2)

        return shap_numpy, test_numpy
        
        

    def compare_models(self, round_num, num_clients):
        # Load the global model explanation
        global_shap_path = os.path.join(self.data_collector_path, 'FedXAIEvaluation', 'global', f"shap_explanation_round_{round_num}.png")
        global_shap_values = plt.imread(global_shap_path)

        # Prepare a plot for comparison
        fig, axes = plt.subplots(1, num_clients + 1, figsize=(20, 3))

        # Plot global model explanation
        axes[0].imshow(global_shap_values)
        axes[0].title.set_text('Global Model')
        axes[0].axis('off')

        # Load each client's explanation and plot
        for client_id in range(num_clients):
            client_shap_path = os.path.join(self.data_collector_path, 'FedXAIEvaluation', f'client_{client_id}', f"shap_explanation_round_{round_num}.png")
            client_shap_values = plt.imread(client_shap_path)

            axes[client_id + 1].imshow(client_shap_values)
            axes[client_id + 1].title.set_text(f'Client {client_id} Model')
            axes[client_id + 1].axis('off')

        # Adjust layout
        plt.tight_layout()
        
        return plt
    
    
    def compare_model_shap_values(self, round_num, num_clients, data_loader):
        explanations = {}
        global_model_path = os.path.join(self.data_collector_path, 'global', 'models', f'global_model_round_{round_num}.pt')
        global_model_state = torch.load(global_model_path)
        global_model = self.global_model
        global_model.load_state_dict(global_model_state)
        global_model.to(self.device).eval()

        background, _ = next(iter(data_loader))
        background = background[:64].to(self.device)
        

        for client_id in range(num_clients):
            global_explainer = shap.GradientExplainer(global_model, background)
            global_shap_values = global_explainer.shap_values(background)
            
            print(f"Comparing client {client_id} and global model shap values")
            client_model_path = os.path.join(self.data_collector_path, 'client', 'localModels', f'client_{client_id}_model_round_{round_num}.pt')
            if not os.path.exists(client_model_path):
                print(f"Client model file not found: {client_model_path}")
                continue  

            client_model_state = torch.load(client_model_path)
            client_model = self.global_model
            client_model.load_state_dict(client_model_state)
            client_model.to(self.device).eval()

            client_explainer = shap.GradientExplainer(client_model, background)
            client_shap_values = client_explainer.shap_values(background)

            comparison = self.evaluate_impact(global_shap_values, client_shap_values, round_num, client_id)
            explanations[client_id] = comparison

        return explanations
    
    
    def evaluate_impact(self, global_shap_values, client_shap_values, round_num, client_id):
        # Ensure SHAP values are numpy arrays and have the same shape
        global_shap_values = np.array(global_shap_values)
        client_shap_values = np.array(client_shap_values)
        assert global_shap_values.shape == client_shap_values.shape, "Shape mismatch in SHAP values"

        # Flatten the SHAP values if they are not already flat
        global_shap_flat = global_shap_values.reshape(global_shap_values.shape[0], -1)
        client_shap_flat = client_shap_values.reshape(client_shap_values.shape[0], -1)

        # Calculate the mean absolute difference in SHAP values across all features
        differences = np.abs(global_shap_flat - client_shap_flat).mean(axis=0)

        # Create a bar plot to visualize the impact
        feature_names = ['Pixel ' + str(i) for i in range(differences.shape[0])]
        plt.barh(feature_names[:30], differences[:30])  # Plotting first 10 features for readability
        plt.xlabel('Mean Absolute Difference in SHAP Value')
        plt.title('Impact of Client Model on Global Model')
        plt.gca().invert_yaxis()  # To display the highest value on top
        plt.tight_layout()

        # Save the plot
        # Define the full path for saving the plot
        shap_plots_dir = os.path.join(self.data_collector_path, 'FedXAIEvaluation', f'client_{client_id}')
        os.makedirs(shap_plots_dir, exist_ok=True)
        comparison_plot_path = os.path.join(shap_plots_dir, f'comparison_shap_values_round_{round_num}.png')

        # Save the plot
        plt.savefig(comparison_plot_path)
        plt.close()
        
        # Return the path to the saved plot and the raw differences
        impact_evaluation = {
            'global': global_shap_values.tolist(),
            'client': client_shap_values.tolist(),
            'difference': differences.tolist(),
            'comparison_plot': comparison_plot_path
        }
        
        return impact_evaluation
        

    def explain_aggregation(self, pre_aggregated_state, post_aggregated_state, data_loader, round_num):
        # Load the global model states into new model instances
        global_model_pre = copy.deepcopy(self.global_model)
        global_model_pre.load_state_dict(pre_aggregated_state)
        global_model_pre.to(self.device).eval()

        global_model_post = copy.deepcopy(self.global_model)
        global_model_post.load_state_dict(post_aggregated_state)
        global_model_post.to(self.device).eval()

        # Calculate SHAP values
        background, _ = next(iter(data_loader))
        background = background[:50].to(self.device)
        explainer_pre = shap.GradientExplainer(global_model_pre, background)
        explainer_post = shap.GradientExplainer(global_model_post, background)
        shap_values_pre = explainer_pre.shap_values(background)
        shap_values_post = explainer_post.shap_values(background)

        # Reshape SHAP values if they are not in 2D format (for image data)
        def reshape_shap_values(shap_values):
            if isinstance(shap_values, list):
                return [val.reshape(val.shape[0], -1) for val in shap_values]
            return shap_values

        shap_values_pre_reshaped = reshape_shap_values(shap_values_pre)
        shap_values_post_reshaped = reshape_shap_values(shap_values_post)

        # Generate comparison plot
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        shap.summary_plot(shap_values_pre_reshaped, show=False)
        plt.title('Global Model Pre-Aggregation')

        plt.subplot(1, 2, 2)
        shap.summary_plot(shap_values_post_reshaped, show=False)
        plt.title('Global Model Post-Aggregation')

       # Save plot to a buffer
        plot_buffer = io.BytesIO()
        plt.savefig(plot_buffer, format='png')
        plt.close()
        plot_buffer.seek(0)

        return plot_buffer

    
    def explain_privacy_impact(self, client_id, round_num, test_loader, privacy_params):
        """
        Explain, evaluate, interpret, and generate visualizations for the impact 
        of differential privacy on a client's model.

        Args:
            client_id: ID of the client.
            round_num: The current round of training.
            test_loader: DataLoader for the test set.
            privacy_params: Parameters used for differential privacy.

        Returns:
            A tuple containing the interpretation text and the visualization plot.
        """
        # Load models before and after applying differential privacy
        model = self.load_model(client_id, round_num, suffix="")
        private_model = self.load_model(client_id, round_num, suffix="_private")

        # Prepare the explainer
        background, _ = next(iter(test_loader))
        explainer = shap.GradientExplainer(model, background[:50].to(self.device))

        # Explain both models
        test_images, _ = next(iter(test_loader))
        shap_values = explainer.shap_values(test_images.to(self.device))
        explainer.model = private_model
        private_shap_values = explainer.shap_values(test_images.to(self.device))

        # Calculate differences in SHAP values
        diff_shap_values = np.abs(np.array(shap_values) - np.array(private_shap_values))
        feature_diff = diff_shap_values.mean(axis=0)
        
        mean_diff = np.mean(diff_shap_values)
        max_diff = np.max(diff_shap_values)
        privacy_noise = privacy_params
        feature_diff = feature_diff.tolist()
       
        # Interpretation Text
        interpretation_text = f"Client {client_id} Differential Privacy Analysis for Round {round_num}:\n"
        interpretation_text += f"The mean difference in SHAP values, averaging at {mean_diff:.4f}, indicates the average change in feature importance due to the applied differential privacy noise. A lower mean difference suggests that the model's interpretability remains stable even after introducing privacy-preserving noise.\n\n"
        interpretation_text += f"The maximum difference in SHAP values reached {max_diff:.4f}, reflecting the largest alteration in any single feature's importance. This metric helps identify if any specific feature's interpretability is significantly impacted by the privacy mechanism.\n\n"
        interpretation_text += f"With a privacy noise level of {privacy_noise}, the differential privacy implementation aims to balance model utility with user privacy. A higher noise level generally increases privacy at the potential cost of model accuracy and interpretability.\n\n"
        interpretation_text += "Privacy-Impact Trade-offs: This round's differential privacy implementation demonstrates a balance where the model maintains reasonable interpretability while providing privacy protection. Adjustments to the noise level may be considered based on the desired privacy-accuracy trade-off.\n\n"
        interpretation_text += "Feature Analysis: Investigating specific features with high SHAP value differences can provide insights into how differential privacy affects the model's reliance on certain features. This analysis is crucial for understanding the model's robustness under privacy constraints.\n"

        # Visualization
        feature_indices = np.arange(len(feature_diff))
        plt.figure(figsize=(10, 6))
        plt.bar(feature_indices, feature_diff)
        plt.xlabel('Feature Index')
        plt.ylabel('Difference in SHAP Value')
        plt.title(f'Impact of DP on Client {client_id} Model Features')
        
        # Save plot to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        return interpretation_text, buf
        
    
    def load_model(self, client_id, round_num, suffix):
        """
        Load a client's model based on the round number and suffix.

        Args:
            client_id: ID of the client.
            round_num: The current round of training.
            suffix: Suffix for the model filename (e.g., '_private').

        Returns:
            A loaded PyTorch model.
        """
        model_path = os.path.join(self.data_collector_path, 'client', 'localModels', f'client_{client_id}_model_round_{round_num}{suffix}.pt')
        model_state = torch.load(model_path)
        model = copy.deepcopy(self.global_model)
        model.load_state_dict(model_state)
        model.to(self.device).eval()
        return model


    def generate_incentive_explanation(self, round_num):
        shapley_values_file = os.path.join(self.data_collector_path, 'client', 'contribution', f'client_shapley_values_round_{round_num}.json')
        incentives_file = os.path.join(self.data_collector_path, 'client', 'contribution', f'client_incentives_round_{round_num}.json')
        explanation_dir = os.path.join(self.data_collector_path, 'client', 'contribution')
        explanation_file = os.path.join(explanation_dir, f'incentive_explanation_round_{round_num}.txt')

        # Ensure the directory exists
        os.makedirs(explanation_dir, exist_ok=True)
        
        # Read the Shapley values and incentives
        with open(shapley_values_file, 'r') as file:
            shapley_values = json.load(file)
        with open(incentives_file, 'r') as file:
            incentives = json.load(file)

        # Generate the explanation text
        explanation = self._compose_incentive_explanation_text(shapley_values, incentives)
        
        # Save the explanation to a file
        with open(explanation_file, 'w') as file:
            file.write(explanation)

        # Generate and save the plot
        self._create_incentive_plot(shapley_values, incentives, round_num)

    def _compose_incentive_explanation_text(self, shapley_values, incentives):
        explanation = "Federated Learning Incentive Allocation Explanation\n"
        explanation += "-------------------------------------------------\n\n"
        explanation += "The incentives for each client in the federated learning system were allocated based on the Shapley values calculated for each client.\n\n"
        explanation += "Shapley Value Breakdown:\n"
        for client_id, value in shapley_values.items():
            explanation += f"- Client {client_id}: Shapley Value = {value}\n"
        explanation += "\nIncentive Allocation:\n"
        for client_id, incentive in incentives.items():
            explanation += f"- Client {client_id}: Incentive = ${incentive:.2f}\n"
        explanation += "\nThis method ensures a fair distribution of the total incentive pool based on the contribution of each client.\n"
        return explanation

    def _create_incentive_plot(self, shapley_values, incentives, round_num):
        plt.figure(figsize=(10, 6))
        plt.bar(shapley_values.keys(), shapley_values.values(), color='blue', label='Shapley Values')
        plt.bar(incentives.keys(), incentives.values(), color='green', alpha=0.5, label='Incentives')
        plt.xlabel('Client ID')
        plt.ylabel('Amount')
        plt.title(f'Incentive Allocation vs Shapley Values (Round {round_num})')
        plt.legend()
        
        plot_file = os.path.join(self.data_collector_path, 'client', 'contribution', f'incentive_plot_round_{round_num}.png')
        plt.savefig(plot_file)
        plt.close()