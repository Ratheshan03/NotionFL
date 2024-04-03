import copy
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

        return (shap_numpy, test_numpy)

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
            
            print(f"Comparing client {client_id} and global model shap values")  # Logging for debugging
            client_model_path = os.path.join(self.data_collector_path, 'client', 'localModels', f'client_{client_id}_model_round_{round_num}.pt')
            if not os.path.exists(client_model_path):
                print(f"Client model file not found: {client_model_path}")  # More detailed logging
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
        

    def explain_aggregation(self, round_num, data_loader):
        # Load the global model before and after aggregation
        global_model_pre_path = os.path.join(self.data_collector_path, 'global', 'models', f'global_model_round_{round_num}_pre.pt')
        global_model_pre = copy.deepcopy(self.global_model)
        global_model_pre.load_state_dict(torch.load(global_model_pre_path))
        global_model_pre.to(self.device).eval()

        global_model_post_path = os.path.join(self.data_collector_path, 'global', 'models', f'global_model_round_{round_num}_post.pt')
        global_model_post = copy.deepcopy(self.global_model)
        global_model_post.load_state_dict(torch.load(global_model_post_path))
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

        # Save the plot
        comparison_plot_path = os.path.join(self.data_collector_path, 'FedXAIEvaluation', 'aggregation_explanation', f'comparison_plot_round_{round_num}.png')
        os.makedirs(os.path.dirname(comparison_plot_path), exist_ok=True)
        plt.savefig(comparison_plot_path)
        plt.close()

        aggregation_explanation = {
            'global_pre': shap_values_pre_reshaped,
            'global_post': shap_values_post_reshaped,
            'comparison_plot': comparison_plot_path
        }

        return aggregation_explanation


    def explain_privacy_mechanism(self, client_id, round_num, test_loader, privacy_info):
        # Load the client's model before differential privacy noise is applied
        model_path = os.path.join(self.data_collector_path, 'client', 'localModels', f'client_{client_id}_model_round_{round_num}.pt')
        model_state = torch.load(model_path)
        model = copy.deepcopy(self.global_model)
        model.load_state_dict(model_state)
        model.eval()

        # Load the client's model after differential privacy noise is applied
        private_model_path = os.path.join(self.data_collector_path, 'client', 'localModels', f'client_{client_id}_model_round_{round_num}_private.pt')
        private_model_state = torch.load(private_model_path)
        private_model = copy.deepcopy(self.global_model)
        private_model.load_state_dict(private_model_state)
        private_model.to(self.device).eval()

        # Create an explainer with a background dataset
        background, _ = next(iter(test_loader))
        explainer = shap.GradientExplainer(model, background[:50].to(self.device))

        # Explain the model's predictions using SHAP values on a subset of test data
        test_images, _ = next(iter(test_loader))
        shap_values_before_privacy = explainer.shap_values(test_images.to(self.device))

        # Now explain the private model's predictions
        explainer.model = private_model
        shap_values_after_privacy = explainer.shap_values(test_images.to(self.device))

        # Compare SHAP values to evaluate the impact of differential privacy
        impact_of_privacy = self.evaluate_privacy_impact(shap_values_before_privacy, shap_values_after_privacy, privacy_info)

        # Store the comparison of SHAP values for later analysis or visualization
        privacy_explanation_path = os.path.join(self.data_collector_path, 'FedXAIEvaluation', 'privacy_explanations', f'client_{client_id}_privacy_explanation_round_{round_num}.json')
        os.makedirs(os.path.dirname(privacy_explanation_path), exist_ok=True)
        with open(privacy_explanation_path, 'w') as file:
            json.dump(impact_of_privacy, file)

        return impact_of_privacy

    
    
    def evaluate_privacy_impact(self, shap_values_without_privacy, shap_values_with_privacy, privacy_params):
        # Convert SHAP values to numpy arrays
        shap_values_without_privacy = np.array(shap_values_without_privacy)
        shap_values_with_privacy = np.array(shap_values_with_privacy)

        # Ensure SHAP values have the same shape
        assert shap_values_without_privacy.shape == shap_values_with_privacy.shape, "Shape mismatch in SHAP values"

        # Calculate the absolute difference in SHAP values for each feature
        diff_shap_values = np.abs(shap_values_without_privacy - shap_values_with_privacy)

        # Calculate the mean difference for each feature
        feature_diff = diff_shap_values.mean(axis=0)

        # Aggregate impact information
        impact_summary = {
            'mean_diff': np.mean(diff_shap_values),
            'max_diff': np.max(diff_shap_values),
            'privacy_noise': privacy_params,
            'feature_diff': feature_diff.tolist()  # Convert numpy array to list for JSON serialization
        }

        return impact_summary


    def interpret_privacy_impact(self, round_num, client_id):
        # Load DP explanation results
        privacy_explanation_path = os.path.join(self.data_collector_path, 'FedXAIEvaluation', 'privacy_explanations', f'client_{client_id}_privacy_explanation_round_{round_num}.json')
        with open(privacy_explanation_path, 'r') as file:
            dp_results = json.load(file)

        mean_diff = dp_results['mean_diff']
        max_diff = dp_results['max_diff']
        privacy_noise = dp_results['privacy_noise']
        feature_diff = np.array(dp_results['feature_diff'])

        # Flatten feature_diff if it's multidimensional (for image data)
        if feature_diff.ndim > 1:
            feature_diff = feature_diff.flatten()

        # Generate the indices for each feature
        feature_indices = np.arange(feature_diff.shape[0])

        # Generate detailed textual explanation
        interpretation_text = f"Client {client_id} Differential Privacy Analysis for Round {round_num}:\n"
        interpretation_text += f"The mean difference in SHAP values, averaging at {mean_diff:.4f}, indicates the average change in feature importance due to the applied differential privacy noise. A lower mean difference suggests that the model's interpretability remains stable even after introducing privacy-preserving noise.\n\n"
        interpretation_text += f"The maximum difference in SHAP values reached {max_diff:.4f}, reflecting the largest alteration in any single feature's importance. This metric helps identify if any specific feature's interpretability is significantly impacted by the privacy mechanism.\n\n"
        interpretation_text += f"With a privacy noise level of {privacy_noise}, the differential privacy implementation aims to balance model utility with user privacy. A higher noise level generally increases privacy at the potential cost of model accuracy and interpretability.\n\n"
        interpretation_text += "Privacy-Impact Trade-offs: This round's differential privacy implementation demonstrates a balance where the model maintains reasonable interpretability while providing privacy protection. Adjustments to the noise level may be considered based on the desired privacy-accuracy trade-off.\n\n"
        interpretation_text += "Feature Analysis: Investigating specific features with high SHAP value differences can provide insights into how differential privacy affects the model's reliance on certain features. This analysis is crucial for understanding the model's robustness under privacy constraints.\n"

        # Ensure directory for saving textual explanation
        interpretation_dir = os.path.join(self.data_collector_path,  'FedXAIEvaluation', 'privacy_explanations')
        os.makedirs(interpretation_dir, exist_ok=True)
        interpretation_path = os.path.join(interpretation_dir, f'interpretation_client_{client_id}_round_{round_num}.txt')

        # Save the textual explanation
        with open(interpretation_path, 'w') as file:
            file.write(interpretation_text)

        # Generate and save visualizations
        plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
        plt.bar(feature_indices, feature_diff)
        plt.xlabel('Feature Index')
        plt.ylabel('Difference in SHAP Value')
        plt.title(f'Impact of DP on Client {client_id} Model Features')

        # Ensure directory for saving visualization
        visualization_dir = os.path.join(self.data_collector_path, 'FedXAIEvaluation', 'privacy_explanations', 'visualizations')
        os.makedirs(visualization_dir, exist_ok=True)
        visualization_path = os.path.join(visualization_dir, f'impact_visualization_client_{client_id}_round_{round_num}.png')

        plt.savefig(visualization_path)
        plt.close()


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