import copy
import io
import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import logging
from utils.file_handler import FileHandler

class FederatedXAI:
    def __init__(self, data_collector_path, device, global_model, server, training_id):
        self.data_collector_path = data_collector_path
        self.device = device
        self.global_model = global_model
        self.server = server
        self.training_id = training_id
        self.file_handler = FileHandler()
        
        
    def explain_client_model(self, client_model_state, client_id, test_loader):
        # Load client model state
        model = copy.deepcopy(self.global_model)
        model.load_state_dict(client_model_state)
        model.to(self.device).eval()

        accuracy = self.server.evaluate_model(model, test_loader, self.device)
        evaluation_text = f"Client {client_id} Model Evaluation\nAccuracy: {accuracy:.2f}\n\n"

        # Get a batch of data for SHAP explanation
        batch = next(iter(test_loader))
        images, _ = batch

        # Use images for SHAP explanation
        background = images[:50].to(self.device)
        test_images = images[50:64].to(self.device)

        # Create the SHAP explainer
        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(test_images)

        # Convert SHAP values for plotting
        shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
        test_numpy = np.swapaxes(np.swapaxes(test_images.cpu().numpy(), 1, -1), 1, 2)

        # Generate the SHAP plot
        shap_plot_buf = io.BytesIO()
        plt.figure()
        shap.image_plot(shap_numpy, -test_numpy)
        plt.savefig(shap_plot_buf, format='png')
        shap_plot_buf.seek(0)
        plt.close()

        return evaluation_text, shap_plot_buf, (shap_numpy, test_numpy)
    
    
    def ex_global_model(self, model_state, test_loader):
        # Load client model state
        model = copy.deepcopy(self.global_model)
        model.load_state_dict(model_state)
        model.to(self.device).eval()

        self.server.evaluate_model(model, test_loader, self.device)

        # Get a batch of data for SHAP explanation
        batch = next(iter(test_loader))
        images, _ = batch

        # Use images for SHAP explanation
        background = images[:50].to(self.device)
        test_images = images[50:64].to(self.device)

        # Create the SHAP explainer
        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(test_images)

        # Convert SHAP values for plotting
        shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
        test_numpy = np.swapaxes(np.swapaxes(test_images.cpu().numpy(), 1, -1), 1, 2)

        # Generate the SHAP plot
        shap_plot_buf = io.BytesIO()
        plt.figure()
        shap.image_plot(shap_numpy, -test_numpy)
        plt.savefig(shap_plot_buf, format='png')
        shap_plot_buf.seek(0)
        plt.close()

        return shap_plot_buf, (shap_numpy, test_numpy)
    

    def explain_global_model(self, test_loader):
        # Evaluate the global model's performance using the server's method
        test_loss, accuracy, precision, recall, f1, conf_matrix = self.server.evaluate_global_model(test_loader, self.device)
        evaluation_text = f"Global Model Evaluation\nAccuracy: {accuracy:.2f}, Loss: {test_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}\n\n"

        # Generate the confusion matrix plot
        fig_conf_matrix = plt.figure(figsize=(8, 8))
        ax = fig_conf_matrix.add_subplot(111)
        ConfusionMatrixDisplay(conf_matrix).plot(ax=ax)
        plt.title("Confusion Matrix")

        # Save confusion matrix plot to a buffer
        buf_conf_matrix = io.BytesIO()
        plt.savefig(buf_conf_matrix, format='png')
        buf_conf_matrix.seek(0)
        plt.close(fig_conf_matrix)

        # Get a batch of data for SHAP explanation
        batch = next(iter(test_loader))
        images, _ = batch

        # Use images for SHAP explanation
        background = images[:50].to(self.device)
        test_images = images[50:64].to(self.device)

        # Create the SHAP explainer
        explainer = shap.GradientExplainer(self.global_model, background)
        shap_values = explainer.shap_values(test_images)

        # Convert SHAP values for plotting
        shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
        test_numpy = np.swapaxes(np.swapaxes(test_images.cpu().numpy(), 1, -1), 1, 2)

        shap_plot_buf = io.BytesIO()
        plt.figure()
        shap.image_plot(shap_numpy, -test_numpy)
        plt.savefig(shap_plot_buf, format='png')
        shap_plot_buf.seek(0)
        plt.close()
        
        return evaluation_text, buf_conf_matrix, shap_plot_buf, (shap_numpy, test_numpy)
        
        
    def compare_models(self, round_num, num_clients):
        global_shap_path = f'FedXAIEvaluation/globals/shap_plot_round_{round_num}.png'
        global_shap_bytes = self.file_handler.retrieve_file(self.training_id, global_shap_path)
        global_shap_values = plt.imread(io.BytesIO(global_shap_bytes), format='png') if global_shap_bytes else None

        fig, axes = plt.subplots(1, max(num_clients, 2), figsize=(20, 3))

        # If there is only one client, axes is a single object, not a list
        if num_clients == 1:
            axes = [axes]

        if global_shap_values is not None:
            axes[0].imshow(global_shap_values)
            axes[0].title.set_text('Global Model')
            axes[0].axis('off')

        for client_id in range(num_clients):
            client_shap_path = f"FedXAIEvaluation/clients/client_{client_id}/evaluation/shap_plot_round_{round_num}.png"
            client_shap_bytes = self.file_handler.retrieve_file(self.training_id, client_shap_path)
            client_shap_values = plt.imread(io.BytesIO(client_shap_bytes), format='png') if client_shap_bytes else None

            if client_shap_values is not None:
                axes[client_id + 1].imshow(client_shap_values)
                axes[client_id + 1].title.set_text(f'Client {client_id} Model')
                axes[client_id + 1].axis('off')

        plt.tight_layout()
        return plt

    # Needs to be fixed
    def explain_combined_models(self, num_clients, data_loader):
        explanations = {}
        plot_buffers = {}
        global_model = self.global_model
        global_model.to(self.device).eval()

        background, _ = next(iter(data_loader))
        background = background[:64].to(self.device)

        global_explainer = shap.GradientExplainer(global_model, background)
        global_shap_values = global_explainer.shap_values(background)
        global_shap_flat = np.array(global_shap_values).reshape(-1)

        for client_id in range(num_clients):
            client_model_path = f'client/localModels/client_{client_id}_final_model.pt'
            client_model_bytes = self.file_handler.retrieve_file(self.training_id, client_model_path)
            if client_model_bytes is None:
                continue

            model_stream = io.BytesIO(client_model_bytes)
            client_model_state = torch.load(model_stream)

            client_model = copy.deepcopy(global_model)
            client_model.load_state_dict(client_model_state)
            client_model.to(self.device).eval()

            client_explainer = shap.GradientExplainer(client_model, background)
            client_shap_values = client_explainer.shap_values(background)
            
            # Handle SHAP values, ensuring they are flattened and have the same dimensions
            client_shap_flat = np.array(client_shap_values).reshape(-1) if isinstance(client_shap_values, np.ndarray) else np.array(client_shap_values[0]).reshape(-1)

            # Check if dimensions match before comparison
            if global_shap_flat.shape[0] != client_shap_flat.shape[0]:
                print(f"Dimension mismatch for client {client_id}: Global SHAP {global_shap_flat.shape}, Client SHAP {client_shap_flat.shape}")
                continue

            # Calculate mean absolute difference in SHAP values
            differences = np.abs(global_shap_flat - client_shap_flat).mean(axis=0)

            # Plotting the comparison
            plt.figure(figsize=(10, 4))
            plt.bar(['Global Model', f'Client {client_id} Model'], [global_shap_flat.mean(), client_shap_flat.mean()])
            plt.ylabel('Mean SHAP Value')
            plt.title(f'Comparison of Mean SHAP Values: Global vs Client {client_id}')
            
            plot_buf = io.BytesIO()
            plt.savefig(plot_buf, format='png')
            plot_buf.seek(0)
            plt.close()

            # Store results
            explanations[client_id] = {
                'global': global_shap_values[0].tolist() if isinstance(global_shap_values, list) else global_shap_values.tolist(),
                'client': client_shap_values[0].tolist() if isinstance(client_shap_values, list) else client_shap_values.tolist(),
                'difference': differences.tolist()
            }
            plot_buffers[client_id] = plot_buf

        return explanations, plot_buffers
    
    
    def explain_aggregation(self, pre_aggregated_state, post_aggregated_state, data_loader, round_num):
        global_model_pre = copy.deepcopy(self.global_model)
        global_model_pre.load_state_dict(pre_aggregated_state)
        global_model_pre.to(self.device).eval()

        global_model_post = copy.deepcopy(self.global_model)
        global_model_post.load_state_dict(post_aggregated_state)
        global_model_post.to(self.device).eval()

        background, _ = next(iter(data_loader))
        background = background[:50].to(self.device)
        explainer_pre = shap.GradientExplainer(global_model_pre, background)
        explainer_post = shap.GradientExplainer(global_model_post, background)
        shap_values_pre = explainer_pre.shap_values(background)
        shap_values_post = explainer_post.shap_values(background)


        def reshape_shap_values(shap_values):
            if isinstance(shap_values, list):
                return [val.reshape(val.shape[0], -1) for val in shap_values]
            return shap_values


        def capture_shap_summary_plot(shap_values, num_features=10):
            fig, ax = plt.subplots(1, 1)
            shap.summary_plot(shap_values, plot_type="bar", max_display=num_features, show=False)
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return image
        
        # Predictions comparison
        def compare_predictions(model_pre, model_post, data_loader):
            inputs, labels = next(iter(data_loader))
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                preds_pre = model_pre(inputs).cpu().numpy()
                preds_post = model_post(inputs).cpu().numpy()
            
            return inputs.cpu().numpy(), labels.cpu().numpy(), preds_pre, preds_post

        inputs, labels, preds_pre, preds_post = compare_predictions(global_model_pre, global_model_post, data_loader)

        # Accuracy comparison
        def calculate_accuracy(model, data_loader):
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, labels in data_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            return correct / total

        accuracy_pre = calculate_accuracy(global_model_pre, data_loader)
        accuracy_post = calculate_accuracy(global_model_post, data_loader)

        shap_values_pre_reshaped = reshape_shap_values(shap_values_pre)
        shap_values_post_reshaped = reshape_shap_values(shap_values_post)

        pre_agg_shap_image = capture_shap_summary_plot(shap_values_pre_reshaped)
        post_agg_shap_image = capture_shap_summary_plot(shap_values_post_reshaped)

        fig, axs = plt.subplots(3, 2, figsize=(20, 15))

        num_examples = min(len(background), 3)
        for i in range(num_examples):
            axs[i, 0].imshow(inputs[i].transpose(1, 2, 0))
            axs[i, 0].set_title(f'Label: {labels[i]}, Pred Pre: {preds_pre[i]}, Pred Post: {preds_post[i]}')
            axs[i, 0].axis('off')

        for j in range(num_examples, 3):
            axs[j, 0].axis('off')

        axs[0, 1].imshow(pre_agg_shap_image)
        axs[0, 1].axis('off')
        axs[0, 1].set_title('Global Model Pre-Aggregation')

        axs[1, 1].imshow(post_agg_shap_image)
        axs[1, 1].axis('off')
        axs[1, 1].set_title('Global Model Post-Aggregation')

        # Accuracy comparison plot
        axs[2, 1].bar(['Pre-Aggregation', 'Post-Aggregation'], [accuracy_pre, accuracy_post])
        axs[2, 1].set_title('Model Accuracy Comparison')
        axs[2, 1].set_ylim(0, 1)

        plt.tight_layout()
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

        # Ensure the SHAP value differences are flattened into 1D
        if diff_shap_values.ndim > 1:
            feature_diff = diff_shap_values.mean(axis=0).flatten()
        else:
            feature_diff = diff_shap_values.mean(axis=0)

        mean_diff = np.mean(diff_shap_values)
        max_diff = np.max(diff_shap_values)
        privacy_noise = privacy_params

        # Interpretation Text
        interpretation_text = f"Client {client_id} Differential Privacy Analysis for Round {round_num}:\n"
        interpretation_text += f"The mean difference in SHAP values, averaging at {mean_diff:.4f}, indicates the average change in feature importance due to the applied differential privacy noise. A lower mean difference suggests that the model's interpretability remains stable even after introducing privacy-preserving noise.\n\n"
        interpretation_text += f"The maximum difference in SHAP values reached {max_diff:.4f}, reflecting the largest alteration in any single feature's importance. This metric helps identify if any specific feature's interpretability is significantly impacted by the privacy mechanism.\n\n"
        interpretation_text += f"With a privacy noise level of {privacy_noise}, the differential privacy implementation aims to balance model utility with user privacy. A higher noise level generally increases privacy at the potential cost of model accuracy and interpretability.\n\n"
        interpretation_text += "Privacy-Impact Trade-offs: This round's differential privacy implementation demonstrates a balance where the model maintains reasonable interpretability while providing privacy protection. Adjustments to the noise level may be considered based on the desired privacy-accuracy trade-off.\n\n"
        interpretation_text += "Feature Analysis: Investigating specific features with high SHAP value differences can provide insights into how differential privacy affects the model's reliance on certain features. This analysis is crucial for understanding the model's robustness under privacy constraints.\n"

        logging.info('Generating DP explanation plot and text file for client model')
        
        # Visualization
        feature_indices = np.arange(feature_diff.size)  # Adjust the size to match the flattened array
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
        
    
    def generate_incentive_explanation(self, shapley_values, incentives):
        # Explanation Text
        explanation = "Federated Learning Incentive Allocation Explanation\n"
        explanation += "-------------------------------------------------\n\n"
        explanation += "Incentives for each client in the federated learning system are allocated based on the Shapley values calculated for each client.\n\n"
        explanation += "Shapley Value Breakdown:\n"
        for client_id, value in shapley_values.items():
            explanation += f"- Client {client_id}: Shapley Value = {value}\n"
        explanation += "\nIncentive Allocation:\n"
        for client_id, incentive in incentives.items():
            explanation += f"- Client {client_id}: Incentive = ${incentive:.2f}\n"

        # Descriptive Statistics
        shapley_array = np.array(list(shapley_values.values()))
        explanation += "\nDescriptive Statistics for Shapley Values:\n"
        explanation += f"Mean: {np.mean(shapley_array):.2f}, Median: {np.median(shapley_array):.2f}, Std Dev: {np.std(shapley_array):.2f}\n"

        # Generate and return each plot separately
        plot_buffers = []

        # Bar Chart: Incentive Allocation vs Shapley Values
        fig1, ax1 = plt.subplots()
        ax1.bar(shapley_values.keys(), shapley_values.values(), color='blue', label='Shapley Values')
        ax1.bar(incentives.keys(), incentives.values(), color='green', alpha=0.5, label='Incentives')
        ax1.set_xlabel('Client ID')
        ax1.set_ylabel('Amount')
        ax1.set_title('Incentive Allocation vs Shapley Values')
        ax1.legend()
        buf1 = io.BytesIO()
        plt.savefig(buf1, format='png')
        buf1.seek(0)
        plt.close(fig1)
        plot_buffers.append(buf1)

        # Pie Chart: Contribution Distribution
        fig2, ax2 = plt.subplots()
        ax2.pie(shapley_values.values(), labels=shapley_values.keys(), autopct='%1.1f%%')
        ax2.set_title('Distribution of Contributions')
        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png')
        buf2.seek(0)
        plt.close(fig2)
        plot_buffers.append(buf2)

        # Scatter Plot: Contributions vs Incentives
        fig3, ax3 = plt.subplots()
        ax3.scatter(shapley_values.values(), incentives.values())
        ax3.set_xlabel('Shapley Values (Contributions)')
        ax3.set_ylabel('Incentives')
        ax3.set_title('Contributions vs Incentives')
        buf3 = io.BytesIO()
        plt.savefig(buf3, format='png')
        buf3.seek(0)
        plt.close(fig3)
        plot_buffers.append(buf3)

        return explanation, plot_buffers


    def load_model(self, client_id, round_num, suffix):
            """
            Load a client's model based on the round number and suffix.

            """
            model_path = f'client/localModels/client_{client_id}_model_round_{round_num}{suffix}.pt'
            model_bytes = self.file_handler.retrieve_file(self.training_id, model_path)
            
            if model_bytes:
                model_stream = io.BytesIO(model_bytes)
                model_state = torch.load(model_stream)
            
            model = copy.deepcopy(self.global_model)
            model.load_state_dict(model_state)
            model.to(self.device).eval()
            return model
        
        