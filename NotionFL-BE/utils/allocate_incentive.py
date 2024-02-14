import json
import os
import matplotlib.pyplot as plt

def read_shapley_values(file_path):
    with open(file_path, 'r') as file:
        shapley_values = json.load(file)
    return shapley_values

def calculate_incentives(shapley_values, total_incentive_pool):
    total_shapley = sum(shapley_values.values())
    return {client_id: (value / total_shapley) * total_incentive_pool for client_id, value in shapley_values.items()}

def save_incentives_log(incentives, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        json.dump(incentives, file, indent=4)

def create_and_save_incentive_plot(incentives, plot_file_path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(plot_file_path), exist_ok=True)

    # Create a bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(incentives.keys(), incentives.values(), color='blue')
    plt.xlabel('Client ID')
    plt.ylabel('Incentive Amount')
    plt.title('Incentive Allocation Based on Shapley Values')
    
    # Save the plot
    plt.savefig(plot_file_path)

def allocate_and_save_incentives(round_num):
    shapley_values_file = f'output/data_collector/client/contribution/client_shapley_values_round_{round_num}.json'
    incentives_log_file = f'output/data_collector/client/contribution/client_incentives_round_{round_num}.json'
    incentive_plot_file = f'output/data_collector/client/contribution/incentive_plot_round_{round_num}.png'
    total_incentive_pool = 10000

    shapley_values = read_shapley_values(shapley_values_file)
    incentives = calculate_incentives(shapley_values, total_incentive_pool)
    save_incentives_log(incentives, incentives_log_file)
    create_and_save_incentive_plot(incentives, incentive_plot_file)

    print(f"Incentives allocated: {incentives}")

