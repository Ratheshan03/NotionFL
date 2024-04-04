import json
import io
import matplotlib.pyplot as plt


def allocate_incentives(shapley_values):
    # Example Pool money for training allocation
    total_incentive_pool = 10000

    total_shapley = sum(shapley_values.values())
    if total_shapley == 0:
        incentives = {str(client_id): 0 for client_id in shapley_values.keys()}
         
    else:
        incentives = {str(client_id): (value / total_shapley) * total_incentive_pool 
                for client_id, value in shapley_values.items()}

    # Generate incentive plot
    buf = io.BytesIO()
    plt.figure(figsize=(10, 6))
    plt.bar(incentives.keys(), incentives.values(), color='green')
    plt.xlabel('Client ID')
    plt.ylabel('Incentive Amount')
    plt.title('Incentive Allocation Based on Shapley Values')
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return incentives, buf
