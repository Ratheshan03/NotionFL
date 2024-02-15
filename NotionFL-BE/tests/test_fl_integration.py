import subprocess
import time
import requests

def run_fl_process():
    # Start the FL process
    process = subprocess.Popen(['python', 'main.py'])
    return process

def test_api_endpoints():
    # Adjust these endpoints according to your API
    response = requests.get('http://localhost:5000/get_global_evaluation/1')
    assert response.status_code == 200, "Global evaluation endpoint failed"
    response.json()  # Check if the response is a valid JSON

    # Add more API endpoint tests
    response2 = requests.get('http://localhost:5000/get_client_shap_plot/0/1')
    assert response2.status_code == 200, "Client evaluation endpoint failed"
    response2.json()  # Check if the response is a valid JSON


def main():
    process = run_fl_process()
    time.sleep(1000)  # Adjust based on your training duration

    test_api_endpoints()

    # Terminate the FL process
    process.terminate()
    process.wait()

    print("Integration test passed")

if __name__ == "__main__":
    main()
