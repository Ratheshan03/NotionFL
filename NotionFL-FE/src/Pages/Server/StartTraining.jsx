import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const StartTraining = () => {
  const navigate = useNavigate();
  // States for each training parameter
  const [batchSize, setBatchSize] = useState('64');
  const [clipThreshold, setClipThreshold] = useState('1.0');
  const [device, setDevice] = useState('cpu');
  const [epochs, setEpochs] = useState('5');
  const [evalEveryNRounds, setEvalEveryNRounds] = useState('1');
  const [flRounds, setFlRounds] = useState('1');
  const [learningRate, setLearningRate] = useState('0.01');
  const [noiseMultiplier, setNoiseMultiplier] = useState('0.1');
  const [numClients, setNumClients] = useState('4');

  // Handler functions for input changes
  const handleBatchSizeChange = (event) => setBatchSize(event.target.value);
  const handleClipThresholdChange = (event) => setClipThreshold(event.target.value);
  const handleDeviceChange = (event) => setDevice(event.target.value);
  const handleEpochsChange = (event) => setEpochs(event.target.value);
  const handleEvalEveryNRoundsChange = (event) => setEvalEveryNRounds(event.target.value);
  const handleFlRoundsChange = (event) => setFlRounds(event.target.value);
  const handleLearningRateChange = (event) => setLearningRate(event.target.value);
  const handleNoiseMultiplierChange = (event) => setNoiseMultiplier(event.target.value);
  const handleNumClientsChange = (event) => setNumClients(event.target.value);

  const handleSubmit = async (event) => {
    event.preventDefault();
    const trainingConfig = {
      batch_size: batchSize,
      clip_threshold: clipThreshold,
      device: device,
      epochs: epochs,
      eval_every_n_rounds: evalEveryNRounds,
      fl_rounds: flRounds,
      learning_rate: learningRate,
      noise_multiplier: noiseMultiplier,
      num_clients: numClients,
    };

  // Log the config to the console to see what's being sent
  console.log('Sending training config:', trainingConfig);

  const API_ENDPOINT = 'http://localhost:5000/start_training';

  try {
    const response = await axios.post(API_ENDPOINT, trainingConfig, {
      headers: {
        'Content-Type': 'application/json'
      }
    });

    // Check if training has started successfully
    if (response.data.status === 'Training started') {
      // Store training_id in local storage
      localStorage.setItem('training_id', response.data.training_id);

      // Show confirmation and redirect to view training page
      alert('Training has started successfully!');
      // Redirect to the training view page
      navigate('/server/view-training');
    } else {
      // Handle other status messages
      alert('Error starting training');
    }
  } catch (error) {
    console.error('Error submitting training config:', error);
    alert('Failed to start training');
  }
  };

  // Form JSX
  return (
    <div className="max-w-2xl mx-auto my-10 p-6 bg-white shadow-md rounded-md">
    <h2 className="text-2xl font-semibold text-gray-800 mb-6">Start Federated Learning Training</h2>
    <form onSubmit={handleSubmit} className="space-y-4">
    <div>
        <label className="block text-sm font-medium text-gray-700">Batch Size</label>
      <input
        type="text"
        name="batch_size"
        value={batchSize}
        onChange={handleBatchSizeChange}
        className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2"
      />
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-700">Training Epochs</label>
      <input
        type="text"
        name="epochs"
        value={epochs}
        onChange={handleEpochsChange}
        className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2"
      />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700">Device - cpu or cuda</label>
      <input
        type="text"
        name="device"
        value={device}
        onChange={handleDeviceChange}
        className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2"
      />
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-700">Learning Rates</label>
      <input
        type="text"
        name="learning_rate"
        value={learningRate}
        onChange={handleLearningRateChange}
        className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2"
      />
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-700">Federated Learning rounds</label>
      <input
        type="text"
        name="fl_rounds"
        value={flRounds}
        onChange={handleFlRoundsChange}
        className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2"
      />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700">Number of Clients</label>
      <input

        type="text"
        name="num_clients"
        value={numClients}
        onChange={handleNumClientsChange}
        className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2"
      />
      </div>


      <div>
        <label className="block text-sm font-medium text-gray-700">Threshold - Privacy mechanism</label>
      <input
        type="text"
        name="clip_threshold"
        value={clipThreshold}
        onChange={handleClipThresholdChange}
        className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2"
      />
      </div>
      
      <div>
        <label className="block text-sm font-medium text-gray-700">Noise Multiplier - Privacy Mechnism</label>
      <input
        type="text"
        name="noise_multiplier"
        value={noiseMultiplier}
        onChange={handleNoiseMultiplierChange}
        className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2"
      />
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-700">Evaluation rounds</label>
      <input
        type="text"
        name="eval_every_n_rounds"
        value={evalEveryNRounds}
        onChange={handleEvalEveryNRoundsChange}
        className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2"
      />
      </div>
      
      <button type="submit" className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-700">Start Training</button>
    </form>
  </div>
  );
};

export default StartTraining;
