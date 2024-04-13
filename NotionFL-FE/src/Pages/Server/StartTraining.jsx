import { useState, useEffect } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../../Authcontext";

const StartTraining = () => {
  const navigate = useNavigate();
  const { currentUser } = useAuth();
  // States for each training parameter
  const [batchSize, setBatchSize] = useState("64");
  const [clipThreshold, setClipThreshold] = useState("1.0");
  const [device, setDevice] = useState("cpu");
  const [epochs, setEpochs] = useState("5");
  const [evalEveryNRounds, setEvalEveryNRounds] = useState("1");
  const [flRounds, setFlRounds] = useState("1");
  const [learningRate, setLearningRate] = useState("0.01");
  const [noiseMultiplier, setNoiseMultiplier] = useState("0.1");
  const [numClients, setNumClients] = useState("4");
  const [selectedDataset, setSelectedDataset] = useState("MNIST");
  const [selectedModel, setSelectedModel] = useState("MNISTModel");
  const [configOptions, setConfigOptions] = useState({});

  // Handler functions for input changes
  const handleBatchSizeChange = (event) => setBatchSize(event.target.value);
  const handleClipThresholdChange = (event) =>
    setClipThreshold(event.target.value);
  const handleDeviceChange = (event) => setDevice(event.target.value);
  const handleEpochsChange = (event) => setEpochs(event.target.value);
  const handleEvalEveryNRoundsChange = (event) =>
    setEvalEveryNRounds(event.target.value);
  const handleFlRoundsChange = (event) => setFlRounds(event.target.value);
  const handleLearningRateChange = (event) =>
    setLearningRate(event.target.value);
  const handleNoiseMultiplierChange = (event) =>
    setNoiseMultiplier(event.target.value);
  const handleNumClientsChange = (event) => setNumClients(event.target.value);
  const handleDatasetChange = (event) => setSelectedDataset(event.target.value);
  const handleModelChange = (event) => setSelectedModel(event.target.value);

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
      dataset: selectedDataset,
      model: selectedModel,
      user_id: currentUser?.user?.id,
    };

    console.log("Sending training config:", trainingConfig);

    try {
      const response = await axios.post(
        "http://localhost:5000/training/start_training",
        trainingConfig,
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      if (response.data.status === "Training started") {
        localStorage.setItem("training_id", response.data.training_id);
        alert("Training has started successfully!");
        navigate("/serverView/view-training");
      } else {
        alert("Error starting training");
      }
    } catch (error) {
      console.error("Error submitting training config:", error);
      alert("Failed to start training");
    }
  };

  // Example of fetching the training configuration in a React component
  useEffect(() => {
    const fetchTrainingConfig = async () => {
      try {
        const response = await axios.get(
          "http://localhost:5000/training/get_training_config"
        );
        if (response.status === 200) {
          setConfigOptions(response.data);
        }
      } catch (error) {
        console.error("Error fetching training configuration:", error);
      }
    };

    fetchTrainingConfig();
  }, []);

  // Tailwind CSS classes
  // Updated Tailwind CSS classes for improved styling
  const selectClass = `
  block w-full mt-1 bg-gray-700 text-white border border-gray-600 rounded-md shadow-sm 
  focus:border-blue-500 focus:ring focus:ring-blue-200 focus:ring-opacity-50 
  transition duration-300 ease-in-out transform hover:scale-105
`;

  const labelClass = `
  block text-sm font-medium text-white
`;

  const buttonClass = `
  mt-6 w-full flex justify-center py-3 px-4 border border-transparent text-lg font-medium 
  rounded-md text-white bg-gradient-to-r from-blue-800 to-blue-900 hover:bg-gradient-to-l 
  focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 shadow-lg
`;

  const formClass = `
  space-y-6 w-full md:w-3/4 lg:w-1/2 mx-auto
`;
  return (
    <div className="max-w-4xl mx-auto my-10 p-8 bg-gradient-to-br from-blue-950 via-gray-900 to-black shadow-xl rounded-xl">
      <h2 className="text-3xl font-semibold text-center text-white mb-8">
        Start Federated Learning Training
      </h2>
      <form onSubmit={handleSubmit} className={formClass}>
        <div>
          <label className={labelClass}>Dataset</label>
          <select
            value={selectedDataset}
            onChange={handleDatasetChange}
            className={selectClass}
          >
            {configOptions.datasets?.map((dataset, idx) => (
              <option key={idx} value={dataset}>
                {dataset}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className={labelClass}>Model</label>
          <select
            value={selectedModel}
            onChange={handleModelChange}
            className={selectClass}
          >
            {configOptions.models?.map((model, idx) => (
              <option key={idx} value={model}>
                {model}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className={labelClass}>Batch Size</label>
          <select
            value={batchSize}
            onChange={handleBatchSizeChange}
            className={selectClass}
          >
            {configOptions.trainingSettings?.batchSize?.map(
              (batchSize, idx) => (
                <option key={idx} value={batchSize}>
                  {batchSize}
                </option>
              )
            )}
          </select>
        </div>
        <div>
          <label className={labelClass}>Training Epochs</label>
          <select
            value={epochs}
            onChange={handleEpochsChange}
            className={selectClass}
          >
            {configOptions.trainingSettings?.epochs?.map((epochs, idx) => (
              <option key={idx} value={epochs}>
                {epochs}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className={labelClass}>Model Runtime Device</label>
          <select
            value={device}
            onChange={handleDeviceChange}
            className={selectClass}
          >
            {configOptions.trainingSettings?.device?.map((device, idx) => (
              <option key={idx} value={device}>
                {device}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className={labelClass}>Model Learning Rate</label>
          <select
            value={learningRate}
            onChange={handleLearningRateChange}
            className={selectClass}
          >
            {configOptions.trainingSettings?.learningRate?.map(
              (learningRate, idx) => (
                <option key={idx} value={learningRate}>
                  {learningRate}
                </option>
              )
            )}
          </select>
        </div>
        <div>
          <label className={labelClass}>Federated Learning Rounds</label>
          <select
            value={flRounds}
            onChange={handleFlRoundsChange}
            className={selectClass}
          >
            {configOptions.trainingSettings?.flRounds?.map((flRounds, idx) => (
              <option key={idx} value={flRounds}>
                {flRounds}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className={labelClass}>Number of Clients</label>
          <select
            value={numClients}
            onChange={handleNumClientsChange}
            className={selectClass}
          >
            {configOptions.clients?.numClients?.map((numClients, idx) => (
              <option key={idx} value={numClients}>
                {numClients}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className={labelClass}>Threshold - Privacy mechanism</label>
          <select
            value={clipThreshold}
            onChange={handleClipThresholdChange}
            className={selectClass}
          >
            {configOptions.privacySettings?.clipThreshold?.map(
              (clipThreshold, idx) => (
                <option key={idx} value={clipThreshold}>
                  {clipThreshold}
                </option>
              )
            )}
          </select>
        </div>

        <div>
          <label className={labelClass}>
            Noise Multiplier - Privacy Mechnism
          </label>
          <select
            value={noiseMultiplier}
            onChange={handleNoiseMultiplierChange}
            className={selectClass}
          >
            {configOptions.privacySettings?.noiseMultiplier?.map(
              (noiseMultiplier, idx) => (
                <option key={idx} value={noiseMultiplier}>
                  {noiseMultiplier}
                </option>
              )
            )}
          </select>
        </div>
        <div>
          <label className={labelClass}>Evaluation Rounds</label>
          <select
            value={evalEveryNRounds}
            onChange={handleEvalEveryNRoundsChange}
            className={selectClass}
          >
            {configOptions.evaluationSettings?.evalEveryNRounds?.map(
              (evalEveryNRounds, idx) => (
                <option key={idx} value={evalEveryNRounds}>
                  {evalEveryNRounds}
                </option>
              )
            )}
          </select>
        </div>

        <button type="submit" className={buttonClass}>
          Start Training
        </button>
      </form>
    </div>
  );
};

export default StartTraining;
