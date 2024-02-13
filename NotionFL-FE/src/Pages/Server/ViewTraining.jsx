import React, { useState, useEffect } from 'react';
import axios from 'axios';

const ViewTraining = () => {
  const [trainingStatus, setTrainingStatus] = useState('Unknown');
  const [trainingLogs, setTrainingLogs] = useState('');
  const trainingId = localStorage.getItem('training_id'); // Retrieve the training ID from local storage
  const intervalId = null;

  // Function to check training status and fetch logs
  const checkTrainingStatus = async () => {
    try {
      const response = await axios.get(`http://localhost:5000/training_status/${trainingId}`);
      const status = response.data.status;
      setTrainingStatus(status);
      setTrainingLogs(response.data.logs); // Update logs

      // If training is completed or failed, stop checking status
      if (status === 'Completed' || status === 'Failed') {
        clearInterval(intervalId);
      }
    } catch (error) {
      console.error('Error fetching training status:', error);
      setTrainingStatus('Error fetching status');
    }
  };

  useEffect(() => {
    // Set up an interval to check training status every minute
    let intervalId = setInterval(() => {
      checkTrainingStatus();
    }, 60000); // 60000 ms = 1 minute

    // Clear the interval when component unmounts
    return () => clearInterval(intervalId);
  }, []);

  return (
    <div className="p-6 bg-white shadow-md rounded-md">
      <h2 className="text-2xl font-semibold text-gray-800 mb-6">Training Status</h2>
      <p>Training ID: {trainingId}</p>
      <p>Status: {trainingStatus}</p>
      {trainingLogs && (
        <>
          <h3 className="mt-4 text-xl font-semibold">Training Logs:</h3>
          <pre className="whitespace-pre-wrap bg-gray-100 p-4 rounded-md">{trainingLogs}</pre>
        </>
      )}
      <button
        className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-700 mt-4"
        onClick={checkTrainingStatus}
      >
        Check Training Status
      </button>
    </div>
  );
};

export default ViewTraining;
