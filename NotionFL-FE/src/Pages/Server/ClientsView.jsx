import React, { useState, useEffect } from 'react';
import axios from 'axios';

const ClientView = () => {
  const [clients, setClients] = useState([]); // This would be fetched from the server or a predefined list
  const [rounds, setRounds] = useState([]); // Same as above
  const [selectedClientId, setSelectedClientId] = useState('');
  const [selectedRoundNumber, setSelectedRoundNumber] = useState('');
  const [clientData, setClientData] = useState({
    evaluationLogs: null,
    modelEvaluation: null,
    globalModelComparison: null,
    contributionShapleyValues: null,
    contributionPlot: null,
  });

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get(`http://localhost:5000/client_data/${selectedClientId}/${selectedRoundNumber}`);
        setClientData({
          ...response.data,
          modelEvaluation: response.data.modelEvaluation ? `data:image/png;base64,${response.data.modelEvaluation}` : null,
          globalModelComparison: response.data.globalModelComparison ? `data:image/png;base64,${response.data.globalModelComparison}` : null,
          contributionPlot: response.data.contributionPlot ? `data:image/png;base64,${response.data.contributionPlot}` : null,
        });
      } catch (error) {
        console.error('Error fetching client data:', error);
      }
    };

    
      fetchData();
  }, [selectedClientId, selectedRoundNumber]);

  return (
    <div className="max-w-2xl mx-auto my-10 p-6 bg-white shadow-md rounded-md">
    <h2 className="text-2xl font-semibold text-gray-800 mb-6">Federated Learning Training Client Stats </h2>
      
    <div className='flex space-x-3 align-middle justify-center'>

    <label className="block text-sm font-medium text-gray-700">Client Id</label>
      <select value={selectedClientId} onChange={(e) => setSelectedClientId(e.target.value)}>
        {/* Map through your client IDs to create options */}
        <option value="">Select Client ID</option>
        <option value="0">0</option>
        <option value="1">1</option>
        <option value="2">2</option>
        {/* Add client options here */}
      </select>

      <label className="block text-sm font-medium text-gray-700">FL Round Number</label>
      <select value={selectedRoundNumber} onChange={(e) => setSelectedRoundNumber(e.target.value)}>
        {/* Map through your round numbers to create options */}
        <option value="">Select Round Number</option>
        <option value="0">0</option>
        <option value="1">1</option>
        <option value="2">2</option>
        {/* Add round options here */}
      </select>
      </div>
    
    <div >

      {clientData.evaluationLogs && (
        <div className="card my-6 max-w-2xl mx-auto bg-gray-300 p-6 shadow-md rounded-md overflow-scroll">
          {/* Assuming evaluationLogs is a JSON string */}
          <h2 className="text-xl font-semibold text-gray-800 mb-6">Evaluation Logs </h2>
          <pre>{JSON.stringify(clientData.evaluationLogs, null, 2)}</pre>
        </div>
      )}
      {clientData.modelEvaluation && (
        <div className="card my-6 max-w-2xl mx-auto bg-gray-300 p-6 shadow-md rounded-md">
          <h2 className="text-xl font-semibold text-gray-800 mb-6">Client model Shap value evaluation </h2>
          <img src={clientData.modelEvaluation} alt="Model Evaluation" />
        </div>
      )}
      {clientData.globalModelComparison && (
        <div className="card my-6 max-w-2xl mx-auto bg-gray-300 p-6 shadow-md rounded-md">
          <h2 className="text-xl font-semibold text-gray-800 mb-6">Client and Global models Shap value evaluation </h2>
          <img src={clientData.globalModelComparison} alt="Global Model Comparison" />
        </div>
      )}
      {clientData.contributionShapleyValues && (
        <div className="card my-6 max-w-2xl mx-auto bg-gray-300 p-6 shadow-md rounded-md">
          <h2 className="text-xl font-semibold text-gray-800 mb-6">Client Contribution Evaluation - SHAP value </h2>
          <pre>{JSON.stringify(clientData.contributionShapleyValues, null, 2)}</pre>
        </div>
      )}
      {clientData.contributionPlot && (
        <div className="card my-6 max-w-2xl mx-auto bg-gray-300 p-6 shadow-md rounded-md">
          <h2 className="text-xl font-semibold text-gray-800 mb-6">Client models Contribution Evaluation (Impact on Global Model)</h2>
          <img src={clientData.contributionPlot} alt="Contribution Plot" />
        </div>
      )}
    </div>
    </div>
  );
};

export default ClientView;
