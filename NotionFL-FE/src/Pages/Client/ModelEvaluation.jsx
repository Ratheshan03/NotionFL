import React, { useState, useEffect } from 'react';
import axios from 'axios';

const ClientView = () => {
  const [selectedClientId, setSelectedClientId] = useState('');
  const [selectedRoundNumber, setSelectedRoundNumber] = useState('');
  const [clientData, setClientData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      if (!selectedClientId || !selectedRoundNumber) return;
      
      setIsLoading(true);
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
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, [selectedClientId, selectedRoundNumber]);

  const renderDataCard = (title, content) => (
    <div className="card my-6 bg-gray-100 p-6 shadow-md rounded-md overflow-auto">
      <h2 className="text-xl font-semibold text-gray-800 mb-4">{title}</h2>
      {content}
    </div>
  );

  return (
    <div className="max-w-4xl mx-auto my-10 p-6 bg-white shadow-md rounded-md">
      <h2 className="text-2xl font-semibold text-gray-800 mb-6">Client model Evaluation</h2>
      
      <div className="flex space-x-4 mb-8">
        <div className="flex-1">
          <label className="block text-sm font-medium text-gray-700 mb-2">Client ID</label>
          <select className="block w-full border border-gray-300 rounded-md p-2" value={selectedClientId} onChange={(e) => setSelectedClientId(e.target.value)}>
            <option value="">Select Client</option>
            {/* Map through your client IDs */}
            <option value="0">Client 0</option>
            <option value="1">Client 1</option>
            {/* Add more options */}
          </select>
        </div>
        
        <div className="flex-1">
          <label className="block text-sm font-medium text-gray-700 mb-2">Round Number</label>
          <select className="block w-full border border-gray-300 rounded-md p-2" value={selectedRoundNumber} onChange={(e) => setSelectedRoundNumber(e.target.value)}>
            <option value="">Select Round</option>
            {/* Map through your round numbers */}
            <option value="1">Round 1</option>
            <option value="2">Round 2</option>
            <option value="3">Round 3</option>
            {/* Add more options */}
          </select>
        </div>
      </div>

      {isLoading ? <p>Loading data...</p> : (
        <div>
          {clientData?.evaluationLogs && renderDataCard("Evaluation Logs", <pre>{JSON.stringify(clientData.evaluationLogs, null, 2)}</pre>)}
          {clientData?.modelEvaluation && renderDataCard("Model Evaluation", <img src={clientData.modelEvaluation} alt="Model Evaluation" />)}
          {clientData?.globalModelComparison && renderDataCard("Global Model Comparison", <img src={clientData.globalModelComparison} alt="Global Model Comparison" />)}
          {clientData?.contributionShapleyValues && renderDataCard("Contribution Shapley Values", <pre>{JSON.stringify(clientData.contributionShapleyValues, null, 2)}</pre>)}
          {clientData?.contributionPlot && renderDataCard("Contribution Plot", <img src={clientData.contributionPlot} alt="Contribution Plot" />)}
        </div>
      )}
    </div>
  );
};

export default ClientView;
