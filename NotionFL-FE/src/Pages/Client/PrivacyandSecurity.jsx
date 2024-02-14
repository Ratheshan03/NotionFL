import React, { useState, useEffect } from 'react';
import axios from 'axios';

const PrivacyAndSecurity = () => {
  const [roundNumber, setRoundNumber] = useState('');
  const [clientNumber, setClientNumber] = useState('');
  const [privacyData, setPrivacyData] = useState({
    dpExplanation: '',
    dpUsageImage: '',
    secureAggregationPlot: '',
  });

  const handleRoundChange = (e) => {
    setRoundNumber(e.target.value);
    setClientNumber(e.target.value);
  };

  const fetchData = async (selectedRound, selectedClientNumber) => {
    try {
      // Adjust the endpoint URL and parameter as needed
      const response = await axios.get(`http://localhost:5000/privacy_data/${selectedClientNumber}/${selectedRound}`);
      setPrivacyData({
        dpExplanation: response.data.dpExplanation,
        dpUsageImage: `data:image/png;base64,${response.data.dpUsageImage}`,
        secureAggregationPlot: `data:image/png;base64,${response.data.secureAggregationPlot}`,
      });
    } catch (error) {
      console.error('Error fetching privacy and security data:', error);
    }
  };

  useEffect(() => {
    if (roundNumber, clientNumber) {
      fetchData(roundNumber, clientNumber);
    }
  }, [roundNumber, clientNumber]);

  return (
    <div className="space-y-4">
      <div className="mb-4">
        <label htmlFor="roundNumber" className="block text-sm font-medium text-gray-700">Select Round Number</label>
        <select id="roundNumber" value={roundNumber} onChange={handleRoundChange} className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2">
        <option value="">Select Round Number</option>
        <option value="0">0</option>
        <option value="1">1</option>
        <option value="2">2</option>
        </select>
      </div>
      <div className="mb-4">
        <label htmlFor="roundNumber" className="block text-sm font-medium text-gray-700">Select Client Number</label>
        <select id="roundNumber" value={roundNumber} onChange={handleRoundChange} className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2">
        <option value="">Select Client ID</option>
        <option value="0">0</option>
        <option value="1">1</option>
        <option value="2">2</option>
        </select>
      </div>

      {privacyData.dpExplanation && (
        <div className="card p-4 shadow-md bg-white">
          <h3 className="text-lg font-semibold">Differential Privacy Explanation</h3>
          <p>{privacyData.dpExplanation}</p>
        </div>
      )}
      {privacyData.dpUsageImage && (
        <div className="card p-4 shadow-md bg-white">
          <h3 className="text-lg font-semibold">DP Usage Visualization</h3>
          <img src={privacyData.dpUsageImage} alt="Differential Privacy Usage" className="w-full h-auto" />
        </div>
      )}
      {privacyData.secureAggregationPlot && (
        <div className="card p-4 shadow-md bg-white">
          <h3 className="text-lg font-semibold">Secure Aggregation Comparison Plot</h3>
          <img src={privacyData.secureAggregationPlot} alt="Secure Aggregation Comparison" className="w-full h-auto" />
        </div>
      )}
    </div>
  );
};

export default PrivacyAndSecurity;
