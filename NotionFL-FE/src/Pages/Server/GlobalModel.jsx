import React, { useState, useEffect } from 'react';
import axios from 'axios';

const GlobalModels = () => {
  const [roundNumber, setRoundNumber] = useState('');
  const [globalData, setGlobalData] = useState({
    evaluationResults: null,
    modelComparisonPlot: '',
    shapValuesPlot: '',
    globalModelUrl: ''
  });

  const handleRoundChange = (e) => {
    setRoundNumber(e.target.value);
  };

  const fetchData = async (selectedRound) => {
    try {
      const response = await axios.get(`http://localhost:5000/global_model_data/${selectedRound}`);
      setGlobalData({
        evaluationResults: response.data.evaluationResults,
        modelComparisonPlot: `data:image/png;base64,${response.data.modelComparisonPlot}`,
        shapValuesPlot: `data:image/png;base64,${response.data.shapValuesPlot}`,
        globalModelUrl: response.data.globalModelUrl
      });

      // Save the URL of the global model in local storage
      localStorage.setItem('globalModelUrl', response.data.globalModelUrl);
      
    } catch (error) {
      console.error('Error fetching global model data:', error);
    }
  };

  useEffect(() => {
    if (roundNumber) {
      fetchData(roundNumber);
    }
  }, [roundNumber]);

  const downloadGlobalModel = () => {
    const url = localStorage.getItem('globalModelUrl');
    if (url) {
      window.open(url, '_blank');
    }
  };

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

      {globalData.evaluationResults && (
        <div className="card p-4 shadow-md bg-white rounded-md overflow-scroll">
          <h3 className="text-lg font-semibold">Global Model Evaluation Results</h3>
          <pre>{JSON.stringify(globalData.evaluationResults, null, 2)}</pre>
        </div>
      )}
      {globalData.modelComparisonPlot && (
        <div className="card p-4 shadow-md bg-white">
          <h3 className="text-lg font-semibold">Global and Client Model Comparison</h3>
          <img src={globalData.modelComparisonPlot} alt="Model Comparison Plot" className="w-full h-auto" />
        </div>
      )}
      {globalData.shapValuesPlot && (
        <div className="card p-4 shadow-md bg-white">
          <h3 className="text-lg font-semibold">Global SHAP Values Explanation</h3>
          <img src={globalData.shapValuesPlot} alt="SHAP Values Plot" className="w-full h-auto" />
        </div>
      )}

      <h3 className="text-lg font-semibold">Global Model Download</h3>
      <button 
        className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-700 mt-4" 
        onClick={downloadGlobalModel}
      >
        Download Global Model
      </button>
    </div>
  );
};

export default GlobalModels;
