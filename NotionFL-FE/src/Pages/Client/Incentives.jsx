import React, { useState, useEffect } from 'react';
import axios from 'axios';

const Incentives = () => {
  const [selectedRound, setSelectedRound] = useState('');
  const [data, setData] = useState({
    shapleyValues: null,
    contributionPlot: null,
    incentives: null,
    incentivesPlot: null,
    incentivesExplanation: null
  });

  // Example round numbers, replace with actual data if needed
  const roundNumbers = [0, 1, 2, 3];

  useEffect(() => {
    if (selectedRound) {
      const fetchData = async () => {
        try {
          const response = await axios.get(`http://localhost:5000/client_incentives/${selectedRound}`);
          const {
            shapleyValues,
            contributionPlot,
            incentives,
            incentivesPlot,
            incentivesExplanation
          } = response.data;

          setData({
            shapleyValues: shapleyValues,
            contributionPlot: contributionPlot ? `data:image/png;base64,${contributionPlot}` : null,
            incentives: incentives,
            incentivesPlot: incentivesPlot ? `data:image/png;base64,${incentivesPlot}` : null,
            incentivesExplanation: incentivesExplanation
          });
        } catch (error) {
          console.error('Error fetching incentives data:', error);
        }
      };
      
      fetchData();
    }
  }, [selectedRound]);

  return (
    <div className="max-w-4xl mx-auto my-10 p-6 bg-white shadow-md rounded-md">
      <h2 className="text-2xl font-semibold text-gray-800 mb-6">Client Incentives</h2>

      <div className="mb-4">
        <label htmlFor="round-select" className="block text-sm font-medium text-gray-700">Select Round:</label>
        <select
          id="round-select"
          value={selectedRound}
          onChange={(e) => setSelectedRound(e.target.value)}
          className="mt-1 block w-full py-2 px-3 border border-gray-300 bg-white rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
        >
          <option value="">--Choose a round--</option>
          {roundNumbers.map((round) => (
            <option key={round} value={round}>Round {round}</option>
          ))}
        </select>
      </div>

      {data.shapleyValues && (
        <div className="card mb-4 p-4 bg-gray-100 rounded-md">
          <h3 className="text-lg font-semibold mb-2">Shapley Values</h3>
          <pre>{JSON.stringify(data.shapleyValues, null, 2)}</pre>
        </div>
      )}

      {data.contributionPlot && (
        <div className="card mb-4">
          <h3 className="text-lg font-semibold mb-2">Client Contribution Plot</h3>
          <img src={data.contributionPlot} alt="Contribution Plot" className="w-full h-auto" />
        </div>
      )}

      {data.incentives && (
        <div className="card mb-4 p-4 bg-gray-100 rounded-md">
          <h3 className="text-lg font-semibold mb-2">Incentives</h3>
          <pre>{JSON.stringify(data.incentives, null, 2)}</pre>
        </div>
      )}

      {data.incentivesPlot && (
        <div className="card mb-4">
          <h3 className="text-lg font-semibold mb-2">Incentives Plot</h3>
          <img src={data.incentivesPlot} alt="Incentives Plot" className="w-full h-auto" />
        </div>
      )}

      {data.incentivesExplanation && (
        <div className="card mb-4 p-4 bg-gray-100 rounded-md">
          <h3 className="text-lg font-semibold mb-2">Incentives Explanation</h3>
          <p>{data.incentivesExplanation}</p>
        </div>
      )}
    </div>
  );
};

export default Incentives;
