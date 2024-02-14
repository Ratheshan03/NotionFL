// src/components/ClientView.jsx

import React, { useState, useEffect } from 'react';
import axios from 'axios';

const ClientView = () => {
  const [selectedClientId, setSelectedClientId] = useState('');
  const [clientData, setClientData] = useState({
    trainingLogs: null,
    trainingStatus: null,
  });



  const handleClientChange = async (event) => {
    const clientId = event.target.value;
    setSelectedClientId(clientId);
    if (clientId) {
      try {
        const response = await axios.get(`http://localhost:5000/client_full_data/${clientId}`);
        setClientData(response.data);

      } catch (error) {
        console.error('Error fetching data for client:', clientId, error);
      }
    }
  };


  const getStatusStyle = (status) => {
    switch (status) {
      case 'Completed':
        return 'text-green-500';
      case 'Failed':
        return 'text-red-500';
      case 'Ongoing':
        return 'text-yellow-500';
      default:
        return 'text-gray-500';
    }
  };

  return (
    <div className="p-6 bg-white shadow-md rounded-md">
      <h2 className="text-2xl font-semibold text-gray-800 mb-6">Client Data</h2>
      
      <select onChange={handleClientChange} className="mb-4 p-2 border border-gray-300 rounded-md">
        <option value="">Select Client</option>
        {/* Replace with dynamic client IDs */}
        {[0, 1, 2, 3].map((id) => (
          <option key={id} value={id}>Client {id}</option>
        ))}
      </select>

      {selectedClientId && (
        <>
          <div className={`status-indicator ${getStatusStyle(clientData.trainingStatus)}`}>
            Status: {clientData.trainingStatus}
          </div>
          <div>

          </div>
          
          
          <div className="logs-card mt-4 p-4 bg-gray-100 max-h-64 overflow-auto rounded-md">
            <pre>{clientData.trainingLogs}</pre>
          </div>
        </>
      )}
    </div>
  );
};

export default ClientView;
