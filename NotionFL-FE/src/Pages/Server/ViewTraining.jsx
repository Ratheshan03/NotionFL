import React, { useState, useEffect } from "react";
import axios from "axios";

const TrainingSessionCard = ({ sessionId, sessionData }) => {
  const [showLogs, setShowLogs] = useState(false);

  const toggleLogs = () => setShowLogs(!showLogs);

  return (
    <div className="card my-4 p-4 bg-gray-100 shadow rounded">
      <h3 className="text-lg font-semibold">Training ID: {sessionId}</h3>
      <p>Status: {sessionData.status}</p>
      <p>Started: {new Date(sessionData.start_time).toLocaleString()}</p>
      {sessionData.end_time && (
        <p>Ended: {new Date(sessionData.end_time).toLocaleString()}</p>
      )}
      <button
        onClick={toggleLogs}
        className="px-4 py-2 bg-blue-500 text-white rounded mt-2"
      >
        {showLogs ? "Hide Logs" : "Show Logs"}
      </button>
      {showLogs && (
        <pre className="whitespace-pre-wrap bg-gray-200 p-4 rounded mt-2">
          {sessionData.logs}
        </pre>
      )}
    </div>
  );
};

const ViewTraining = () => {
  const [trainingSessions, setTrainingSessions] = useState({});

  useEffect(() => {
    const fetchTrainingStatus = async () => {
      try {
        const { data } = await axios.get(
          "http://localhost:5000/training_status"
        );
        setTrainingSessions(data);
      } catch (error) {
        console.error("Error fetching training statuses:", error);
      }
    };

    fetchTrainingStatus();
  }, []);

  return (
    <div className="p-6 bg-white shadow-md rounded-md">
      <h2 className="text-2xl font-semibold text-gray-800 mb-6">
        Training Sessions
      </h2>
      {Object.entries(trainingSessions).map(([sessionId, sessionData]) => (
        <TrainingSessionCard
          key={sessionId}
          sessionId={sessionId}
          sessionData={sessionData}
        />
      ))}
    </div>
  );
};

export default ViewTraining;
