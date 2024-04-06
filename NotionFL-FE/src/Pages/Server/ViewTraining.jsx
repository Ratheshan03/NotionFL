import React, { useState, useEffect } from "react";
import axios from "axios";
import { useAuth } from "../../Authcontext";

const TrainingSessionCard = ({ trainingSession }) => {
  const [showLogs, setShowLogs] = useState(false);
  const toggleLogs = () => setShowLogs(!showLogs);

  return (
    <div className="card my-4 p-6 bg-gradient-to-br from-blue-950 via-gray-900 to-black text-white shadow rounded">
      <h3 className="text-xl font-bold mb-2">
        Training ID: {trainingSession.training_id}
      </h3>
      <div className="mb-4">
        <p>Status: {trainingSession.status}</p>
        <p>Started: {new Date(trainingSession.start_time).toLocaleString()}</p>
        {trainingSession.end_time && (
          <p>Ended: {new Date(trainingSession.end_time).toLocaleString()}</p>
        )}
        <p>Initiated by: {trainingSession.initiator.username}</p>
      </div>

      <h4 className="text-lg font-semibold mb-2">Model Configuration:</h4>
      <div className="mb-4">
        <p>Dataset: {trainingSession.config.dataset}</p>
        <p>Model: {trainingSession.config.model}</p>
        <p>Batch Size: {trainingSession.config.batch_size}</p>
        <p>Epochs: {trainingSession.config.epochs}</p>
        {/* Include other important configurations as needed */}
      </div>

      <button
        onClick={toggleLogs}
        className="px-4 py-2 bg-transparent border border-white hover:bg-white hover:text-black rounded transition duration-300"
      >
        {showLogs ? "Hide Details" : "Show Details"}
      </button>

      {showLogs && (
        <div className="whitespace-pre-wrap bg-gray-200 text-black p-4 rounded mt-2">
          {/* Here you can display logs or other details */}
        </div>
      )}
    </div>
  );
};

const ViewTraining = () => {
  const [trainingSessions, setTrainingSessions] = useState([]);
  const { currentUser } = useAuth(); // Use your auth context to get the current user

  useEffect(() => {
    const fetchTrainingSessions = async () => {
      try {
        const userId = currentUser?.user?.id;
        const { data } = await axios.get(
          `http://localhost:5000/training/training_sessions/${userId}`
        );

        console.log(data);
        setTrainingSessions(data);
      } catch (error) {
        console.error("Error fetching training sessions:", error);
      }
    };

    fetchTrainingSessions();
  }, [currentUser]);

  return (
    <div className="p-6 bg-gradient-to-br from-gray-900 via-gray-800 to-black text-white shadow-md rounded-md">
      <h2 className="text-3xl font-bold mb-6">Training Sessions</h2>
      {trainingSessions.length > 0 ? (
        trainingSessions.map((session) => (
          <TrainingSessionCard
            key={session.training_id}
            trainingSession={session}
          />
        ))
      ) : (
        <p>No training sessions found.</p>
      )}
    </div>
  );
};

export default ViewTraining;
