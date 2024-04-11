import React, { useState, useEffect } from "react";
import axios from "axios";
import { useAuth } from "../../Authcontext";

const DataCard = ({ title, content, type }) => {
  const renderContent = () => {
    if (type === "json") {
      // JSON data will be formatted in a scrollable preformatted block
      return (
        <pre className="text-white max-w-full overflow-x-auto">
          {JSON.stringify(JSON.parse(content), null, 2)}
        </pre>
      );
    } else if (type === "image") {
      // Images will scale to fit their container
      return (
        <img
          src={`data:image/png;base64,${content}`}
          alt="Data Plot"
          className="w-full h-auto"
        />
      );
    } else if (type === "text") {
      // Text content will be in a paragraph tag, normal overflow applies
      return <p className="text-white">{content}</p>;
    }
    // Other content will just be returned as is
    return content;
  };

  return (
    <div className="bg-gray-700 p-4 rounded-lg shadow-sm mb-4 overflow-hidden">
      <h3 className="text-lg font-semibold text-white mb-2">{title}</h3>
      <div className="max-w-full">{renderContent()}</div>
    </div>
  );
};

const ViewTraining = () => {
  const [trainingSessions, setTrainingSessions] = useState([]);
  const [selectedTrainingSession, setSelectedTrainingSession] = useState(null);
  const [selectedRoundNum, setSelectedRoundNum] = useState("");
  const [trainingData, setTrainingData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const { currentUser } = useAuth();

  useEffect(() => {
    const fetchTrainingSessions = async () => {
      try {
        const userId = currentUser?.user?.id;
        const { data } = await axios.get(
          `http://localhost:5000/client/get_training_sessions/${userId}`
        );
        setTrainingSessions(data);
      } catch (error) {
        console.error("Error fetching training sessions:", error);
      }
    };

    fetchTrainingSessions();
  }, [currentUser]);

  useEffect(() => {
    const fetchTrainingData = async () => {
      if (selectedTrainingSession && selectedRoundNum) {
        setIsLoading(true);
        try {
          const { training_id, client_id } = selectedTrainingSession;
          const response = await axios.get(
            `http://localhost:5000/client/get_training_data/${training_id}/${client_id}/${selectedRoundNum}`
          );
          setTrainingData(response.data);
        } catch (error) {
          console.error("Error fetching training data:", error);
        } finally {
          setIsLoading(false);
        }
      }
    };

    fetchTrainingData();
  }, [selectedTrainingSession, selectedRoundNum]);

  const handleTrainingSessionChange = (e) => {
    const trainingId = e.target.value;
    const session = trainingSessions.find((s) => s.training_id === trainingId);
    setSelectedTrainingSession(session || null);
    setSelectedRoundNum("");
  };

  const handleRoundChange = (e) => {
    setSelectedRoundNum(e.target.value);
  };

  return (
    <div className="max-w-4xl mx-auto my-10 p-6 bg-gray-800 text-white shadow-lg rounded-lg overflow-hidden">
      <h2 className="text-2xl font-semibold mb-6">
        Client Training Data Viewer
      </h2>
      <div className="flex space-x-4 mb-6">
        <select
          onChange={handleTrainingSessionChange}
          className="flex-1 p-2 border border-gray-500 bg-gray-700 rounded-md"
          value={selectedTrainingSession?.training_id || ""}
        >
          <option value="">Select Training Session</option>
          {trainingSessions.map((session) => (
            <option key={session.training_id} value={session.training_id}>
              Training {session.training_id}
            </option>
          ))}
        </select>

        <select
          onChange={handleRoundChange}
          className="flex-1 p-2 border border-gray-500 bg-gray-700 rounded-md"
          value={selectedRoundNum}
        >
          <option value="">Select Round Number</option>
          <option value="0">Round 1</option>
          <option value="1">Round 2</option>
          <option value="2">Round 3</option>
          {/* Add more rounds as needed */}
        </select>
      </div>

      {isLoading ? (
        <p>Loading...</p>
      ) : (
        trainingData && (
          <>
            {trainingData.training_logs && (
              <DataCard
                title="Training Logs"
                content={trainingData.training_logs}
                type="json"
              />
            )}
            {trainingData.eval_logs && (
              <DataCard
                title="Evaluation Logs"
                content={trainingData.eval_logs}
                type="json"
              />
            )}
            {trainingData.eval_plot && (
              <DataCard
                title="Evaluation Plot"
                content={trainingData.eval_plot}
                type="image"
              />
            )}
            {trainingData.eval_text && (
              <DataCard
                title="Evaluation Text"
                content={trainingData.eval_text}
                type="text"
              />
            )}
          </>
        )
      )}
    </div>
  );
};

export default ViewTraining;
