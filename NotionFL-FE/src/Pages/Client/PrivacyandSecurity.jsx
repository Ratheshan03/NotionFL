import React, { useState, useEffect } from "react";
import axios from "axios";
import { useAuth } from "../../Authcontext";

const DataCard = ({ title, content, type }) => {
  const renderContent = () => {
    switch (type) {
      case "json":
        return (
          <pre className="text-white max-w-full overflow-x-auto">
            {JSON.stringify(content, null, 2)}
          </pre>
        );
      case "image":
        return (
          <img
            src={`data:image/png;base64,${content}`}
            alt={title}
            className="w-full h-auto"
          />
        );
      case "text":
        return <p className="text-white">{content}</p>;
      default:
        return <p>Content type not supported.</p>;
    }
  };

  return (
    <div className="bg-gray-700 p-4 rounded-lg shadow-sm mb-4 overflow-hidden">
      <h3 className="text-lg font-semibold text-white mb-2">{title}</h3>
      {renderContent()}
    </div>
  );
};

const PrivacyAndSecurity = () => {
  const [trainingSessions, setTrainingSessions] = useState([]);
  const [selectedTrainingSession, setSelectedTrainingSession] = useState(null);
  const [selectedRoundNum, setSelectedRoundNum] = useState("");
  const [privacyData, setPrivacyData] = useState(null);
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
            `http://localhost:5000/client/get_privacy_data/${training_id}/${client_id}/${selectedRoundNum}`
          );
          console.log(response);
          setPrivacyData(response.data);
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

  const renderPrivacyData = () => {
    if (!privacyData) return null;

    return (
      <>
        {privacyData.privacy_explanation && (
          <DataCard
            title="Differential Privacy Integration Explanation"
            content={privacyData.privacy_explanation}
            type="text"
          />
        )}
        {privacyData.privacy_explanation_plot && (
          <DataCard
            title="Privacy Explanation Plot"
            content={privacyData.privacy_explanation_plot}
            type="image"
          />
        )}
        {privacyData.dp_json && (
          <DataCard
            title="Differential Privacy Data"
            content={privacyData.dp_json}
            type="json"
          />
        )}
      </>
    );
  };

  return (
    <div className="max-w-4xl mx-auto my-10 p-6 bg-gray-800 text-white shadow-lg rounded-lg overflow-hidden">
      <h2 className="text-2xl font-semibold mb-6">Privacy & Security View</h2>
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

      {isLoading ? <p>Loading...</p> : renderPrivacyData()}
    </div>
  );
};

export default PrivacyAndSecurity;
