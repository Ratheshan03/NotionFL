import React, { useState, useEffect } from "react";
import axios from "axios";
import { useAuth } from "../../Authcontext"; // Make sure this path is correct for your auth context

const DataCard = ({ title, content, isJson }) => (
  <div className="bg-gray-700 p-4 rounded-lg shadow-sm mb-4 overflow-auto">
    <h3 className="text-lg font-semibold text-white mb-2">{title}</h3>
    {isJson ? (
      <pre className="text-white max-w-full overflow-x-auto">{content}</pre>
    ) : (
      <div className="text-white max-w-full overflow-x-auto">{content}</div>
    )}
  </div>
);

const PrivacyAndSecurity = () => {
  const [trainingSessions, setTrainingSessions] = useState([]);
  const [selectedTrainingId, setSelectedTrainingId] = useState("");
  const [privacyData, setPrivacyData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const { currentUser } = useAuth();

  useEffect(() => {
    const fetchTrainingSessions = async () => {
      if (currentUser?.user?.id) {
        try {
          const { data } = await axios.get(
            `http://localhost:5000/training/training_sessions/${currentUser.user.id}`
          );
          setTrainingSessions(data);
        } catch (error) {
          console.error("Error fetching training sessions:", error);
        }
      }
    };

    fetchTrainingSessions();
  }, [currentUser]);

  useEffect(() => {
    const fetchPrivacyData = async () => {
      if (selectedTrainingId) {
        setIsLoading(true);
        try {
          const response = await axios.get(
            `http://localhost:5000/server/privacy_data/${selectedTrainingId}`
          );
          console.log(response);
          setPrivacyData(response.data);
        } catch (error) {
          console.error("Error fetching privacy and security data:", error);
        } finally {
          setIsLoading(false);
        }
      }
    };

    fetchPrivacyData();
  }, [selectedTrainingId]);

  const renderPrivacyContent = (data, type) => {
    switch (type) {
      case "image":
        return (
          <img
            src={`data:image/png;base64,${data}`}
            alt="Privacy Data"
            className="w-full h-auto"
          />
        );
      case "json":
        // Assuming the JSON data is already an object
        return JSON.stringify(data, null, 2);
      case "text":
      default:
        return <p>{data}</p>;
    }
  };

  return (
    <div className="max-w-4xl mx-auto my-10 p-6 bg-gray-800 text-white shadow-lg rounded-lg">
      <h2 className="text-2xl font-semibold mb-6">Privacy & Security</h2>

      <div className="mb-6">
        <label className="block text-sm font-medium mb-2">Training ID</label>
        <select
          className="block w-full border border-gray-500 bg-gray-700 text-white rounded-md p-2"
          value={selectedTrainingId}
          onChange={(e) => setSelectedTrainingId(e.target.value)}
        >
          <option value="">Select Training Session</option>
          {trainingSessions.map((session, index) => (
            <option key={index} value={session.training_id}>
              {session.training_id}
            </option>
          ))}
        </select>
      </div>

      {isLoading ? (
        <p>Loading...</p>
      ) : (
        <div>
          {privacyData && (
            <>
              <DataCard
                title="Privacy Explanation"
                content={renderPrivacyContent(
                  privacyData.privacy_explanation,
                  "text"
                )}
              />
              <DataCard
                title="Privacy Explanation Plot"
                content={renderPrivacyContent(
                  privacyData.privacy_explanation_plot,
                  "image"
                )}
              />
              <DataCard
                title="Differential Privacy Data"
                content={renderPrivacyContent(privacyData.dp_json, "json")}
                isJson={true}
              />
              <DataCard
                title="Secure Aggregation Data"
                content={renderPrivacyContent(privacyData.aggregation, "json")}
                isJson={true}
              />
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default PrivacyAndSecurity;
