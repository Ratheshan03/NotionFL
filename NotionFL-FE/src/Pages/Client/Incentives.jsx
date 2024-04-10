import React, { useState, useEffect } from "react";
import axios from "axios";
import { useAuth } from "../../Authcontext";
import "../../styles/page_styling.css";

const DataCard = ({ title, content, type }) => {
  const renderContent = () => {
    switch (type) {
      case "json":
        return (
          <pre className="json-content">
            {JSON.stringify(JSON.parse(content), null, 2)}
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
        return <pre className="text-white whitespace-pre-wrap">{content}</pre>;
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

const Incentives = () => {
  const [trainingSessions, setTrainingSessions] = useState([]);
  const [selectedTrainingSession, setSelectedTrainingSession] = useState(null);
  const [incentivesData, setIncentivesData] = useState(null);
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
    const fetchIncentivesData = async () => {
      if (selectedTrainingSession) {
        setIsLoading(true);
        try {
          const { training_id, client_id } = selectedTrainingSession;
          const response = await axios.get(
            `http://localhost:5000/client/get_incentives_data/${training_id}/${client_id}`
          );
          console.log(response);
          setIncentivesData(response.data);
        } catch (error) {
          console.error("Error fetching incentives data:", error);
        } finally {
          setIsLoading(false);
        }
      }
    };

    fetchIncentivesData();
  }, [selectedTrainingSession]);

  const handleTrainingSessionChange = (e) => {
    const trainingId = e.target.value;
    const session = trainingSessions.find((s) => s.training_id === trainingId);
    setSelectedTrainingSession(session || null);
  };

  // const handleRoundChange = (e) => {
  //   setSelectedRoundNum(e.target.value);
  // };

  const renderIncentivesData = () => {
    if (!incentivesData) return null;

    return (
      <>
        {incentivesData.contribution_distribution && (
          <DataCard
            title="Contribution Distribution"
            content={incentivesData.contribution_distribution}
            type="image"
          />
        )}
        {incentivesData.contribution_vs_incentives && (
          <DataCard
            title="Contribution vs Incentives"
            content={incentivesData.contribution_vs_incentives}
            type="image"
          />
        )}
        {incentivesData.contribution_allocation_vs_shap_values && (
          <DataCard
            title="Contribution Allocation vs SHAP Values"
            content={incentivesData.contribution_allocation_vs_shap_values}
            type="image"
          />
        )}
        {incentivesData.incentive_explanation && (
          <DataCard
            title="Incentive Explanation"
            content={incentivesData.incentive_explanation}
            type="text"
          />
        )}
      </>
    );
  };

  return (
    <div className="max-w-4xl mx-auto my-10 p-6 bg-gray-800 text-white shadow-lg rounded-lg overflow-hidden">
      <h2 className="text-2xl font-semibold mb-6">Incentives View</h2>
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
      </div>
      {isLoading ? <p>Loading...</p> : renderIncentivesData()}
    </div>
  );
};

export default Incentives;
