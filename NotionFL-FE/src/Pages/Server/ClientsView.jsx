import React, { useState, useEffect } from "react";
import axios from "axios";
import { useAuth } from "../../Authcontext";
import "../../styles/page_styling.css";

const DataCard = ({ title, content }) => (
  <div className="bg-gray-700 p-4 rounded-lg shadow-sm mb-4 overflow-auto">
    <h3 className="text-lg font-semibold text-white mb-2">{title}</h3>
    <div className="text-white max-w-full overflow-x-auto">{content}</div>
  </div>
);

const ClientView = () => {
  const [trainingSessions, setTrainingSessions] = useState([]);
  const [clientCounts, setClientCounts] = useState({});
  const [selectedTrainingId, setSelectedTrainingId] = useState("");
  const [selectedClientId, setSelectedClientId] = useState("");
  const [selectedRoundNum, setSelectedRoundNum] = useState("");
  const [clientData, setClientData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const { currentUser } = useAuth();

  useEffect(() => {
    const fetchTrainingSessions = async () => {
      try {
        const userId = currentUser?.user?.id;
        const { data } = await axios.get(
          `http://localhost:5000/training/training_sessions/${userId}`
        );
        console.log(data);
        processTrainingSessions(data);
      } catch (error) {
        console.error("Error fetching training sessions:", error);
      }
    };

    if (currentUser) {
      fetchTrainingSessions();
    }
  }, [currentUser]);

  useEffect(() => {
    const fetchClientData = async () => {
      if (selectedTrainingId && selectedClientId && selectedRoundNum) {
        setIsLoading(true);
        try {
          const response = await axios.get(
            `http://localhost:5000/server/get_client_data/${selectedTrainingId}/${selectedClientId}/${selectedRoundNum}`
          );
          console.log(response);
          setClientData(response.data);
        } catch (error) {
          console.error("Error fetching client-specific data:", error);
        } finally {
          setIsLoading(false);
        }
      }
    };

    fetchClientData();
  }, [selectedTrainingId, selectedClientId, selectedRoundNum]);

  const processTrainingSessions = (sessions) => {
    const newTrainingSessions = sessions.map((session) => session.training_id);
    setTrainingSessions(newTrainingSessions);

    const newClientCountPerSession = {};
    sessions.forEach((session) => {
      const numClients = session.config.num_clients;
      newClientCountPerSession[session.training_id] = numClients;
    });
    setClientCounts(newClientCountPerSession);
  };

  const renderDataContent = (data, type) => {
    if (type === "image") {
      return <img src={`data:image/png;base64,${data}`} alt="Data Plot" />;
    } else if (type === "text") {
      return <p>{data}</p>;
    } else if (type === "json") {
      return (
        <pre className="json-content">
          {JSON.stringify(JSON.parse(data), null, 2)}
        </pre>
      );
    }
    return <p>Data not available</p>;
  };

  return (
    <div className="max-w-4xl mx-auto my-10 p-6 bg-gray-800 text-white shadow-lg rounded-lg">
      <h2 className="text-2xl font-semibold mb-6">Client Data Viewer</h2>

      <div className="flex space-x-4 mb-8">
        <div className="flex-1">
          <label className="block text-sm font-medium mb-2">Training ID</label>
          <select
            className="block w-full border border-gray-500 bg-gray-700 text-white rounded-md p-2"
            value={selectedTrainingId}
            onChange={(e) => setSelectedTrainingId(e.target.value)}
          >
            <option value="">Select Training Session</option>
            {trainingSessions.map((session, index) => (
              <option key={index} value={session}>
                {session}
              </option>
            ))}
          </select>
        </div>

        <div className="flex-1">
          <label className="block text-sm font-medium mb-2">Client ID</label>
          <select
            className="block w-full border border-gray-500 bg-gray-700 text-white rounded-md p-2"
            value={selectedClientId}
            onChange={(e) => setSelectedClientId(e.target.value)}
          >
            <option value="">Select Client</option>
            {/* Hardcoded client options */}
            <option value="0">Client 0</option>
            <option value="1">Client 1</option>
            <option value="2">Client 2</option>
            <option value="3">Client 3</option>
          </select>
        </div>
        <div className="flex-1">
          <label className="block text-sm font-medium mb-2">Round Number</label>
          <select
            className="block w-full border border-gray-500 bg-gray-700 text-white rounded-md p-2"
            value={selectedRoundNum}
            onChange={(e) => setSelectedRoundNum(e.target.value)}
          >
            <option value="">Select Round</option>
            {/* Example: You can hardcode or dynamically populate these options */}
            <option value="1">Round 1</option>
            <option value="2">Round 2</option>
            {/* ... */}
          </select>
        </div>
      </div>
      {isLoading ? (
        <p>Loading...</p>
      ) : (
        <div>
          {clientData && (
            <>
              {clientData.eval_logs && (
                <DataCard
                  title="Evaluation Logs"
                  content={renderDataContent(clientData.eval_logs, "json")}
                />
              )}
              {clientData.eval_plot && (
                <DataCard
                  title="Evaluation Plot"
                  content={renderDataContent(clientData.eval_plot, "image")}
                />
              )}
              {clientData.eval_text && (
                <DataCard
                  title="Evaluation Text"
                  content={renderDataContent(clientData.eval_text, "text")}
                />
              )}
              {clientData.training_logs && (
                <DataCard
                  title="Training Logs"
                  content={renderDataContent(clientData.training_logs, "json")}
                />
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default ClientView;
