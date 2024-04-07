import React, { useState, useEffect } from "react";
import axios from "axios";
import { useAuth } from "../../Authcontext";

const DataCard = ({ title, content, isJson, isImage }) => (
  <div className="bg-gray-700 p-4 rounded-lg shadow-sm mb-4 overflow-auto">
    <h3 className="text-lg font-semibold text-white mb-2">{title}</h3>
    {isJson && (
      <pre className="text-white max-w-full overflow-x-auto">{content}</pre>
    )}
    {isImage && (
      <img
        src={`data:image/png;base64,${content}`}
        alt={title}
        className="w-full h-auto"
      />
    )}
    {!isJson && !isImage && (
      <div className="text-white max-w-full overflow-x-auto">{content}</div>
    )}
  </div>
);

const GlobalModels = () => {
  const [trainingSessions, setTrainingSessions] = useState([]);
  const [selectedTrainingId, setSelectedTrainingId] = useState("");
  const [globalData, setGlobalData] = useState(null);
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
    const fetchGlobalModelData = async () => {
      if (selectedTrainingId) {
        setIsLoading(true);
        try {
          const response = await axios.get(
            `http://localhost:5000/server/global_model_data/${selectedTrainingId}`
          );
          console.log(response);
          setGlobalData(response.data);
        } catch (error) {
          console.error("Error fetching global model data:", error);
        } finally {
          setIsLoading(false);
        }
      }
    };

    fetchGlobalModelData();
  }, [selectedTrainingId]);

  const downloadGlobalModel = () => {
    if (globalData && globalData.final_global_model) {
      // Assuming the final_global_model is a base64 string without the prefix `data:application/octet-stream;base64,`
      const base64 = globalData.final_global_model;
      const byteCharacters = atob(base64);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      const fileBlob = new Blob([byteArray], {
        type: "application/octet-stream",
      });
      const blobUrl = URL.createObjectURL(fileBlob);

      const link = document.createElement("a");
      link.href = blobUrl;
      link.download = `Global_Model_${selectedTrainingId}.pt`;
      document.body.appendChild(link);
      link.click();
      window.URL.revokeObjectURL(blobUrl); // Clean up the URL object
      document.body.removeChild(link);
    }
  };

  const renderDataContent = (data, type) => {
    if (type === "image" && data) {
      const base64String = `data:image/png;base64,${data}`;
      return (
        <img src={base64String} alt="Data Plot" className="w-full h-auto" />
      );
    } else {
      return <p>{data}</p>;
    }
  };

  return (
    <div className="max-w-4xl mx-auto my-10 p-6 bg-gray-800 text-white shadow-lg rounded-lg">
      <h2 className="text-2xl font-semibold mb-6">Global Model</h2>

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
          {globalData && (
            <>
              <DataCard
                title="Global Model Evaluation Text"
                content={globalData.global_model_eval}
                isJson={false}
              />
              <DataCard
                title="Global Model Confusion Matrix"
                content={renderDataContent(
                  globalData.global_model_cmatrix,
                  "image"
                )}
                isImage={true}
              />
              <DataCard
                title="Global SHAP Values Plot"
                content={renderDataContent(
                  globalData.global_model_shap_plot,
                  "image"
                )}
                isImage={true}
              />

              <button
                className="w-full flex justify-center py-2 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-gradient-to-r from-blue-800 to-blue-950 hover:from-blue-800 hover:to-blue-900 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                onClick={downloadGlobalModel}
              >
                Download Global Model
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default GlobalModels;
