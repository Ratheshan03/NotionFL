import React from "react";
import { AiOutlineBarChart, AiOutlineDatabase } from "react-icons/ai"; // For models and databases
import { FaUserFriends } from "react-icons/fa"; // For clients
import { MdExposure } from "react-icons/md"; // Placeholder for explainability

const Overview = () => {
  // Fetch these counts from the database, using placeholders for now
  const modelsCount = 5; // Placeholder
  const datasetsCount = 3; // Placeholder
  const clientsCount = 10; // Placeholder

  return (
    <div className="p-4 space-y-6">
      <h2 className="text-3xl font-bold text-gray-800 mb-4">Server Overview</h2>
      <p className="text-md text-gray-600">
        As a server administrator, you have the capability to monitor
        activities, manage models and datasets, and review client contributions
        in real-time.
      </p>

      {/* Explainability Section */}
      <section className="flex items-center space-x-4">
        <MdExposure size={64} className="text-gray-400" />
        <div>
          <h3 className="text-2xl font-semibold text-gray-800 mb-2">
            Explainability Mechanism
          </h3>
          <p className="text-md text-gray-600">
            The explainability mechanism in NotionFL helps you understand and
            trust the FL workflows by providing human-interpretable explanations
            of the server decisions.
          </p>
        </div>
      </section>

      {/* Models and Datasets Section */}
      <section className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Models Card */}
        <div className="bg-white rounded-lg shadow p-4 flex items-center space-x-4">
          <AiOutlineBarChart size={48} className="text-gray-400" />
          <div>
            <h4 className="text-lg font-semibold">Models</h4>
            <p>{modelsCount} Available</p>
          </div>
        </div>
        {/* Datasets Card */}
        <div className="bg-white rounded-lg shadow p-4 flex items-center space-x-4">
          <AiOutlineDatabase size={48} className="text-gray-400" />
          <div>
            <h4 className="text-lg font-semibold">Datasets</h4>
            <p>{datasetsCount} Available</p>
          </div>
        </div>
      </section>

      {/* Clients Section */}
      <section className="bg-white rounded-lg shadow p-4 flex items-center space-x-4">
        <FaUserFriends size={48} className="text-gray-400" />
        <div>
          <h3 className="text-2xl font-semibold text-gray-800 mb-2">
            Active Clients
          </h3>
          <h4 className="text-lg font-semibold">Clients</h4>
          <p>{clientsCount} Currently Active</p>
        </div>
      </section>

      {/* Additional sections can be added as necessary */}
    </div>
  );
};

export default Overview;
