import React from "react";
import { AiOutlineDatabase, AiOutlineDollarCircle } from "react-icons/ai"; // For databases and incentives
import { FaRegEye } from "react-icons/fa"; // For explainability and privacy
import { BiBook } from "react-icons/bi"; // For use cases

const ClientOverview = () => {
  // Placeholder data
  const useCasesCount = 4;
  const databasesCount = 3;
  const privacyUpdates = "Latest differential privacy mechanisms in place";
  const fairIncentives = "Competitive payouts based on contributions";

  return (
    <div className="p-4 space-y-6">
      <h2 className="text-3xl font-bold text-gray-800 mb-4">Client Overview</h2>
      <p className="text-md text-gray-600">
        Explore how your contributions power federated learning, understand our
        privacy practices, and see how fair incentives are calculated.
      </p>

      {/* Use Cases Section */}
      <section className="flex items-center space-x-4">
        <BiBook size={64} className="text-gray-400" />
        <div>
          <h3 className="text-2xl font-semibold text-gray-800 mb-2">
            Training Use Cases
          </h3>
          <p className="text-md text-gray-600">
            {useCasesCount} different use cases are available to contribute to,
            spanning various industries and research fields.
          </p>
        </div>
      </section>

      {/* Databases and Privacy Section */}
      <section className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Databases Card */}
        <div className="bg-white rounded-lg shadow p-4 flex items-center space-x-4">
          <AiOutlineDatabase size={48} className="text-gray-400" />
          <div>
            <h4 className="text-lg font-semibold">Datasets</h4>
            <p>{databasesCount} available for training</p>
          </div>
        </div>
        {/* Privacy Updates Card */}
        <div className="bg-white rounded-lg shadow p-4 flex items-center space-x-4">
          <FaRegEye size={48} className="text-gray-400" />
          <div>
            <h4 className="text-lg font-semibold">Privacy Mechanism</h4>
            <p>{privacyUpdates}</p>
          </div>
        </div>
      </section>

      {/* Incentives Section */}
      <section className="bg-white rounded-lg shadow p-4 flex items-center space-x-4">
        <AiOutlineDollarCircle size={48} className="text-gray-400" />
        <div>
          <h3 className="text-2xl font-semibold text-gray-800 mb-2">
            Fair Incentives
          </h3>
          <p className="text-lg font-semibold">{fairIncentives}</p>
        </div>
      </section>

      {/* Additional sections can be added as necessary */}
    </div>
  );
};

export default ClientOverview;
