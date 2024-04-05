import React from "react";
import { useNavigate } from "react-router-dom";
import backgroundImage from "../assets/bg-4.jpg";

const LandingPage = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-950 via-gray-900 to-black text-white p-4">
      <div
        className="w-11/12 md:w-3/4 lg:w-2/3 xl:max-w-4xl p-10 rounded-2xl bg-opacity-20 backdrop-filter backdrop-blur-lg shadow-2xl"
        style={{
          backgroundImage: `url(${backgroundImage})`,
          backgroundSize: "cover",
          backgroundRepeat: "no-repeat",
          height: "85vh",
          width: "85vw",
        }}
      >
        <h1 className="text-4xl md:text-5xl font-bold mb-6 text-center">
          Welcome to NotionFL
        </h1>
        <p className="text-md md:text-lg mb-8 text-center">
          The Trustworthy Explainable ediator for Cross-Silo Federated Learning
        </p>
        <div className="flex flex-col md:flex-row items-center md:space-x-4 space-y-4 md:space-y-0">
          <button
            onClick={() => navigate("/signupoptions")}
            className="md:w-1/2 bg-white bg-opacity-40 text-lg px-5 py-2 rounded-2xl hover:bg-opacity-50 transition duration-200"
          >
            Sign Up
          </button>
          <button
            onClick={() => navigate("/login")}
            className="md:w-1/2 bg-white bg-opacity-40 text-lg px-5 py-2 rounded-2xl hover:bg-opacity-50 transition duration-200"
          >
            Log In
          </button>
        </div>
      </div>
    </div>
  );
};

export default LandingPage;
