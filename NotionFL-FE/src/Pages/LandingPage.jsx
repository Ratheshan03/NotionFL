import React from "react";
import { useNavigate } from "react-router-dom";
import backgroundImage from "../assets/bg-4.jpg";
import logoImage from "../assets/NFL2.png"; // Assuming you have a logo.png in your assets folder

const LandingPage = () => {
  const navigate = useNavigate();

  return (
    <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-blue-950 via-gray-900 to-black p-4 text-white">
      <div
        className="flex flex-col items-center w-full max-w-5xl p-10 space-y-9 bg-white/20 rounded-2xl backdrop-blur-lg shadow-2xl"
        style={{
          backgroundImage: `url(${backgroundImage})`,
          backgroundSize: "cover",
          backgroundRepeat: "no-repeat",
        }}
      >
        <img src={logoImage} alt="NotionFL Logo" className="w-90 h-auto mb-4" />
        <h1 className="text-center text-4xl font-bold mb-2 md:text-5xl">
          Welcome to NotionFL
        </h1>
        <p className="text-center text-md mb-4 md:text-lg">
          The Trustworthy Explainable Mediator for Cross-Silo Federated Learning
        </p>
        <div className="flex w-full flex-col items-center space-y-4 md:flex-row md:space-y-0 md:space-x-4">
          <button
            onClick={() => navigate("/signupoptions")}
            className="w-full md:w-1/2 bg-white/40 text-lg px-5 py-2 rounded-2xl transition duration-200 hover:bg-white/50"
          >
            Sign Up
          </button>
          <button
            onClick={() => navigate("/login")}
            className="w-full md:w-1/2 bg-white/40 text-lg px-5 py-2 rounded-2xl transition duration-200 hover:bg-white/50"
          >
            Log In
          </button>
        </div>
      </div>
    </div>
  );
};

export default LandingPage;
