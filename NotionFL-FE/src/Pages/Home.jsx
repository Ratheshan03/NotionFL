// src/Pages/Home.js

import NavBar from "../components/NavBar";
import ContentArea from "../components/ContentArea";
import logo from "../assets/banner.png"; // Replace with the path to your actual logo image

const Home = () => {
  return (
    <div>
      <NavBar />
      <ContentArea>
        <div className="flex flex-col items-center justify-center min-h-screen p-4">
          <img src={logo} alt="NotionFL Logo" className="h-50 w-auto mb-4" />
          <h1 className="text-3xl md:text-4xl font-bold text-center my-4">
            Welcome to NotionFL
          </h1>
          <h2 className="text-xl md:text-2xl font-semibold text-gray-600 text-center mb-3">
            An Explainable Mediator for Trustworthy Cross-Silo Federated
            Learning
          </h2>
          <p className="text-sm md:text-base text-center mb-6">
            NotionFL is your gateway to a new era of collaborative,
            privacy-preserving, and secure machine learning.
          </p>
          <div className="flex flex-wrap justify-center gap-4">
            <button className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
              Learn More
            </button>
            <button className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded">
              Start Training
            </button>
          </div>
        </div>
      </ContentArea>
    </div>
  );
};

export default Home;
