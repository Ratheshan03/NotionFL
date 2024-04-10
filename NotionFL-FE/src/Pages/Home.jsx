import React, { useState, useEffect } from "react";
import NavBar from "../components/NavBar";
import logo from "../assets/NFL2.png";
import heroImage from "../assets/bg-4.jpg";
import learnMoreImage from "../assets/app.gif";
import cardImageOne from "../assets/f1.png";
import cardImageTwo from "../assets/f2.png";
import Footer from "../components/Footer";

const Home = () => {
  const [isScrolled, setIsScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 0) {
        setIsScrolled(true);
      } else {
        setIsScrolled(false);
      }
    };

    // Add the event listener
    window.addEventListener("scroll", handleScroll);

    // Clean up the event listener
    return () => {
      window.removeEventListener("scroll", handleScroll);
    };
  }, []);

  const glassButtonStyle = `
    bg-transparent 
    text-white 
    font-bold 
    py-2 
    px-6   
    rounded-lg 
    transition 
    duration-300 
    shadow-lg 
    backdrop-filter 
    backdrop-blur-xl 
    bg-white/30 
    hover:bg-white/50
    w-40 h-12
  `;
  return (
    <div className="bg-gradient-to-br from-blue-950 via-gray-900 to-black text-white">
      <NavBar isScrolled={isScrolled} />
      <div
        className="relative min-h-screen bg-cover bg-center text-center flex flex-col justify-center items-center"
        style={{ backgroundImage: `url(${heroImage})` }}
      >
        <div className="bg-gray-900 bg-opacity-50 w-full h-full absolute top-0 left-0"></div>
        <div className="z-10 p-4">
          <img src={logo} alt="Logo" className="w-100 h-auto mb-4 mx-auto" />
          <p className="text-2xl text-orange-400 mb-12">
            An Explainable Mediator for Trustworthy Cross-Silo Federated
            Learning
          </p>
          <div className="flex justify-center gap-4">
            <button className={glassButtonStyle}>Learn More</button>
            <button className={glassButtonStyle}>Documentation</button>
          </div>
        </div>
      </div>

      <div className="flex flex-col md:flex-row items-center bg-dark-bg-color py-10 px-4 text-center">
        <img
          src={learnMoreImage}
          alt="Learn More"
          className="md:w-1/2 object-cover"
        />
        <div className="md:w-1/2 text-white p-6">
          <h2 className="text-4xl font-semibold mb-3">About NotionFL</h2>
          <p className="mb-4">
            The NotionFL system will act as a mediator between clients and
            servers, providing human-interpretable explanations to help
            understand FL workflows. This will improve the system&apos;s
            trustworthiness, fairness, and unbiasedness while maintaining
            privacy.
          </p>
        </div>
      </div>

      {/* Card Section */}
      <div className="bg-dark-bg-color py-10 text-center">
        <div className="flex flex-col md:flex-row justify-around items-center">
          {/* Card One */}
          <div className="bg-gray-800 m-4 p-6 rounded-lg shadow-lg max-w-sm w-full">
            <img
              src={cardImageOne}
              alt="Card One"
              className="w-full object-cover mb-4 max-h-72"
            />
            <h3 className="text-2xl mb-3">Lack of Explainability</h3>
            <p>
              Explainability is a key requirement to encourage trust in AI
              systems, yet current FL research lacks human interpretable
              explanations and makes it difficult to interpret the server
              decisions on multi-objectives.
            </p>
          </div>
          {/* Card Two */}
          <div className="bg-gray-800 m-4 p-6 rounded-lg shadow-lg max-w-sm w-full">
            <img
              src={cardImageTwo}
              alt="Card Two"
              className="w-full object-cover mb-4 max-h-72"
            />
            <h3 className="text-2xl mb-3">Lack of Trustworthiness</h3>
            <p>
              It is critical to implement trustworthy architecture in cross-silo
              FL scenarios to foster client cooperation between the clients to
              an effective FL system and to ensure compliance with AI data
              protection regulations and laws.
            </p>
          </div>
        </div>
      </div>

      <div className="bg-dark-bg-color py-10 text-center">
        <div className="flex flex-col md:flex-row items-center justify-around">
          <div className="md:w-1/2 p-6">
            <h2 className="text-4xl font-semibold mb-3">
              Getting Started with NotionFL
            </h2>
            <p className="mb-4">
              Start your journey with NotionFL today and contribute to the
              future of federated learning. Get involved with our community and
              make a difference.
            </p>
          </div>
          {/* Placeholder for Image Carousel */}
        </div>
      </div>
      <Footer />
    </div>
  );
};

export default Home;
