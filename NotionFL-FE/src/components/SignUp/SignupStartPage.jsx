import React from "react";
import { useNavigate } from "react-router-dom";
import serverAdminImage from "../../assets/admin.png";
import clientContributorImage from "../../assets/client.png";
import backgroundImage from "../../assets/bg-4.jpg";

const SignupStartPage = () => {
  const navigate = useNavigate();

  const handleSignUpOptionClick = (role) => {
    navigate(`/signup/${role}`);
  };

  const handleLoginClick = () => {
    navigate("/login");
  };

  const cardStyle = {
    backgroundImage: `url(${backgroundImage})`,
    backgroundSize: "cover",
    backgroundRepeat: "no-repeat",
    backgroundPosition: "center",
    backdropFilter: "blur(8px)",
    WebkitBackdropFilter: "blur(8px)",
    height: "70vh",
    width: "85vw",
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-blue-950 via-gray-900 to-black p-4 text-white">
      <div
        className="bg-white bg-opacity-10 backdrop-filter backdrop-blur-md shadow-md rounded-3xl p-10 w-full max-w-5xl mb-5"
        style={{ ...cardStyle }}
      >
        <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold text-center mb-8">
          SIGN UP BASED ON THE ROLES
        </h1>
        <div className="flex flex-col sm:flex-row items-center justify-center gap-8">
          {/* Server Admin Card */}
          <div
            className="signup-card bg-opacity-40 backdrop-filter backdrop-blur-lg shadow-xl rounded-xl p-5 cursor-pointer hover:scale-105 transition-transform"
            onClick={() => handleSignUpOptionClick("server")}
          >
            <img
              src={serverAdminImage}
              alt="Server Admin"
              className="w-32 h-32 md:w-40 md:h-40 object-cover rounded-lg mb-4 mx-auto" // Responsive image size
            />
            <h2 className="text-xl md:text-2xl font-bold text-center mb-2">
              Server Admin
            </h2>
            <p className="text-sm text-center mb-4">
              Manage and monitor federated learning sessions and maintain system
              integrity
            </p>
          </div>

          {/* Client Contributor Card */}
          <div
            className="signup-card bg-opacity-40 backdrop-filter backdrop-blur-lg shadow-xl rounded-xl p-5 cursor-pointer hover:scale-105 transition-transform"
            onClick={() => handleSignUpOptionClick("client")}
          >
            <img
              src={clientContributorImage}
              alt="Client Contributor"
              className="w-32 h-32 md:w-40 md:h-40 object-cover rounded-lg mb-4 mx-auto" // Responsive image size
            />
            <h2 className="text-xl md:text-2xl font-bold text-center mb-2">
              Client Contributor
            </h2>
            <p className="text-sm text-center">
              Contribute to federated learning projects and help improve machine
              learning models
            </p>
          </div>
        </div>
        <div className="mt-8 text-center">
          <p>
            Already a member?{" "}
            <span
              onClick={handleLoginClick}
              className="text-blue-300 cursor-pointer hover:underline"
            >
              Log in
            </span>
          </p>
        </div>
      </div>
    </div>
  );
};

export default SignupStartPage;
