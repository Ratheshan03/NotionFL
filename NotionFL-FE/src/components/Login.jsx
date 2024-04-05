import React, { useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import { toast, ToastContainer } from "react-toastify";
import clientImage from "../assets/login.png"; // Update with appropriate login image
import "react-toastify/dist/ReactToastify.css";
import { EyeIcon, EyeOffIcon } from "@heroicons/react/outline";
import { useAuth } from "../Authcontext";

const Login = () => {
  const [credentials, setCredentials] = useState({
    username: "",
    password: "",
  });
  const [passwordShown, setPasswordShown] = useState(false);
  const navigate = useNavigate();
  const { login } = useAuth();

  const handleChange = (e) => {
    const { name, value } = e.target;
    setCredentials((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/auth/login",
        credentials
      );
      console.log(response);

      if (response.status === 200 && response.data.access_token) {
        const { access_token, user } = response.data;
        // useAuth context and passing the user data
        login({ access_token, user });
        toast.success("Login successful!", {
          onClose: () => navigate("/home"), // Redirect to login after toast closes
        });
      }
    } catch (error) {
      console.error("Error during login: ", error);
      // Check if there's a response with a message
      if (error.response && error.response.data.message) {
        toast.error(error.response.data.message);
      } else {
        toast.error("Login failed due to an unexpected error.");
      }
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-950 via-gray-900 to-black p-4 text-white">
      <ToastContainer />
      <div
        className="bg-white bg-opacity-10 backdrop-filter backdrop-blur-md shadow-md rounded-3xl p-10 flex flex-col md:flex-row w-full max-w-4xl"
        style={{ height: "60vh" }}
      >
        <div className="md:w-1/2 flex justify-center items-center mb-6 md:mb-0">
          <img
            src={clientImage}
            alt="Login"
            className="w-100 h-100 object-cover rounded-full"
          />
        </div>
        <div className="md:w-1/2">
          <h2 className="text-4xl font-bold text-center mb-3">Log in</h2>{" "}
          {/* Adjusted text size */}
          <p className="text-center mb-10">
            Welcome back! Please log in to your account.
          </p>
          <form onSubmit={handleSubmit} className="space-y-6">
            <input
              type="text"
              name="username"
              value={credentials.username}
              onChange={handleChange}
              placeholder="Username"
              className="w-full px-3 py-2 rounded-lg text-gray-700 focus:outline-none focus:ring-2 focus:ring-red-400"
              required
            />
            <div className="relative">
              <input
                type={passwordShown ? "text" : "password"}
                name="password"
                value={credentials.password}
                onChange={handleChange}
                placeholder="Password"
                className="w-full px-3 py-2 rounded-lg text-gray-700 focus:outline-none focus:ring-2 focus:ring-red-400"
                required
              />
              <div
                className="absolute inset-y-0 right-0 pr-3 flex items-center text-sm leading-5"
                onClick={() => setPasswordShown(!passwordShown)}
              >
                {passwordShown ? (
                  <EyeOffIcon className="h-6 w-6 text-gray-700 cursor-pointer" />
                ) : (
                  <EyeIcon className="h-6 w-6 text-gray-700 cursor-pointer" />
                )}
              </div>
            </div>
            <button
              type="submit"
              className="w-full text-white bg-gradient-to-r from-blue-800 to-blue-950 py-2 rounded-lg hover:from-blue-800 hover:to-blue-900 transition duration-200"
            >
              Login
            </button>
          </form>
          <p className="text-center mt-4">
            Don&apos;t have an account?
            <span
              className="text-blue-500 cursor-pointer"
              onClick={() => navigate("/signupoptions")}
            >
              {" "}
              Sign up
            </span>
          </p>
        </div>
      </div>
    </div>
  );
};

export default Login;
