import React, { useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import serverImage from "../../assets/admin.png";
import backgroundImage from "../../assets/bg-4.jpg";
import { toast, ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import { EyeIcon, EyeOffIcon } from "@heroicons/react/outline";

const ServerSignUp = () => {
  const [formData, setFormData] = useState({
    username: "",
    email: "",
    password: "",
    organization: "",
    contact: "",
    role: "admin",
  });
  const [passwordShown, setPasswordShown] = useState(false);
  const navigate = useNavigate();

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prevState) => ({
      ...prevState,
      [name]: value,
    }));
  };

  // Simple front-end validation checks
  const isFormValid = () => {
    return (
      formData.username.trim() !== "" &&
      formData.email.trim() !== "" &&
      formData.password.trim() !== "" &&
      formData.organization.trim() !== "" &&
      formData.contact.trim() !== ""
    );
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!isFormValid()) {
      toast.error("Please fill in all fields correctly.");
      return;
    }

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/auth/register",
        formData
      );

      if (response.status === 201) {
        // If the signup was successful
        toast.success("User Registered successful!", {
          onClose: () => navigate("/login"), // Redirect to login after toast closes
        });
      } else {
        // Handle errors based on status code or error message
        toast.error(
          response.data.message || "Signup failed. Please try again."
        );
      }
    } catch (error) {
      const errorMessage = error.response
        ? error.response.data.message
        : "An error occurred. Please try again later.";
      toast.error(errorMessage);
    }
  };

  const inputFocusStyle =
    "w-full px-4 py-2 rounded-lg text-gray-700 focus:outline-none focus:ring-2 focus:ring-red-400";

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-950 via-gray-900 to-black p-4 text-white">
      <ToastContainer />
      <div className="bg-white bg-opacity-10 backdrop-filter backdrop-blur-md shadow-md rounded-3xl p-10 flex flex-col md:flex-row w-full max-w-4xl">
        <div className="md:w-1/2 flex justify-center items-center mb-6 md:mb-0">
          <img
            src={serverImage}
            alt="Server Admin"
            className="w-64 h-64 object-cover rounded-full"
          />
        </div>
        <div className="md:w-1/2">
          <h2 className="text-3xl font-bold text-center mb-6">
            Become a Server Admin
          </h2>
          <form onSubmit={handleSubmit} className="space-y-6">
            <input
              type="text"
              name="username"
              value={formData.username}
              onChange={handleChange}
              placeholder="Username"
              className={inputFocusStyle}
              required
            />
            <input
              type="email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              placeholder="Email"
              className={inputFocusStyle}
              required
            />
            <div className="relative">
              <input
                type={passwordShown ? "text" : "password"}
                name="password"
                value={formData.password}
                onChange={handleChange}
                placeholder="Password"
                className={inputFocusStyle}
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
            <input
              type="text"
              name="organization"
              value={formData.organization}
              onChange={handleChange}
              placeholder="Organization Name"
              className={inputFocusStyle}
              required
            />
            <input
              type="text"
              name="contact"
              value={formData.contact}
              onChange={handleChange}
              placeholder="Contact Number"
              className={inputFocusStyle}
              required
            />
            <button
              type="submit"
              className="w-full text-white bg-gradient-to-r from-blue-800 to-blue-950 py-2 rounded-lg hover:from-blue-800 hover:to-blue-900 transition duration-200"
            >
              SIGN UP
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

export default ServerSignUp;
