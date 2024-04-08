import React, { useState } from "react";

const ProfilePage = () => {
  const [editMode, setEditMode] = useState(false);
  const [userData, setUserData] = useState({
    username: "",
    email: "",
    organization: "",
    contact: "",
  });

  const handleEditClick = () => {
    setEditMode(!editMode);
  };

  const handleChange = (event) => {
    setUserData({ ...userData, [event.target.name]: event.target.value });
  };

  // Implement logic to fetch user data and contributions on component mount

  return (
    <div className="p-4 space-y-6">
      <h2 className="text-3xl font-bold text-gray-800 mb-4">My Profile</h2>

      <div className="bg-white rounded-lg shadow-md p-4 flex flex-col space-y-4">
        {/* Profile Picture */}
        <div className="flex items-center justify-center">
          <img
            src="https://via.placeholder.com/150" // Placeholder image
            alt="Profile Picture"
            className="w-24 h-24 rounded-full object-cover"
          />
        </div>

        {/* User Details */}
        <div className="flex flex-col space-y-2">
          {editMode ? (
            <>
              <input
                type="text"
                name="username"
                placeholder="Username"
                className="border rounded-md px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-500"
                value={userData.username}
                onChange={handleChange}
              />
              <input
                type="email"
                name="email"
                placeholder="Email"
                className="border rounded-md px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-500"
                value={userData.email}
                onChange={handleChange}
              />
              <input
                type="text"
                name="organization"
                placeholder="Organization"
                className="border rounded-md px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-500"
                value={userData.organization}
                onChange={handleChange}
              />
              <input
                type="text"
                name="contact"
                placeholder="Contact"
                className="border rounded-md px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-500"
                value={userData.contact}
                onChange={handleChange}
              />
            </>
          ) : (
            <>
              <p className="text-lg font-medium text-gray-700">
                {userData.username}
              </p>
              <p className="text-base text-gray-500">{userData.email}</p>
              <p className="text-base text-gray-500">{userData.organization}</p>
              <p className="text-base text-gray-500">{userData.contact}</p>
            </>
          )}
        </div>

        {/* Edit Button */}
        <button
          className={`text-white px-4 py-2 rounded-md ${
            editMode ? "bg-blue-500" : "bg-gray-400"
          }`}
          onClick={handleEditClick}
        >
          {editMode ? "Save" : "Edit Bio"}
        </button>
      </div>

      {/* Contributions Section */}
      <h2>Your Contributions</h2>
      {/* Implement logic to display user contributions here */}
      <p className="text-gray-500">
        This section will display your past contributions.
      </p>
    </div>
  );
};

export default ProfilePage;
