import React, { createContext, useState, useContext, useEffect } from "react";

const AuthContext = createContext();

export const useAuth = () => {
  return useContext(AuthContext);
};

export const AuthProvider = ({ children }) => {
  const [currentUser, setCurrentUser] = useState(() => {
    const userData = localStorage.getItem("userData");
    return userData ? JSON.parse(userData) : null;
  });

  useEffect(() => {
    if (currentUser) {
      localStorage.setItem("userData", JSON.stringify(currentUser));
    } else {
      localStorage.removeItem("userData");
    }
  }, [currentUser]);

  const login = (userData) => {
    setCurrentUser(userData);
  };

  const logout = () => {
    setCurrentUser(null);
  };

  const value = {
    currentUser,
    login,
    logout,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};
