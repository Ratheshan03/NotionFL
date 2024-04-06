import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import LandingPage from "./Pages/LandingPage";
import Home from "./Pages/Home";
import Server from "./Pages/Server";
import Client from "./Pages/Client";
import SignupStartPage from "./components/SignUp/SignupStartPage";
import ServerSignup from "./components/SignUp/ServerSignUp";
import ClientSignup from "./components/SignUp/ClientSignUp";
import Login from "./components/Login";
import "./index.css";
import axios from "axios";
import { AuthProvider } from "./Authcontext";

// Configure axios to send cookies with each request
axios.defaults.withCredentials = true;

function App() {
  return (
    <Router>
      <AuthProvider>
        <div className="App">
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/home" element={<Home />} />
            <Route path="/serverView/*" element={<Server />} />
            <Route path="/clientView/*" element={<Client />} />
            <Route path="/signupoptions" element={<SignupStartPage />} />
            <Route path="/signup/server" element={<ServerSignup />} />
            <Route path="/signup/client" element={<ClientSignup />} />
            <Route path="/login" element={<Login />} />
            {/* Define other routes here */}
          </Routes>
        </div>
      </AuthProvider>
    </Router>
  );
}

export default App;
