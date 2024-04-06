import { Routes, Route } from "react-router-dom";
import ContentArea from "../components/ContentArea";
import NavBar from "../components/NavBar";
import SideBar from "../components/Sidebar";
import Overview from "./Server/Overview";
import ClientView from "./Server/ClientsView";
import StartTraining from "./Server/StartTraining";
import ViewTraining from "./Server/ViewTraining";
import Visualizations from "./Server/Visualizations";
import PrivacyandSecurity from "./Server/PrivacyandSecurity";
import GlobalModel from "./Server/GlobalModel";
import Footer from "../components/Footer";

const serverLinks = [
  { name: "Overview", path: "/serverView/overview" },
  { name: "Client View", path: "/serverView/client-view" },
  { name: "Start Training", path: "/serverView/start-training" },
  { name: "View Training", path: "/serverView/view-training" },
  { name: "Visualizations", path: "/serverView/visualizations" },
  { name: "Privacy & Security", path: "/serverView/privacy-security" },
  { name: "Global Model", path: "/serverView/global-model" },
];

const Server = () => {
  return (
    <>
      <NavBar solidBackground={true} />
      <div className="flex" style={{ marginTop: "64px" }}>
        <SideBar links={serverLinks} />
        <ContentArea>
          <Routes>
            <Route index element={<Overview />} />
            <Route path="overview" element={<Overview />} />
            <Route path="client-view" element={<ClientView />} />
            <Route path="start-training" element={<StartTraining />} />
            <Route path="view-training" element={<ViewTraining />} />
            <Route path="visualizations" element={<Visualizations />} />
            <Route path="privacy-security" element={<PrivacyandSecurity />} />
            <Route path="global-model" element={<GlobalModel />} />
          </Routes>
        </ContentArea>
      </div>
      <Footer />
    </>
  );
};

export default Server;
