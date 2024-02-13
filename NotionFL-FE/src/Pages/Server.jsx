// src/Pages/Server.jsx

import { Routes, Route } from 'react-router-dom';
import ContentArea from '../components/ContentArea';
import NavBar from '../components/NavBar';
import SideBar from '../components/Sidebar';
import Overview from './Server/Overview';
import ClientView from './Server/ClientsView';
import StartTraining from './Server/StartTraining';
import ViewTraining from './Server/ViewTraining';
import Visualizations from './Server/Visualizations';
import PrivacyandSecurity from './Server/PrivacyandSecurity';
import GlobalModel from './Server/GlobalModel';

const serverLinks = [
  { name: "Overview", path: "/server/overview" },
  { name: "Client View", path: "/server/client-view" },
  { name: "Start Training", path: "/server/start-training" },
  { name: "View Training", path: "/server/view-training" },
  { name: "Visualizations", path: "/server/visualizations" },
  { name: "Privacy & Security", path: "/server/privacy-security" },
  { name: "Global Model", path: "/server/global-model" },
];

const Server = () => {
  return (
    <>
      <NavBar />
      <div className=" flex h-screen">
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
    </>
  );
};

export default Server;
