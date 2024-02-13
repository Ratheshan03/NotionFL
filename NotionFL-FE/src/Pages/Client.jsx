import { Routes, Route } from 'react-router-dom';
import ContentArea from '../components/ContentArea';
import NavBar from '../components/NavBar';
import SideBar from '../components/Sidebar';
import Overview from './Client/Overview';
import ViewTraining from './Client/ViewTraining';
import PrivacyandSecurity from './Client/PrivacyandSecurity';
import Aggregation from './Client/Aggregation';
import ModelEvaluation from './Client/ModelEvaluation';
import Incentives from './Client/Incentives';

const clientLinks = [
  { name: "Overview", path: "/client/overview" },
  { name: "View Training", path: "/client/view-training" },
  { name: "Privacy & Security", path: "/client/privacy-security" },
  { name: "Aggregation", path: "/client/aggregation" },
  { name: "Model Evaluation", path: "/client/model-evaluation" },
  { name: "Incentives", path: "/client/incentives" },
];

const Client = () => {
  return (
    <>
      <NavBar />
      <div className="flex">
        <SideBar links={clientLinks} />
        <ContentArea>
          <Routes>
            <Route index element={<Overview />} />
            <Route path="overview" element={<Overview />} />
            <Route path="view-training" element={<ViewTraining />} />
            <Route path="privacy-security" element={<PrivacyandSecurity />} />
            <Route path="aggregation" element={<Aggregation />} />
            <Route path="model-evaluation" element={<ModelEvaluation />} />
            <Route path="incentives" element={<Incentives />} />
          </Routes>
        </ContentArea>
      </div>
    </>
  );
};

export default Client;
