import { Routes, Route } from "react-router-dom";
import ContentArea from "../components/ContentArea";
import NavBar from "../components/NavBar";
import SideBar from "../components/Sidebar";
import Overview from "./Client/Overview";
import ViewTraining from "./Client/ViewTraining";
import PrivacyandSecurity from "./Client/PrivacyandSecurity";
import Aggregation from "./Client/Aggregation";
import ModelEvaluation from "./Client/ModelEvaluation";
import Incentives from "./Client/Incentives";
import Footer from "../components/Footer";

const clientLinks = [
  { name: "Overview", path: "/clientView/overview" },
  { name: "View Training", path: "/clientView/view-training" },
  { name: "Privacy & Security", path: "/clientView/privacy-security" },
  { name: "Aggregation", path: "/clientView/aggregation" },
  { name: "Model Evaluation", path: "/clientView/model-evaluation" },
  { name: "Incentives", path: "/clientView/incentives" },
];

const Client = () => {
  return (
    <>
      <NavBar solidBackground={true} />
      <div className="flex" style={{ marginTop: "64px" }}>
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
      <Footer />
    </>
  );
};

export default Client;
