// src/Pages/Home.js

import NavBar from '../components/NavBar';
import ContentArea from '../components/ContentArea';
import '../index.css'; 

const Home = () => {
  return (
    <div>
      <NavBar />
      <ContentArea>
        <div className="flex flex-col items-center">
          <div className="logo-animation mb-4">
            {/* Your logo goes here */}
            <h2 className="text-2xl font-semibold">NotionFL</h2>
          </div>
          <h1 className="text-4xl font-bold text-center my-10">NotionFL</h1>
          {/* Additional content for the Home page goes here */}
          <h2 className="text-2xl font-semibold">An Explainable Mediator for Trustworthy Cross-Silo Federated Learning</h2>
        </div>
        
      </ContentArea>
    </div>
  );
};

export default Home;
