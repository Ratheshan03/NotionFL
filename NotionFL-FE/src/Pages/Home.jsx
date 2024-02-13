// src/Pages/Home.js

import NavBar from '../components/NavBar';
import ContentArea from '../components/ContentArea';

const Home = () => {
  return (
    <div>
      <NavBar />
      <ContentArea>
        <h1 className="text-4xl font-bold text-center my-10">Welcome to NotionFL</h1>
        {/* Additional content for the Home page goes here */}
      </ContentArea>
    </div>
  );
};

export default Home;
