// src/App.jsx

import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './Pages/Home';
import Server from './Pages/Server';
import Client from './Pages/Client';
import './index.css';

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/server/*" element={<Server />} />
          <Route path="/client/*" element={<Client />} />
          
          {/* Define other routes here */}
        </Routes>
      </div>
    </Router>
  );
}

export default App;
