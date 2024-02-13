// src/components/NavBar.js

import { NavLink } from 'react-router-dom';

const NavBar = () => {
  return (
    <nav className="bg-blue-500 p-4 shadow-md">
      <ul className="flex justify-center space-x-10">
        <li>
          <NavLink
            to="/"
            className="text-white text-lg hover:text-blue-300 transition duration-300"
            activeClassName="font-bold border-b-2 border-blue-300"
            exact
          >
            Home
          </NavLink>
        </li>
        <li>
          <NavLink
            to="/server"
            className="text-white text-lg hover:text-blue-300 transition duration-300"
            activeClassName="font-bold border-b-2 border-blue-300"
          >
            Server
          </NavLink>
        </li>
        <li>
          <NavLink
            to="/client"
            className="text-white text-lg hover:text-blue-300 transition duration-300"
            activeClassName="font-bold border-b-2 border-blue-300"
          >
            Client
          </NavLink>
        </li>
      </ul>
    </nav>
  );
};

export default NavBar;
