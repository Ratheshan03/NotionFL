import { NavLink } from "react-router-dom";

const SideBar = ({ links }) => {
  return (
    <div className="w-64 bg-gradient-to-br from-blue-950 via-gray-900 to-black min-h-screen p-5">
      <ul className="flex flex-col space-y-3">
        {links.map((link, index) => (
          <li key={index}>
            <NavLink
              to={link.path}
              className="text-white text-sm hover:text-blue-400 transition duration-300 block p-2"
              activeClassName="font-bold bg-blue-700 rounded-md"
            >
              {link.name}
            </NavLink>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default SideBar;
