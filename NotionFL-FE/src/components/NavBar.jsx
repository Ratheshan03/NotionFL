import { useState } from "react";
import { NavLink } from "react-router-dom";
import { useAuth } from "../Authcontext";
import { MenuIcon, XIcon } from "@heroicons/react/solid"; // Ensure you have these icons installed

const NavBar = ({ isScrolled, solidBackground }) => {
  const navClass = isScrolled ? "" : "bg-transparent";
  const { currentUser } = useAuth();
  const isServerAdmin = currentUser?.user.role === "admin";
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const handleToggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  const bgColorClass = solidBackground
    ? "bg-gradient-to-r from-gray-900 from-gray-700 to-gray-900 z-50"
    : isScrolled
    ? "bg-gradient-to-r from-gray-900 from-gray-700 to-gray-900 z-50"
    : "bg-transparent";

  return (
    <nav
      className={`fixed top-0 left-0 w-full transition duration-300 z-10 ${bgColorClass}`}
    >
      <div className="max-w-6xl mx-auto px-4">
        <div className="flex justify-between items-center h-16">
          <div className="flex">
            <div>
              {/* Website Logo */}
              <NavLink to="/" className="flex items-center">
                <span className="font-semibold text-xl tracking-wide text-red-400">
                  NotionFL
                </span>
              </NavLink>
            </div>
          </div>
          {/* Primary Navbar items */}
          <div className="hidden md:flex justify-center flex-1">
            <NavLink
              to="/home"
              className="text-white hover:text-blue-400 px-3 py-2 rounded-md text-md font-medium"
            >
              Home
            </NavLink>
            <NavLink
              to={isServerAdmin ? "/serverView" : "/clientView"}
              className="text-white hover:text-blue-400 px-3 py-2 rounded-md text-md font-medium"
            >
              {isServerAdmin ? "Server View" : "Client View"}
            </NavLink>
            <NavLink
              to="/documentation"
              className="text-white hover:text-blue-400 px-3 py-2 rounded-md text-md font-medium"
            >
              Documentation
            </NavLink>
          </div>
          {/* Mobile menu button */}
          <div className="md:hidden flex items-center">
            <button
              type="button"
              className="outline-none mobile-menu-button"
              onClick={handleToggleMenu}
            >
              {isMenuOpen ? (
                <XIcon className="h-6 w-6" />
              ) : (
                <MenuIcon className="h-6 w-6" />
              )}
            </button>
          </div>
          {/* Secondary Navbar items */}
          <div className="hidden md:flex items-center">
            <NavLink
              to="/profile"
              className="text-white hover:text-blue-400 px-3 py-2 rounded-md text-md font-medium"
            >
              Profile
            </NavLink>
          </div>
        </div>
      </div>
      {/* Mobile Menu */}
      <div
        className={`${isMenuOpen ? "block" : "hidden"} md:hidden bg-gray-900`}
      >
        <NavLink
          to="/home"
          className="block text-white hover:bg-gray-700 px-3 py-2 rounded-md text-base font-medium"
        >
          Home
        </NavLink>
        <NavLink
          to={isServerAdmin ? "/serverView" : "/clientView"}
          className="block text-white hover:bg-gray-700 px-3 py-2 rounded-md text-base font-medium"
        >
          {isServerAdmin ? "Server View" : "Client View"}
        </NavLink>
        <NavLink
          to="/documentation"
          className="block text-white hover:bg-gray-700 px-3 py-2 rounded-md text-base font-medium"
        >
          Documentation
        </NavLink>
        <NavLink
          to="/profile"
          className="block text-white hover:bg-gray-700 px-3 py-2 rounded-md text-base font-medium"
        >
          Profile
        </NavLink>
      </div>
    </nav>
  );
};

export default NavBar;
