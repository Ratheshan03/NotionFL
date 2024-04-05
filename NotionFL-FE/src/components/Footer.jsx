import React from "react";

const Footer = () => {
  const year = new Date().getFullYear(); // Dynamic year for copyright

  return (
    <footer className="bg-gradient-to-br from-blue-950 via-gray-900 to-black text-white p-4 md:p-8">
      <div className="max-w-6xl mx-auto flex flex-col md:flex-row justify-between items-start md:items-center space-y-8 md:space-y-0">
        {/* Left Section */}
        <div className="flex flex-col md:items-start">
          <h5 className="text-xl font-bold mb-4">Quick links</h5>
          <ul>
            <li>
              <a href="/home" className="hover:underline mb-2">
                Home
              </a>
            </li>
            <li>
              <a href="/documentation" className="hover:underline mb-2">
                Documentation
              </a>
            </li>
            <li>
              <a href="/about" className="hover:underline">
                About Us
              </a>
            </li>
          </ul>
        </div>

        {/* Center Section */}
        <div className="text-center mb-4 md:mb-0">
          <p>© {year} NotionFL. All rights reserved.</p>
        </div>

        {/* Right Section */}
        <div className="flex flex-col md:items-start">
          <h5 className="text-xl font-bold mb-4">Contact</h5>
          <ul>
            <li>
              <a
                href="mailto:info@notionfl.com"
                className="hover:underline mb-2"
              >
                info@notionfl.com
              </a>
            </li>
            <li>
              <a href="tel:+1234567890" className="hover:underline">
                +1234567890
              </a>
            </li>
          </ul>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
