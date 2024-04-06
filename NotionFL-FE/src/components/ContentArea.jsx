const ContentArea = ({ children }) => {
  return (
    <div className="flex-grow p-6 bg-gray-100 min-h-screen overflow-auto">
      {children}
    </div>
  );
};

export default ContentArea;
