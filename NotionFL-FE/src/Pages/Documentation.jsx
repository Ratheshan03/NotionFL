import React from "react";

const Documentation = () => {
  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold text-gray-800 mb-4">
        Project Documentation
      </h1>

      {/* What is the Project? */}
      <section className="bg-white rounded-lg shadow-md p-4 mb-8">
        <h2 className="text-2xl font-semibold text-gray-800 mb-2">
          What is [Project Name]?
        </h2>
        <p className="text-base text-gray-600">
          [Project Name] is a web application designed to [brief description of
          the project `&apos;` s purpose]. It empowers users to [explain the
          benefits users gain from the application].
        </p>
      </section>

      {/* Why was it Created? */}
      <section className="bg-white rounded-lg shadow-md p-4 mb-8">
        <h2 className="text-2xl font-semibold text-gray-800 mb-2">
          Why was it Created?
        </h2>
        <p className="text-base text-gray-600">
          [Project Name] was created to address the challenge of [explain the
          problem your project solves]. By providing a platform for [explain how
          your application tackles the problem], it aims to [describe the
          positive impact your project has].
        </p>
      </section>

      {/* How to Use It? */}
      <section className="bg-white rounded-lg shadow-md p-4">
        <h2 className="text-2xl font-semibold text-gray-800 mb-2">
          How to Use It?
        </h2>
        <p className="text-base text-gray-600">
          Getting started with [Project Name] is simple:
        </p>

        <ol className="list-decimal pl-4 mt-2">
          <li>Create an account or sign in if you already have one.</li>
          <li>
            Explore the features available and choose the one that suits your
            needs.
          </li>
          <li>
            [Provide specific steps on how to use a core feature, with
            screenshots for better understanding (optional)]
          </li>
          <li>
            For more advanced functionalities, refer to the dedicated sections
            within the application or consult the additional resources mentioned
            below.
          </li>
        </ol>
      </section>

      {/* Additional Resources (Optional) */}
      <section className="mt-8">
        <h2 className="text-2xl font-semibold text-gray-800 mb-2">
          Additional Resources
        </h2>
        <p className="text-base text-gray-600">
          For further information and in-depth guides, you can refer to the
          following resources:
        </p>
        <ul className="list-disc pl-4 mt-2">
          <li>
            <a href="#" className="text-blue-500 hover:underline">
              Project Website (if applicable)
            </a>
          </li>
          <li>
            <a href="#" className="text-blue-500 hover:underline">
              Project Wiki or Knowledge Base (if applicable)
            </a>
          </li>
          <li>
            <a href="#" className="text-blue-500 hover:underline">
              Video Tutorials (if applicable, link to YouTube videos)
            </a>
          </li>
        </ul>
      </section>
    </div>
  );
};

export default Documentation;
