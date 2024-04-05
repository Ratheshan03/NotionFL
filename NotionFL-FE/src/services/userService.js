// userService.js

const BASE_URL = "http://127.0.0.1:5000";

async function registerUser(userData) {
  const response = await fetch(`${BASE_URL}/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(userData),
  });
  return response.json();
}

// Export for use in your components
export { registerUser };
