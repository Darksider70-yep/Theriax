import { Navigate } from "react-router-dom";

function isExpired(token) {
  try {
    const payload = JSON.parse(atob(token.split(".")[1]));
    if (!payload?.exp) {
      return false;
    }
    return Date.now() >= payload.exp * 1000;
  } catch {
    return true;
  }
}

export default function ProtectedRoute({ children }) {
  const accessToken = localStorage.getItem("access_token");

  if (!accessToken || isExpired(accessToken)) {
    localStorage.removeItem("access_token");
    localStorage.removeItem("refresh_token");
    return <Navigate to="/" />;
  }

  return children;
}
