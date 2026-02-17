import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { parseHashFromURL } from "../utils/supabaseAuth";
import { refreshAccessToken } from "../utils/tokenManager";

export default function AuthRedirect() {
  const navigate = useNavigate();

  useEffect(() => {
    const session = parseHashFromURL();

    if (session.access_token) {
      localStorage.setItem("access_token", session.access_token);
      if (session.refresh_token) {
        localStorage.setItem("refresh_token", session.refresh_token);
      }
      navigate("/dashboard");
      return;
    }

    navigate("/");
  }, [navigate]);

  useEffect(() => {
    const interval = setInterval(() => {
      refreshAccessToken().catch(() => {
        localStorage.removeItem("access_token");
        localStorage.removeItem("refresh_token");
      });
    }, 55 * 60 * 1000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex items-center justify-center h-screen">
      <p className="text-lg">Logging you in...</p>
    </div>
  );
}
