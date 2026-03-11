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
    <div className="min-h-screen px-4 py-8 sm:px-6 lg:px-8">
      <div className="mx-auto flex h-[70vh] max-w-3xl items-center justify-center">
        <div className="theriax-surface w-full max-w-lg p-8 text-center sm:p-10">
          <p className="theriax-kicker">Authenticating</p>
          <h2 className="theriax-display mt-2 text-3xl font-bold text-slate-900">Logging you in</h2>
          <p className="theriax-muted mt-3 text-sm sm:text-base">
            We are validating your session and preparing your dashboard.
          </p>
          <div className="mt-6 flex items-center justify-center">
            <div className="h-11 w-11 animate-spin rounded-full border-4 border-teal-200 border-t-teal-600" />
          </div>
        </div>
      </div>
    </div>
  );
}
