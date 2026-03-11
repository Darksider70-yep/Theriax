import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { FormInput } from "../components/Forminput";
import api from "../utils/api";

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [feedback, setFeedback] = useState({ type: "", text: "" });
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);
    setFeedback({ type: "", text: "" });

    try {
      const res = await api.post("/login", { email, password });
      localStorage.setItem("access_token", res.data.access_token);
      if (res.data.refresh_token) {
        localStorage.setItem("refresh_token", res.data.refresh_token);
      }
      setFeedback({ type: "success", text: "Login successful. Redirecting to dashboard..." });
      setTimeout(() => navigate("/dashboard"), 800);
    } catch (err) {
      setFeedback({ type: "error", text: err.response?.data?.detail || "Login failed" });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen px-4 py-8 sm:px-6 lg:px-8">
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45 }}
        className="mx-auto max-w-5xl overflow-hidden theriax-surface theriax-auth-grid"
      >
        <section className="theriax-auth-brand bg-gradient-to-br from-teal-700 via-teal-600 to-cyan-700 p-8 text-white sm:p-10">
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-teal-100">Theriax</p>
          <h1 className="theriax-display mt-3 text-3xl font-extrabold leading-tight sm:text-4xl">
            AI-guided medicine insights for every clinical case.
          </h1>
          <p className="mt-4 max-w-md text-sm text-teal-50 sm:text-base">
            Access predictive recommendations, dosage context, and model-backed confidence from one workspace.
          </p>
          <ul className="mt-8 space-y-3 text-sm text-teal-50/95">
            <li>Fast condition triage with model-generated ranking.</li>
            <li>Structured recommendation logs for auditability.</li>
            <li>Seamless handoff between dashboard analytics and live AI search.</li>
          </ul>
        </section>

        <section className="bg-white/90 p-8 sm:p-10">
          <p className="theriax-kicker">Welcome Back</p>
          <h2 className="theriax-display mt-2 text-3xl font-bold text-slate-900">Sign in to continue</h2>
          <p className="theriax-muted mt-2 text-sm">Use your registered credentials to open the workspace.</p>

          <form onSubmit={handleLogin} className="mt-6 space-y-4">
            <FormInput
              id="login-email"
              label="Email Address"
              type="email"
              placeholder="name@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              autoComplete="email"
            />
            <FormInput
              id="login-password"
              label="Password"
              type="password"
              placeholder="Enter password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="current-password"
            />

            <button type="submit" disabled={loading} className="theriax-btn theriax-btn-primary mt-2 w-full">
              {loading ? "Signing in..." : "Log In"}
            </button>
          </form>

          {feedback.text && (
            <div
              className={`mt-4 theriax-alert ${
                feedback.type === "error" ? "theriax-alert-error" : "theriax-alert-success"
              }`}
            >
              {feedback.text}
            </div>
          )}

          <p className="mt-6 text-sm text-slate-600">
            New to Theriax?{" "}
            <Link to="/signup" className="font-semibold text-teal-700 hover:text-teal-800">
              Create account
            </Link>
          </p>
        </section>
      </motion.div>
    </div>
  );
}
