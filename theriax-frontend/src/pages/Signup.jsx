import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { FormInput } from "../components/Forminput";
import api from "../utils/api";

export default function Signup() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [feedback, setFeedback] = useState({ type: "", text: "" });
  const [success, setSuccess] = useState(false);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSignup = async (e) => {
    e.preventDefault();
    setLoading(true);
    setFeedback({ type: "", text: "" });

    try {
      await api.post("/signup", { email, password });
      setFeedback({ type: "success", text: "Signup successful. Redirecting to login..." });
      setSuccess(true);
    } catch (err) {
      setFeedback({ type: "error", text: err.response?.data?.detail || "Signup failed" });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!success) return;
    const timer = setTimeout(() => navigate("/"), 2000);
    return () => clearTimeout(timer);
  }, [success, navigate]);

  return (
    <div className="min-h-screen px-4 py-8 sm:px-6 lg:px-8">
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45 }}
        className="mx-auto max-w-5xl overflow-hidden theriax-surface theriax-auth-grid"
      >
        <section className="theriax-auth-brand bg-gradient-to-br from-orange-600 via-amber-600 to-teal-600 p-8 text-white sm:p-10">
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-amber-100">Theriax</p>
          <h1 className="theriax-display mt-3 text-3xl font-extrabold leading-tight sm:text-4xl">
            Create your medical intelligence workspace.
          </h1>
          <p className="mt-4 max-w-md text-sm text-amber-50 sm:text-base">
            Start tracking AI recommendations and unlock dashboard analytics for treatment decision support.
          </p>
          <ul className="mt-8 space-y-3 text-sm text-amber-50/95">
            <li>Store model predictions with confidence and severity context.</li>
            <li>Visualize medicine frequency patterns over time.</li>
            <li>Access instant AI search for symptom-condition scenarios.</li>
          </ul>
        </section>

        <section className="bg-white/90 p-8 sm:p-10">
          <p className="theriax-kicker">Get Started</p>
          <h2 className="theriax-display mt-2 text-3xl font-bold text-slate-900">Create your account</h2>
          <p className="theriax-muted mt-2 text-sm">Use a valid email and secure password to register.</p>

          <form onSubmit={handleSignup} className="mt-6 space-y-4">
            <FormInput
              id="signup-email"
              label="Email Address"
              type="email"
              placeholder="name@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              autoComplete="email"
            />

            <FormInput
              id="signup-password"
              label="Password"
              type="password"
              placeholder="Create password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="new-password"
            />

            <button type="submit" disabled={loading} className="theriax-btn theriax-btn-primary mt-2 w-full">
              {loading ? "Creating account..." : "Sign Up"}
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
            Already have an account?{" "}
            <Link to="/" className="font-semibold text-teal-700 hover:text-teal-800">
              Log In
            </Link>
          </p>
        </section>
      </motion.div>
    </div>
  );
}
