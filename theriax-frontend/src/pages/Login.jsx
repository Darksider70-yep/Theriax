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

  const featuresList = [
    { icon: "🧠", text: "AI-powered condition analysis" },
    { icon: "📊", text: "Real-time confidence scores" },
    { icon: "📋", text: "Detailed recommendation logs" },
  ];

  const containerVariants = {
    hidden: { opacity: 0, y: 20 },
    show: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.6, staggerChildren: 0.1 }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 10 },
    show: { opacity: 1, y: 0 }
  };

  return (
    <div className="min-h-screen px-4 py-8 sm:px-6 lg:px-8 relative overflow-hidden">
      {/* Sophisticated Background Gradients */}
      <div className="fixed inset-0 -z-10 overflow-hidden pointer-events-none">
        {/* Gradient Orbs */}
        <div className="absolute -top-40 -right-40 w-96 h-96 bg-gradient-to-br from-primary-400/25 to-secondary-400/15 rounded-full blur-3xl animate-float"></div>
        <div className="absolute -bottom-32 -left-48 w-80 h-80 bg-gradient-to-tr from-tertiary-400/20 to-primary-400/15 rounded-full blur-3xl animate-float" style={{ animationDelay: "2s" }}></div>
        <div className="absolute top-1/3 right-1/4 w-72 h-72 bg-gradient-to-bl from-secondary-400/15 to-tertiary-400/10 rounded-full blur-3xl animate-float" style={{ animationDelay: "4s" }}></div>

        {/* Gradient Lines */}
        <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-primary-400/20 to-transparent"></div>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45 }}
        className="mx-auto max-w-5xl overflow-hidden theriax-surface theriax-auth-grid relative z-10"
      >
        {/* Left Brand Section */}
        <motion.section
          variants={containerVariants}
          initial="hidden"
          animate="show"
          className="theriax-auth-brand bg-gradient-to-br from-primary-700 via-primary-600 to-secondary-700 p-8 text-white sm:p-10 flex flex-col justify-between"
        >
          <div>
            <motion.p variants={itemVariants} className="text-xs font-semibold uppercase tracking-[0.2em] text-primary-100">
              Theriax Intelligence
            </motion.p>
            <motion.h1
              variants={itemVariants}
              className="theriax-display mt-4 text-3xl font-extrabold leading-tight sm:text-4xl"
            >
              Clinical Excellence Powered by AI
            </motion.h1>
            <motion.p variants={itemVariants} className="mt-4 max-w-md text-sm text-primary-50 sm:text-base leading-relaxed">
              Access predictive medicine recommendations, evidence-backed dosage context, and real-time confidence metrics from a unified intelligent workspace.
            </motion.p>
          </div>

          <motion.ul variants={containerVariants} className="mt-8 space-y-4">
            {featuresList.map((feature, idx) => (
              <motion.li key={idx} variants={itemVariants} className="flex items-start gap-3 text-sm text-primary-50/95">
                <span className="text-xl mt-0.5">{feature.icon}</span>
                <span>{feature.text}</span>
              </motion.li>
            ))}
          </motion.ul>
        </motion.section>

        {/* Right Form Section */}
        <motion.section
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="bg-gradient-to-br from-white/96 via-primary-50/30 to-secondary-50/20 p-8 sm:p-10 flex flex-col justify-center relative"
        >
          {/* Subtle Background Pattern */}
          <div className="absolute inset-0 opacity-40 pointer-events-none">
            <div className="absolute inset-0 bg-gradient-to-b from-transparent via-primary-100/5 to-transparent"></div>
          </div>
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="show"
          >
            <motion.p variants={itemVariants} className="theriax-kicker">
              Welcome Back
            </motion.p>
            <motion.h2
              variants={itemVariants}
              className="theriax-display mt-3 text-3xl font-bold bg-gradient-to-r from-primary-900 to-secondary-800 bg-clip-text text-transparent"
            >
              Sign In
            </motion.h2>
            <motion.p variants={itemVariants} className="theriax-muted mt-2 text-sm">
              Enter your credentials to access the clinical intelligence console.
            </motion.p>

            <form onSubmit={handleLogin} className="mt-7 space-y-5">
              <motion.div variants={itemVariants}>
                <FormInput
                  id="login-email"
                  label="Email Address"
                  type="email"
                  placeholder="your.email@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  autoComplete="email"
                />
              </motion.div>

              <motion.div variants={itemVariants}>
                <FormInput
                  id="login-password"
                  label="Password"
                  type="password"
                  placeholder="••••••••••••"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  autoComplete="current-password"
                />
              </motion.div>

              <motion.button
                variants={itemVariants}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                type="submit"
                disabled={loading}
                className="theriax-btn theriax-btn-primary mt-2 w-full group"
              >
                <span>{loading ? "Signing in..." : "Sign In"}</span>
                {!loading && <span className="group-hover:translate-x-1 transition-transform">→</span>}
              </motion.button>
            </form>

            {feedback.text && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className={`mt-5 p-4 rounded-12 text-sm font-medium ${
                  feedback.type === "error"
                    ? "bg-red-50/80 text-red-700 border border-red-200/50"
                    : "bg-green-50/80 text-green-700 border border-green-200/50"
                }`}
              >
                {feedback.text}
              </motion.div>
            )}

            <motion.p variants={itemVariants} className="mt-8 text-center text-sm text-slate-600">
              New to Theriax?{" "}
              <Link to="/signup" className="font-semibold text-primary-700 hover:text-primary-800 transition-colors">
                Create an account
              </Link>
            </motion.p>
          </motion.div>
        </motion.section>
      </motion.div>
    </div>
  );
}
