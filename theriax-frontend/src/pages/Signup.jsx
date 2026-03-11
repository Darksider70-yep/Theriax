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

  const featuresList = [
    { icon: "💾", text: "Store AI predictions with full context" },
    { icon: "📈", text: "Visualize patterns over time" },
    { icon: "⚡", text: "Instant AI-powered search" },
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
      {/* Premium Background Gradients */}
      <div className="fixed inset-0 -z-10 overflow-hidden pointer-events-none">
        {/* Multiple Animated Gradient Orbs */}
        <div className="absolute -top-40 -right-40 w-96 h-96 bg-gradient-to-br from-tertiary-400/25 to-primary-400/15 rounded-full blur-3xl animate-float"></div>
        <div className="absolute -bottom-32 -left-48 w-80 h-80 bg-gradient-to-tr from-secondary-400/20 to-tertiary-400/15 rounded-full blur-3xl animate-float" style={{ animationDelay: "2s" }}></div>
        <div className="absolute top-1/4 left-1/3 w-72 h-72 bg-gradient-to-br from-primary-400/15 to-secondary-400/10 rounded-full blur-3xl animate-float" style={{ animationDelay: "4s" }}></div>

        {/* Gradient Accent Lines */}
        <div className="absolute bottom-0 right-0 left-0 h-px bg-gradient-to-l from-transparent via-tertiary-400/20 to-transparent"></div>
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
          className="theriax-auth-brand bg-gradient-to-br from-tertiary-600 via-orange-500 to-secondary-700 p-8 text-white sm:p-10 flex flex-col justify-between"
        >
          <div>
            <motion.p variants={itemVariants} className="text-xs font-semibold uppercase tracking-[0.2em] text-orange-100">
              Theriax Intelligence
            </motion.p>
            <motion.h1
              variants={itemVariants}
              className="theriax-display mt-4 text-3xl font-extrabold leading-tight sm:text-4xl"
            >
              Join the Medical Revolution
            </motion.h1>
            <motion.p variants={itemVariants} className="mt-4 max-w-md text-sm text-orange-50 sm:text-base leading-relaxed">
              Create your intelligent workspace to track, analyze, and leverage AI-powered medicine recommendations for smarter clinical decisions.
            </motion.p>
          </div>

          <motion.ul variants={containerVariants} className="mt-8 space-y-4">
            {featuresList.map((feature, idx) => (
              <motion.li key={idx} variants={itemVariants} className="flex items-start gap-3 text-sm text-orange-50/95">
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
          className="bg-gradient-to-br from-white/96 via-secondary-50/30 to-tertiary-50/20 p-8 sm:p-10 flex flex-col justify-center relative"
        >
          {/* Subtle Background Texture */}
          <div className="absolute inset-0 opacity-40 pointer-events-none">
            <div className="absolute inset-0 bg-gradient-to-b from-transparent via-secondary-100/5 to-transparent"></div>
          </div>
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="show"
          >
            <motion.p variants={itemVariants} className="theriax-kicker">
              Get Started
            </motion.p>
            <motion.h2
              variants={itemVariants}
              className="theriax-display mt-3 text-3xl font-bold bg-gradient-to-r from-tertiary-900 to-secondary-800 bg-clip-text text-transparent"
            >
              Create Account
            </motion.h2>
            <motion.p variants={itemVariants} className="theriax-muted mt-2 text-sm">
              Join the Theriax intelligence network in seconds.
            </motion.p>

            <form onSubmit={handleSignup} className="mt-7 space-y-5">
              <motion.div variants={itemVariants}>
                <FormInput
                  id="signup-email"
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
                  id="signup-password"
                  label="Password"
                  type="password"
                  placeholder="••••••••••••"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  autoComplete="new-password"
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
                <span>{loading ? "Creating Account..." : "Sign Up"}</span>
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
              Already have an account?{" "}
              <Link to="/" className="font-semibold text-tertiary-700 hover:text-tertiary-800 transition-colors">
                Sign in
              </Link>
            </motion.p>
          </motion.div>
        </motion.section>
      </motion.div>
    </div>
  );
}
