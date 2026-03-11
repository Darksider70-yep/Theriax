import { motion } from "framer-motion";
import { Link, useLocation, useNavigate } from "react-router-dom";

const navLinks = [
  { to: "/dashboard", label: "Dashboard", icon: "📊" },
  { to: "/ai-search", label: "AI Search", icon: "⚡" },
];

function readEmailFromToken(token) {
  try {
    const payload = JSON.parse(atob(token.split(".")[1]));
    return payload?.email || "User";
  } catch {
    return "User";
  }
}

export default function WorkspaceShell({ title, subtitle, children }) {
  const navigate = useNavigate();
  const location = useLocation();
  const token = localStorage.getItem("access_token");
  const email = readEmailFromToken(token || "");

  const handleLogout = () => {
    localStorage.removeItem("access_token");
    localStorage.removeItem("refresh_token");
    navigate("/");
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.05, delayChildren: 0.05 }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: -10 },
    show: { opacity: 1, y: 0, transition: { duration: 0.4 } }
  };

  return (
    <div className="min-h-screen px-4 py-6 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-7xl space-y-6">
        <motion.header
          variants={itemVariants}
          initial="hidden"
          animate="show"
          className="theriax-surface theriax-surface-strong px-6 py-5 sm:px-7 rounded-20 backdrop-blur-xl"
        >
          <div className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
            <div className="flex-1">
              <motion.p
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.1 }}
                className="theriax-kicker"
              >
                🏥 Theriax Workspace
              </motion.p>
              <motion.h1
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.15 }}
                className="theriax-display mt-2 text-2xl sm:text-3xl font-extrabold bg-gradient-to-r from-primary-700 to-secondary-700 bg-clip-text text-transparent"
              >
                {title || "Clinical Intelligence Console"}
              </motion.h1>
              {subtitle && (
                <motion.p
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.2 }}
                  className="theriax-muted mt-2 text-sm max-w-2xl leading-relaxed"
                >
                  {subtitle}
                </motion.p>
              )}
            </div>

            <motion.div
              variants={containerVariants}
              initial="hidden"
              animate="show"
              className="flex flex-col sm:flex-row flex-wrap items-start sm:items-center gap-2 sm:gap-3"
            >
              <motion.nav
                variants={itemVariants}
                className="flex flex-wrap items-center gap-1 bg-gradient-to-r from-primary-50/60 to-secondary-50/40 px-2 py-2 rounded-full border border-primary-200/30 backdrop-blur-sm"
              >
                {navLinks.map((item) => {
                  const isActive =
                    location.pathname === item.to ||
                    location.pathname.startsWith(`${item.to}/`);
                  return (
                    <Link
                      key={item.to}
                      to={item.to}
                      className={`theriax-nav-link transition-all duration-300 flex items-center gap-2 px-3 py-1.5 rounded-lg ${
                        isActive
                          ? "theriax-nav-link-active bg-white/70 shadow-sm"
                          : "hover:bg-white/30"
                      }`}
                    >
                      <span className="text-base">{item.icon}</span>
                      <span className="text-sm font-semibold">{item.label}</span>
                    </Link>
                  );
                })}
              </motion.nav>

              <motion.div
                variants={itemVariants}
                className="flex items-center gap-2.5 min-w-fit"
              >
                <motion.span
                  whileHover={{ scale: 1.02 }}
                  className="rounded-full border-2 border-primary-200/50 bg-gradient-to-br from-primary-50/80 to-secondary-50/60 px-4 py-1.5 text-xs font-semibold text-primary-800 shadow-sm"
                >
                  👤 {email.split("@")[0]}
                </motion.span>

                <motion.button
                  variants={itemVariants}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  type="button"
                  onClick={handleLogout}
                  className="theriax-btn theriax-btn-danger px-3.5 py-1.5 text-sm font-semibold"
                >
                  ↪️ Logout
                </motion.button>
              </motion.div>
            </motion.div>
          </div>
        </motion.header>

        <motion.section
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.5 }}
          className="space-y-6"
        >
          {children}
        </motion.section>
      </div>
    </div>
  );
}
