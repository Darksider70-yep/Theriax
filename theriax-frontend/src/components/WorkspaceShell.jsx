import { motion } from "framer-motion";
import { Link, useLocation, useNavigate } from "react-router-dom";

const navLinks = [
  { to: "/dashboard", label: "Dashboard" },
  { to: "/ai-search", label: "AI Search" },
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

  return (
    <div className="min-h-screen px-4 py-6 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-7xl space-y-6">
        <motion.header
          initial={{ opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35 }}
          className="theriax-surface theriax-surface-strong px-5 py-4 sm:px-6"
        >
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <p className="theriax-kicker">Theriax Workspace</p>
              <h1 className="theriax-display mt-1 text-2xl font-extrabold text-slate-900">
                Clinical Intelligence Console
              </h1>
            </div>

            <div className="flex flex-wrap items-center gap-2 sm:gap-3">
              <nav className="flex flex-wrap items-center gap-2">
                {navLinks.map((item) => {
                  const isActive =
                    location.pathname === item.to ||
                    location.pathname.startsWith(`${item.to}/`);
                  return (
                    <Link
                      key={item.to}
                      to={item.to}
                      className={`theriax-nav-link ${isActive ? "theriax-nav-link-active" : ""}`}
                    >
                      {item.label}
                    </Link>
                  );
                })}
              </nav>
              <span className="rounded-full border border-slate-200 bg-white/75 px-3 py-1 text-xs font-semibold text-slate-600">
                {email}
              </span>
              <button
                type="button"
                onClick={handleLogout}
                className="theriax-btn theriax-btn-danger px-3 py-2 text-sm"
              >
                Logout
              </button>
            </div>
          </div>
        </motion.header>

        <motion.section
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.05, duration: 0.35 }}
          className="theriax-surface px-5 py-5 sm:px-6"
        >
          <h2 className="theriax-display text-2xl font-bold text-slate-900">{title}</h2>
          {subtitle && <p className="theriax-muted mt-2 max-w-3xl text-sm sm:text-base">{subtitle}</p>}
        </motion.section>

        <main className="space-y-6">{children}</main>
      </div>
    </div>
  );
}
