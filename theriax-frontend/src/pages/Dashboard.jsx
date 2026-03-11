import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Card } from "../components/Card";
import WorkspaceShell from "../components/WorkspaceShell";
import api from "../utils/api";

const MotionTr = motion.tr;

// Icons for stats
const statIcons = {
  predictions: "📊",
  confidence: "✨",
  severity: "⚠️",
  conditions: "🔍",
};

function toSeverityClass(severity) {
  const level = (severity || "").toLowerCase();
  if (level === "high") return "theriax-pill theriax-pill-high";
  if (level === "medium") return "theriax-pill theriax-pill-medium";
  return "theriax-pill theriax-pill-low";
}

function formatConfidence(confidence) {
  const numericConfidence = Number(confidence);
  if (!Number.isFinite(numericConfidence)) return "N/A";
  return `${(numericConfidence * 100).toFixed(1)}%`;
}

function formatTimestamp(timestamp) {
  if (!timestamp) return "N/A";
  try {
    return new Date(timestamp).toLocaleString(undefined, {
      dateStyle: "medium",
      timeStyle: "short",
    });
  } catch {
    return "N/A";
  }
}

export default function Dashboard() {
  const [topMeds, setTopMeds] = useState([]);
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [searching, setSearching] = useState(false);
  const [visibleCount, setVisibleCount] = useState(8);
  const [loadingMore, setLoadingMore] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    const token = localStorage.getItem("access_token");
    if (!token) {
      navigate("/");
      return;
    }

    const fetchData = async () => {
      try {
        const [topMedsRes, logsRes] = await Promise.all([api.get("/top-medicines"), api.get("/dashboard-logs")]);

        setTopMeds(topMedsRes.data || []);
        setLogs(logsRes.data || []);
      } catch {
        setError("Failed to load dashboard data.");
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [navigate]);

  const stats = useMemo(() => {
    const numericConfidences = logs
      .map((entry) => Number(entry.confidence))
      .filter((entryConfidence) => Number.isFinite(entryConfidence));

    const avgConfidence = numericConfidences.length
      ? (numericConfidences.reduce((sum, value) => sum + value, 0) / numericConfidences.length) * 100
      : null;

    const highSeverityCount = logs.filter(
      (entry) => (entry.severity || "").toLowerCase() === "high",
    ).length;

    const uniqueConditions = new Set(
      logs.map((entry) => (entry.condition || "").trim()).filter((entryCondition) => Boolean(entryCondition)),
    ).size;

    return [
      {
        label: "Total Predictions",
        value: logs.length.toLocaleString(),
        tone: "from-primary-500/25 to-primary-700/15",
        icon: statIcons.predictions,
        color: "primary",
      },
      {
        label: "Average Confidence",
        value: avgConfidence === null ? "N/A" : `${avgConfidence.toFixed(1)}%`,
        tone: "from-secondary-500/25 to-secondary-700/15",
        icon: statIcons.confidence,
        color: "secondary",
      },
      {
        label: "High Severity Cases",
        value: highSeverityCount.toLocaleString(),
        tone: "from-red-500/25 to-red-700/15",
        icon: statIcons.severity,
        color: "danger",
      },
      {
        label: "Unique Conditions",
        value: uniqueConditions.toLocaleString(),
        tone: "from-tertiary-500/25 to-tertiary-700/15",
        icon: statIcons.conditions,
        color: "tertiary",
      },
    ];
  }, [logs]);

  const visibleLogs = logs.slice(0, visibleCount);

  const handleOpenSearch = () => {
    setSearching(true);
    navigate("/ai-search");
  };

  const handleLoadMore = () => {
    setLoadingMore(true);
    setTimeout(() => {
      setVisibleCount((currentCount) => currentCount + 8);
      setLoadingMore(false);
    }, 250);
  };

  if (loading) {
    return (
      <WorkspaceShell
        title="Dashboard Overview"
        subtitle="Loading your latest model activity and recommendation logs."
      >
        <div className="space-y-6">
          {[...Array(2)].map((_, i) => (
            <div key={i} className="h-24 bg-gradient-to-r from-gray-200 to-gray-100 rounded-16 animate-pulse"></div>
          ))}
        </div>
      </WorkspaceShell>
    );
  }

  if (error) {
    return (
      <WorkspaceShell
        title="Dashboard Overview"
        subtitle="There was a problem loading your analytics workspace."
      >
        <div className="theriax-alert theriax-alert-error">{error}</div>
      </WorkspaceShell>
    );
  }

  const containerVariants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.1, delayChildren: 0.1 }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0, transition: { duration: 0.5 } }
  };

  return (
    <WorkspaceShell
      title="Dashboard Overview"
      subtitle="Track recommendation quality, medicine trends, and recent case outcomes in one view."
    >
      {/* Background Enhancement Layer */}
      <div className="fixed -inset-0 -z-10 overflow-hidden pointer-events-none">
        <div className="absolute top-0 right-1/4 w-96 h-96 bg-gradient-to-bl from-primary-300/10 to-transparent rounded-full blur-3xl animate-float"></div>
        <div className="absolute bottom-1/4 left-0 w-80 h-80 bg-gradient-to-tr from-secondary-300/10 to-transparent rounded-full blur-3xl animate-float" style={{ animationDelay: "2s" }}></div>
      </div>
      <motion.section
        variants={containerVariants}
        initial="hidden"
        animate="show"
        className="grid gap-5 sm:grid-cols-2 xl:grid-cols-4"
      >
        {stats.map((stat, index) => (
          <motion.article
            key={stat.label}
            variants={itemVariants}
            whileHover={{ translateY: -4 }}
            className="theriax-stat-card theriax-surface-glow p-6 group cursor-pointer"
          >
            <div className="theriax-stat-card-content">
              <div className="flex items-start justify-between mb-3">
                <div className={`h-12 w-12 rounded-12 bg-gradient-to-br ${stat.tone} flex items-center justify-center text-2xl group-hover:scale-110 transition-transform duration-300`}>
                  {stat.icon}
                </div>
              </div>
              <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500 mb-1">{stat.label}</p>
              <p className="theriax-display text-3xl font-extrabold bg-gradient-to-r from-slate-900 to-slate-700 bg-clip-text text-transparent">{stat.value}</p>
            </div>
          </motion.article>
        ))}
      </motion.section>

      <motion.section
        variants={itemVariants}
        initial="hidden"
        animate="show"
        transition={{ delay: 0.3 }}
      >
        <Card title="📊 Top Prescribed Medicines" subtitle="Most frequently recommended medicines by the AI model" variant="glow">
          <div className="h-[330px]">
            {topMeds.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={topMeds} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorBar" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="rgb(15, 118, 110)" stopOpacity={0.8} />
                      <stop offset="100%" stopColor="rgb(15, 93, 85)" stopOpacity={0.6} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(19, 36, 54, 0.12)" />
                  <XAxis
                    dataKey="medicine"
                    tick={{ fontSize: 12, fill: "#415e73" }}
                    axisLine={{ stroke: "rgba(19, 36, 54, 0.2)" }}
                    tickLine={{ stroke: "rgba(19, 36, 54, 0.2)" }}
                  />
                  <YAxis
                    allowDecimals={false}
                    tick={{ fontSize: 12, fill: "#415e73" }}
                    axisLine={{ stroke: "rgba(19, 36, 54, 0.2)" }}
                    tickLine={{ stroke: "rgba(19, 36, 54, 0.2)" }}
                  />
                  <Tooltip
                    cursor={{ fill: "rgba(15, 118, 110, 0.08)" }}
                    contentStyle={{
                      borderRadius: "12px",
                      border: "2px solid rgba(15, 118, 110, 0.3)",
                      backgroundColor: "rgba(255, 255, 255, 0.98)",
                      boxShadow: "0 12px 30px rgba(10, 34, 51, 0.15)",
                    }}
                    labelStyle={{ color: "#132436" }}
                  />
                  <Bar dataKey="count" fill="url(#colorBar)" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-full items-center justify-center rounded-xl border border-dashed border-slate-300 bg-white/65">
                <p className="theriax-muted text-sm">No chart data available yet.</p>
              </div>
            )}
          </div>
        </Card>
      </motion.section>

      <motion.section
        variants={itemVariants}
        initial="hidden"
        animate="show"
        transition={{ delay: 0.4 }}
      >
        <Card
          title="🔬 Recent AI Recommendations"
          subtitle="Most recent prediction logs with confidence and severity context"
        >
          <div className="mb-6 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
            <p className="theriax-muted text-sm">
              <span className="font-semibold text-primary-700">{Math.min(visibleCount, logs.length)}</span> of <span className="font-semibold text-slate-900">{logs.length}</span> entries
            </p>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              type="button"
              onClick={handleOpenSearch}
              disabled={searching}
              className="theriax-btn theriax-btn-primary px-5 py-2.5 text-sm group"
            >
              <span>⚡</span>
              <span>{searching ? "Opening..." : "Open AI Search"}</span>
              <span className="group-hover:translate-x-1 transition-transform">→</span>
            </motion.button>
          </div>

          {logs.length === 0 ? (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="rounded-xl border-2 border-dashed border-primary-300/50 bg-gradient-to-br from-primary-50/50 to-secondary-50/30 p-12 text-center"
            >
              <p className="text-2xl mb-2">📋</p>
              <p className="theriax-muted text-sm">No recommendations logged yet. Start by using AI Search to generate predictions.</p>
            </motion.div>
          ) : (
            <>
              <div className="theriax-scroll">
                <table className="theriax-table">
                  <thead>
                    <tr>
                      <th>Condition</th>
                      <th>Symptoms</th>
                      <th>Medicine</th>
                      <th>Severity</th>
                      <th>Confidence</th>
                      <th>Timestamp</th>
                    </tr>
                  </thead>
                  <tbody>
                    <AnimatePresence initial={false}>
                      {visibleLogs.map((log, index) => (
                        <MotionTr
                          key={log.id ?? `${log.timestamp || "row"}-${index}`}
                          initial={{ opacity: 0, y: 6 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, y: -6 }}
                          transition={{ duration: 0.22 }}
                          className="hover:bg-primary-50/40 transition-all"
                        >
                          <td className="font-medium text-slate-900">{log.condition || "N/A"}</td>
                          <td className="max-w-[260px] truncate text-slate-700" title={log.symptoms || "N/A"}>
                            {log.symptoms || "N/A"}
                          </td>
                          <td className="font-semibold text-primary-700">{log.predicted_medicine || log.medicine || "N/A"}</td>
                          <td>
                            <span className={toSeverityClass(log.severity)}>{log.severity || "low"}</span>
                          </td>
                          <td>
                            <span className="text-sm font-semibold text-secondary-700">{formatConfidence(log.confidence)}</span>
                          </td>
                          <td className="text-xs text-slate-500">{formatTimestamp(log.timestamp)}</td>
                        </MotionTr>
                      ))}
                    </AnimatePresence>
                  </tbody>
                </table>
              </div>

              {visibleCount < logs.length && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="mt-6 text-center"
                >
                  <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    type="button"
                    onClick={handleLoadMore}
                    disabled={loadingMore}
                    className="theriax-btn theriax-btn-ghost px-6 py-2.5 text-sm"
                  >
                    {loadingMore ? "⏳ Loading..." : "📥 Load More Entries"}
                  </motion.button>
                </motion.div>
              )}
            </>
          )}
        </Card>
      </motion.section>
    </WorkspaceShell>
  );
}
