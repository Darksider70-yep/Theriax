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
        tone: "from-teal-500/20 to-teal-700/15",
      },
      {
        label: "Average Confidence",
        value: avgConfidence === null ? "N/A" : `${avgConfidence.toFixed(1)}%`,
        tone: "from-amber-500/20 to-orange-700/15",
      },
      {
        label: "High Severity Cases",
        value: highSeverityCount.toLocaleString(),
        tone: "from-red-500/20 to-rose-700/15",
      },
      {
        label: "Unique Conditions",
        value: uniqueConditions.toLocaleString(),
        tone: "from-cyan-500/20 to-sky-700/15",
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
        <Card>
          <p className="theriax-muted text-sm">Loading dashboard data...</p>
        </Card>
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

  return (
    <WorkspaceShell
      title="Dashboard Overview"
      subtitle="Track recommendation quality, medicine trends, and recent case outcomes in one view."
    >
      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {stats.map((stat, index) => (
          <motion.article
            key={stat.label}
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.06, duration: 0.3 }}
            className="theriax-surface p-5"
          >
            <div className={`mb-3 h-2 w-16 rounded-full bg-gradient-to-r ${stat.tone}`} />
            <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">{stat.label}</p>
            <p className="theriax-display mt-2 text-2xl font-extrabold text-slate-900">{stat.value}</p>
          </motion.article>
        ))}
      </section>

      <motion.section
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.18, duration: 0.35 }}
      >
        <Card title="Top Prescribed Medicines" subtitle="Frequency of model recommendations by medicine">
          <div className="h-[330px]">
            {topMeds.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={topMeds} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
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
                      border: "1px solid rgba(19, 36, 54, 0.14)",
                      backgroundColor: "rgba(255, 255, 255, 0.96)",
                      boxShadow: "0 12px 30px rgba(10, 34, 51, 0.13)",
                    }}
                  />
                  <Bar dataKey="count" fill="#0f766e" radius={[8, 8, 0, 0]} />
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
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.24, duration: 0.35 }}
      >
        <Card
          title="Recent AI Recommendations"
          subtitle="Most recent prediction logs with confidence and severity context"
        >
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
            <p className="theriax-muted text-sm">
              Showing {Math.min(visibleCount, logs.length)} of {logs.length} entries
            </p>
            <button
              type="button"
              onClick={handleOpenSearch}
              disabled={searching}
              className="theriax-btn theriax-btn-primary px-4 py-2 text-sm"
            >
              {searching ? "Opening..." : "Open AI Search"}
            </button>
          </div>

          {logs.length === 0 ? (
            <div className="rounded-xl border border-dashed border-slate-300 bg-white/65 p-6 text-center">
              <p className="theriax-muted text-sm">No recommendations logged yet.</p>
            </div>
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
                        >
                          <td>{log.condition || "N/A"}</td>
                          <td className="max-w-[260px] truncate" title={log.symptoms || "N/A"}>
                            {log.symptoms || "N/A"}
                          </td>
                          <td>{log.predicted_medicine || log.medicine || "N/A"}</td>
                          <td>
                            <span className={toSeverityClass(log.severity)}>{log.severity || "low"}</span>
                          </td>
                          <td>{formatConfidence(log.confidence)}</td>
                          <td>{formatTimestamp(log.timestamp)}</td>
                        </MotionTr>
                      ))}
                    </AnimatePresence>
                  </tbody>
                </table>
              </div>

              {visibleCount < logs.length && (
                <div className="mt-4 text-center">
                  <button
                    type="button"
                    onClick={handleLoadMore}
                    disabled={loadingMore}
                    className="theriax-btn theriax-btn-ghost px-4 py-2 text-sm"
                  >
                    {loadingMore ? "Loading..." : "Load More"}
                  </button>
                </div>
              )}
            </>
          )}
        </Card>
      </motion.section>
    </WorkspaceShell>
  );
}
