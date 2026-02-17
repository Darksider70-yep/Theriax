import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Card } from "../components/Card";
import api from "../utils/api";

const MotionDiv = motion.div;
const MotionTr = motion.tr;

function getTokenPayload(token) {
  try {
    const payload = token.split(".")[1];
    return JSON.parse(atob(payload));
  } catch {
    return null;
  }
}

export default function Dashboard() {
  const [user, setUser] = useState(null);
  const [topMeds, setTopMeds] = useState([]);
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searching, setSearching] = useState(false);
  const [visibleCount, setVisibleCount] = useState(7);
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
        const [topMedsRes, logsRes] = await Promise.all([
          api.get("/top-medicines"),
          api.get("/dashboard-logs"),
        ]);

        setTopMeds(topMedsRes.data || []);
        setLogs(logsRes.data || []);

        const payload = getTokenPayload(token);
        setUser({ email: payload?.email || "User" });
      } catch {
        setError("Failed to load dashboard data.");
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [navigate]);

  const handleLogout = () => {
    localStorage.removeItem("access_token");
    localStorage.removeItem("refresh_token");
    navigate("/");
  };

  const handleSearch = () => {
    setSearching(true);
    navigate("/ai-search");
  };

  const handleLoadMore = () => {
    setLoadingMore(true);
    setTimeout(() => {
      setVisibleCount((prev) => prev + 7);
      setLoadingMore(false);
    }, 350);
  };

  const severityBadge = (severity) => {
    const base = "text-white px-2 py-1 rounded text-xs";
    if (severity === "high") return <span className={`${base} bg-red-500`}>High</span>;
    if (severity === "medium") return <span className={`${base} bg-yellow-500`}>Medium</span>;
    return <span className={`${base} bg-green-500`}>Low</span>;
  };

  if (loading) return <div className="text-center mt-10 text-lg">Loading dashboard...</div>;
  if (error) return <div className="text-center mt-10 text-red-600">{error}</div>;

  return (
    <div className="min-h-screen bg-gray-50 p-6 space-y-8 w-full">
      <MotionDiv
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white shadow rounded-2xl px-6 py-4 flex flex-col sm:flex-row sm:items-center sm:justify-between"
      >
        <div>
          <h1 className="text-2xl font-bold text-gray-800">Welcome, {user?.email}</h1>
          <p className="text-sm text-gray-500">Your personalized AI dashboard</p>
        </div>
        <button
          onClick={handleLogout}
          className="mt-4 sm:mt-0 px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition"
        >
          Logout
        </button>
      </MotionDiv>

      <MotionDiv initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}>
        <Card title="Top Prescribed Medicines" className="mb-6 overflow-hidden">
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={topMeds}>
              <XAxis dataKey="medicine" />
              <YAxis allowDecimals={false} />
              <Tooltip contentStyle={{ backgroundColor: "#fff", borderRadius: "8px", fontSize: "0.9rem" }} />
              <Bar dataKey="count" fill="#4ade80" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </MotionDiv>

      <MotionDiv initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <Card title="Recent AI Recommendations">
          <div className="flex justify-between items-center mb-4">
            <div />
            <button
              onClick={handleSearch}
              disabled={searching}
              className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 disabled:opacity-60"
            >
              {searching ? "Opening..." : "Search AI"}
            </button>
          </div>

          <div className="overflow-x-auto rounded-lg border border-gray-200">
            <table className="min-w-full text-sm text-gray-800">
              <thead className="bg-gray-100 text-left font-medium border-b">
                <tr>
                  <th className="p-3">Condition</th>
                  <th className="p-3">Symptoms</th>
                  <th className="p-3">Medicine</th>
                  <th className="p-3">Severity</th>
                  <th className="p-3">Confidence</th>
                  <th className="p-3">Timestamp</th>
                </tr>
              </thead>
              <tbody>
                <AnimatePresence initial={false}>
                  {logs.slice(0, visibleCount).map((log) => {
                    const confidenceText =
                      typeof log.confidence === "number" ? `${(log.confidence * 100).toFixed(1)}%` : "N/A";

                    return (
                      <MotionTr
                        key={log.id}
                        initial={{ opacity: 0, y: 5 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -5 }}
                        transition={{ duration: 0.25 }}
                        className="border-t even:bg-gray-50 hover:bg-gray-100 transition"
                      >
                        <td className="p-3">{log.condition || "N/A"}</td>
                        <td className="p-3 max-w-sm truncate">{log.symptoms || "N/A"}</td>
                        <td className="p-3">{log.predicted_medicine || log.medicine || "N/A"}</td>
                        <td className="p-3">{severityBadge(log.severity)}</td>
                        <td className="p-3">{confidenceText}</td>
                        <td className="p-3">
                          {log.timestamp
                            ? new Date(log.timestamp).toLocaleString(undefined, {
                                dateStyle: "short",
                                timeStyle: "short",
                              })
                            : "N/A"}
                        </td>
                      </MotionTr>
                    );
                  })}
                </AnimatePresence>
              </tbody>
            </table>

            {visibleCount < logs.length && (
              <div className="p-4 text-center">
                <button
                  onClick={handleLoadMore}
                  className="px-4 py-2 bg-gray-200 rounded-lg hover:bg-gray-300 disabled:opacity-50 transition"
                  disabled={loadingMore}
                >
                  {loadingMore ? "Loading..." : "Load More"}
                </button>
              </div>
            )}
          </div>
        </Card>
      </MotionDiv>
    </div>
  );
}
