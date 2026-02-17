import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { FormInput } from "../components/Forminput";
import api from "../utils/api";

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const res = await api.post("/login", { email, password });
      localStorage.setItem("access_token", res.data.access_token);
      if (res.data.refresh_token) {
        localStorage.setItem("refresh_token", res.data.refresh_token);
      }
      setMessage("Logged in successfully");
      setTimeout(() => navigate("/dashboard"), 800);
    } catch (err) {
      setMessage(err.response?.data?.detail || "Login failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-100 to-indigo-200 flex items-center justify-center px-4">
      <form onSubmit={handleLogin} className="bg-white p-8 rounded-2xl shadow-2xl w-full max-w-sm">
        <h2 className="text-3xl font-bold text-center text-blue-700 mb-6">Welcome Back</h2>

        <FormInput type="email" placeholder="Email" value={email} onChange={(e) => setEmail(e.target.value)} />
        <FormInput type="password" placeholder="Password" value={password} onChange={(e) => setPassword(e.target.value)} />

        <button
          disabled={loading}
          className="w-full bg-blue-600 text-white p-3 rounded-lg hover:bg-blue-700 transition disabled:opacity-60"
        >
          {loading ? "Logging in..." : "Log In"}
        </button>

        {message && <p className="mt-4 text-sm text-center text-gray-700">{message}</p>}

        <p className="mt-6 text-sm text-center">
          Don't have an account?{" "}
          <Link to="/signup" className="text-blue-500 hover:underline font-medium">
            Sign Up
          </Link>
        </p>
      </form>
    </div>
  );
}
