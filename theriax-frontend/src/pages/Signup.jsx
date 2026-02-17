import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { FormInput } from "../components/Forminput";
import api from "../utils/api";

export default function Signup() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState("");
  const [success, setSuccess] = useState(false);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSignup = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      await api.post("/signup", { email, password });
      setMessage("Signup successful. Redirecting to login...");
      setSuccess(true);
    } catch (err) {
      setMessage(err.response?.data?.detail || "Signup failed");
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
    <div className="min-h-screen bg-gradient-to-br from-green-100 to-teal-200 flex items-center justify-center px-4">
      <form onSubmit={handleSignup} className="bg-white p-8 rounded-2xl shadow-2xl w-full max-w-sm">
        <h2 className="text-3xl font-bold text-center text-green-700 mb-6">Create Your Theriax Account</h2>

        <FormInput
          type="email"
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />

        <FormInput
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />

        <button
          disabled={loading}
          className="w-full bg-green-600 text-white p-3 rounded-lg hover:bg-green-700 transition disabled:opacity-60"
        >
          {loading ? "Creating account..." : "Sign Up"}
        </button>

        {message && <p className="mt-4 text-sm text-center text-gray-700">{message}</p>}

        <p className="mt-4 text-sm text-center">
          Already have an account?{" "}
          <Link to="/" className="text-blue-500 hover:underline">
            Log In
          </Link>
        </p>
      </form>
    </div>
  );
}
