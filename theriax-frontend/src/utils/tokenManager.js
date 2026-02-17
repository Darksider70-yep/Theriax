import { SUPABASE_ANON_KEY, SUPABASE_URL } from "../config";

export async function refreshAccessToken() {
  const refreshToken = localStorage.getItem("refresh_token");
  if (!refreshToken || !SUPABASE_URL) {
    return false;
  }

  const headers = {
    "Content-Type": "application/json",
  };

  if (SUPABASE_ANON_KEY) {
    headers.apikey = SUPABASE_ANON_KEY;
  }

  const res = await fetch(`${SUPABASE_URL}/auth/v1/token?grant_type=refresh_token`, {
    method: "POST",
    headers,
    body: JSON.stringify({ refresh_token: refreshToken }),
  });

  if (!res.ok) {
    localStorage.removeItem("access_token");
    localStorage.removeItem("refresh_token");
    return false;
  }

  const data = await res.json();
  if (data.access_token) {
    localStorage.setItem("access_token", data.access_token);
    localStorage.setItem("refresh_token", data.refresh_token);
    return true;
  }

  return false;
}
