const defaultApiBaseUrl = "http://localhost:8000";

function trimTrailingSlash(url) {
  return url ? url.replace(/\/+$/, "") : url;
}

export const API_BASE_URL = trimTrailingSlash(import.meta.env.VITE_API_BASE_URL) || defaultApiBaseUrl;
export const SUPABASE_URL = trimTrailingSlash(import.meta.env.VITE_SUPABASE_URL || "");
export const SUPABASE_ANON_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY || "";
