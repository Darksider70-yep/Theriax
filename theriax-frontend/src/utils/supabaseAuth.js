// src/utils/supabaseAuth.js
export function parseHashFromURL() {
    const hash = window.location.hash.substring(1); // remove '#'
    const params = new URLSearchParams(hash);
    return {
      access_token: params.get("access_token"),
      refresh_token: params.get("refresh_token"),
      expires_in: params.get("expires_in"),
      token_type: params.get("token_type"),
      type: params.get("type"),
    };
  }
  