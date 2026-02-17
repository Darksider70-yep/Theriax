// src/utils/authHelpers.js
export function logoutUser() {
    localStorage.removeItem("access_token");
    localStorage.removeItem("refresh_token");
    window.location.href = "/";
  }
  