import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const hubUrl = process.env.VITE_HUB_URL ?? "http://localhost:8787";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Bind all interfaces so phones on the same Wi‑Fi can open Control.
    host: "0.0.0.0",
    // IPv4 LAN hosts are already allowed by Vite; keep true for hostname LAN too.
    allowedHosts: true,
    proxy: {
      // REST: same-origin `/api` + `/health` → hub (proxy targets localhost on the PC).
      // WS: the web client connects directly to hub :8787 in DEV (see hubWsBase),
      // using window.location.hostname so a phone hits ws://<lan-ip>:8787.
      // Keep ws:true as a fallback if VITE_HUB_WS_URL forces proxy path.
      "/health": {
        target: hubUrl,
        changeOrigin: true,
      },
      "/api": {
        target: hubUrl,
        changeOrigin: true,
        ws: true,
        timeout: 60_000,
        proxyTimeout: 60_000,
      },
    },
  },
  preview: {
    port: 5173,
    host: "0.0.0.0",
    allowedHosts: true,
  },
});
