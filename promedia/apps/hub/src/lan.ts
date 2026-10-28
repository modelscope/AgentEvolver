import os from "node:os";

/**
 * Non-internal IPv4 addresses for LAN remote Control / hub ads.
 * Skips link-local (169.254.x) and loopback.
 */
export function listLanIPv4(): string[] {
  const out: string[] = [];
  const ifaces = os.networkInterfaces();
  for (const addrs of Object.values(ifaces)) {
    if (!addrs) continue;
    for (const a of addrs) {
      const family = String(a.family);
      const isV4 = family === "IPv4" || family === "4";
      if (!isV4 || a.internal) continue;
      if (a.address.startsWith("169.254.")) continue;
      if (!out.includes(a.address)) out.push(a.address);
    }
  }
  return out;
}

/**
 * Hub listen host. Default `0.0.0.0` so phones on the same Wi‑Fi can reach
 * API/WS when `PROMEDIA_MODE=lan` (the product default). Override with
 * `HUB_HOST=127.0.0.1` for localhost-only.
 */
export function resolveHubListenHost(mode: string): string {
  const fromEnv = (process.env.HUB_HOST ?? "").trim();
  if (fromEnv) return fromEnv;
  const m = mode.toLowerCase();
  if (m === "lan" || m === "portable" || m === "cloud") {
    return "0.0.0.0";
  }
  return "0.0.0.0";
}
