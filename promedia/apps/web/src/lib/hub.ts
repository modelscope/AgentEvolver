import type {
  CreateSessionResponse,
  JoinSessionResponse,
  ClientRole,
} from "@promedia/shared";

/** Default hub HTTP port (dev). Keep in sync with apps/hub and README. */
export const HUB_DEV_PORT = 8787;

/**
 * REST base. Default: same-origin (Vite proxies `/api` → hub).
 * Set `VITE_HUB_URL` only when calling the hub without the proxy.
 */
export function hubHttpBase(): string {
  const env = import.meta.env.VITE_HUB_URL as string | undefined;
  if (env && env.length > 0) {
    return env.replace(/\/$/, "");
  }
  return "";
}

/**
 * WebSocket base URL.
 *
 * Prefer a direct hub WS in Vite dev — proxying `/api` WebSockets through Vite
 * is flaky on some Mac setups. REST stays same-origin via the proxy.
 *
 * Override with `VITE_HUB_WS_URL` (e.g. `ws://192.168.1.10:8787`) or
 * `VITE_HUB_URL` (http→ws rewrite) when needed.
 */
export function hubWsBase(): string {
  const wsEnv = import.meta.env.VITE_HUB_WS_URL as string | undefined;
  if (wsEnv && wsEnv.length > 0) {
    return wsEnv.replace(/\/$/, "");
  }
  const http = hubHttpBase();
  if (http) {
    return http.replace(/^http/, "ws");
  }
  if (import.meta.env.DEV) {
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    const host = window.location.hostname || "localhost";
    return `${proto}://${host}:${HUB_DEV_PORT}`;
  }
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.host}`;
}

export function isHubUnreachableError(err: unknown): boolean {
  if (err instanceof TypeError) {
    return true;
  }
  const msg = String(err ?? "");
  return (
    /failed to fetch/i.test(msg) ||
    /networkerror/i.test(msg) ||
    /load failed/i.test(msg) ||
    /network request failed/i.test(msg) ||
    /econnrefused/i.test(msg)
  );
}

export type HubHealth = {
  ok: boolean;
  mode?: string;
  listenHost?: string;
  lanAddresses?: string[];
};

export async function probeHubHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${hubHttpBase()}/health`, {
      method: "GET",
      cache: "no-store",
    });
    return res.ok;
  } catch {
    return false;
  }
}

/** Hub /health — includes LAN IPs when mode=lan for phone Control URL. */
export async function fetchHubHealth(): Promise<HubHealth | null> {
  try {
    const res = await fetch(`${hubHttpBase()}/health`, {
      method: "GET",
      cache: "no-store",
    });
    if (!res.ok) return null;
    return (await res.json()) as HubHealth;
  } catch {
    return null;
  }
}

/** True when the page is opened via loopback (PC), not via LAN IP on a phone. */
export function isLoopbackPageHost(): boolean {
  const host = (window.location.hostname || "").toLowerCase();
  return (
    host === "localhost" ||
    host === "127.0.0.1" ||
    host === "[::1]" ||
    host === "::1"
  );
}

/**
 * Control URL(s) a phone should open on the same Wi‑Fi.
 * Uses current Vite port + hub-reported LAN IPs.
 */
export function buildLanControlUrls(lanAddresses: string[]): string[] {
  const port =
    window.location.port ||
    (window.location.protocol === "https:" ? "443" : "80");
  const path = "/control";
  return lanAddresses
    .filter((ip) => typeof ip === "string" && ip.length > 0)
    .map((ip) => `http://${ip}:${port}${path}`);
}

export async function createSession(
  mode: "lan" | "cloud" | "portable" = "lan",
): Promise<CreateSessionResponse> {
  let res: Response;
  try {
    res = await fetch(`${hubHttpBase()}/api/v1/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    });
  } catch (err) {
    throw new Error("HUB_UNREACHABLE", { cause: err });
  }
  if (!res.ok) {
    throw new Error(`create failed: ${res.status}`);
  }
  return (await res.json()) as CreateSessionResponse;
}

export async function joinSession(opts: {
  code?: string;
  accessToken?: string;
  role: ClientRole;
}): Promise<JoinSessionResponse> {
  const body =
    opts.code !== undefined && opts.code.length > 0
      ? { code: opts.code, role: opts.role }
      : { accessToken: opts.accessToken, role: opts.role };

  let res: Response;
  try {
    res = await fetch(`${hubHttpBase()}/api/v1/sessions/join`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (err) {
    throw new Error("HUB_UNREACHABLE", { cause: err });
  }
  if (!res.ok) {
    throw new Error(`join failed: ${res.status}`);
  }
  return (await res.json()) as JoinSessionResponse;
}

export function buildWsUrl(
  sessionId: string,
  role: ClientRole,
  wsTicket: string,
): string {
  const q = new URLSearchParams({
    sessionId,
    role,
    credential: wsTicket,
  });
  return `${hubWsBase()}/api/v1/ws?${q.toString()}`;
}
