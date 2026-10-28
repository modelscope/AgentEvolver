import { serve } from "@hono/node-server";
import { Hono } from "hono";
import { cors } from "hono/cors";
import type { Server } from "node:http";
import { SHARED_PACKAGE_VERSION } from "@promedia/shared";
import { setSessionBroadcaster } from "./broadcast.js";
import { listLanIPv4, resolveHubListenHost } from "./lan.js";
import { ensureLibraryLayout } from "./library-store.js";
import { createApiRoutes } from "./routes.js";
import { ensureSubtitleDirs } from "./subtitle-paths.js";
import { WsHub } from "./ws-hub.js";

const app = new Hono();
const port = Number(process.env.PORT ?? 8787);
const deployMode = (process.env.PROMEDIA_MODE ?? "lan").toLowerCase();
const listenHost = resolveHubListenHost(deployMode);
const lanAddresses = listLanIPv4();

// Create data/library/{files,bibles/import,songs/import} on boot so Finder
// shows drop folders after first launch (no manual mkdir).
try {
  const layout = await ensureLibraryLayout();
  console.log(`[hub] library ready: ${layout.root}`);
  console.log(`[hub] Bible import folder: ${layout.bibleImport}`);
  console.log(`[hub] Song import folder: ${layout.songImport}`);
} catch (err) {
  console.error(
    "[hub] WARN: no se pudo crear la carpeta de biblioteca:",
    err instanceof Error ? err.message : err,
  );
}

try {
  const sub = await ensureSubtitleDirs();
  console.log(`[hub] subtitle models: ${sub.models}`);
  console.log(`[hub] youtube-cache: ${sub.youtubeCache}`);
} catch (err) {
  console.error(
    "[hub] WARN: no se pudo crear data/models|youtube-cache (subtítulos):",
    err instanceof Error ? err.message : err,
  );
}

app.use(
  "*",
  cors({
    origin: "*",
    allowMethods: ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allowHeaders: ["Content-Type", "Authorization"],
  }),
);

app.get("/health", (c) =>
  c.json({
    ok: true,
    service: "promedia-hub",
    shared: SHARED_PACKAGE_VERSION,
    mode: deployMode,
    listenHost,
    lanAddresses,
    mediaAdapter: process.env.MEDIA_ADAPTER ?? (deployMode === "cloud" ? "s3-compatible" : "local"),
  }),
);

app.get("/", (c) =>
  c.json({
    name: "ProMedia Hub",
    status: "hub-ws",
    health: "/health",
    api: "/api/v1",
    note: "Sessions are in-memory; hub restart clears them.",
  }),
);

app.route("/api/v1", createApiRoutes());

function exitPortInUse(): never {
  console.error(
    `[hub] ERROR: el puerto ${port} ya está en uso (EADDRINUSE).`,
  );
  console.error(
    `[hub] Hay otra instancia de ProMedia (o un hub viejo) escuchando en :${port}.`,
  );
  console.error(
    `[hub] Cerrá esa ventana / proceso, o liberá el puerto:\n` +
      `  lsof -nP -iTCP:${port} -sTCP:LISTEN\n` +
      `  kill <PID>\n` +
      `En Mac, start-mac.command intenta cerrar procesos node/promedia en 8787 y 5173.`,
  );
  process.exit(1);
}

const server = serve(
  {
    fetch: app.fetch,
    port,
    hostname: listenHost,
  },
  (info) => {
    // Only log after a successful bind — never before serve() listens.
    console.log(
      `ProMedia hub listening on http://localhost:${info.port} (mode=${deployMode}, bind=${listenHost})`,
    );
    if (lanAddresses.length > 0) {
      for (const ip of lanAddresses) {
        console.log(
          `[hub] LAN API/WS → http://${ip}:${info.port}  (teléfono: Control en :5173, misma IP)`,
        );
      }
      console.log(
        `[hub] Control en el teléfono (misma Wi‑Fi): http://${lanAddresses[0]}:5173/control`,
      );
    } else {
      console.log(
        "[hub] Sin IPv4 LAN detectada — Control remoto necesita Wi‑Fi / Ethernet.",
      );
    }
  },
);

server.on("error", (err: NodeJS.ErrnoException) => {
  if (err.code === "EADDRINUSE") {
    exitPortInUse();
  }
  console.error("[hub] ERROR al escuchar:", err.message || err);
  process.exit(1);
});

// Attach WS upgrade handler (ticket auth on /api/v1/ws)
const hub = new WsHub(server as Server);
setSessionBroadcaster((sessionId, message) => hub.broadcast(sessionId, message));
