#!/usr/bin/env node
/**
 * `pnpm dev` entry: build packages, then run hub + web via concurrently
 * with [hub] / [web] prefixes.
 *
 * Ready only when:
 * - our hub child logged "listening" (real bind, not a stale :8787), and
 * - /health answers while that child is still alive.
 *
 * Prints the ACTUAL Vite URL from "Local: http://localhost:PORT"
 * (ANSI-safe, line-buffered). Writes `.promedia-ready-url` as soon as
 * that URL is known so launchers can open Control without racing.
 * Also writes `.promedia-lan-url` with http://<lan-ip>:PORT/control for phones.
 */
import concurrently from "concurrently";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import { setTimeout as delay } from "node:timers/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const READY_URL_FILE = path.join(ROOT, ".promedia-ready-url");
const LAN_URL_FILE = path.join(ROOT, ".promedia-lan-url");
const HUB_URL = (process.env.HUB_URL ?? "http://127.0.0.1:8787").replace(
  /\/$/,
  "",
);
const HEALTH = `${HUB_URL}/health`;
const HUB_WAIT_MS = Number(process.env.PROMEDIA_HUB_WAIT_MS ?? 45_000);
const WEB_URL_WAIT_MS = Number(process.env.PROMEDIA_WEB_URL_WAIT_MS ?? 20_000);
const VITE_PORTS = [5173, 5174, 5175, 5176, 5177, 5178];

function listLanIPv4() {
  const out = [];
  for (const addrs of Object.values(os.networkInterfaces())) {
    if (!addrs) continue;
    for (const a of addrs) {
      const isV4 = a.family === "IPv4" || a.family === 4;
      if (!isV4 || a.internal) continue;
      if (String(a.address).startsWith("169.254.")) continue;
      if (!out.includes(a.address)) out.push(a.address);
    }
  }
  return out;
}

function clearReadyUrlFile() {
  for (const file of [READY_URL_FILE, LAN_URL_FILE]) {
    try {
      fs.unlinkSync(file);
    } catch {
      // ignore
    }
  }
}

function writeReadyUrlFile(controlUrl) {
  try {
    fs.writeFileSync(READY_URL_FILE, `${controlUrl}\n`, "utf8");
  } catch {
    // ignore — launchers still probe /control on 5173–5178
  }
}

function writeLanUrlFile(controlUrl) {
  try {
    fs.writeFileSync(LAN_URL_FILE, `${controlUrl}\n`, "utf8");
  } catch {
    // ignore
  }
}

function advertiseLanControl(localUrl) {
  let port = 5173;
  try {
    const u = new URL(localUrl);
    port = Number(u.port) || (u.protocol === "https:" ? 443 : 80);
  } catch {
    // keep 5173
  }
  const ips = listLanIPv4();
  if (ips.length === 0) {
    console.log(
      "[dev] Sin IPv4 LAN — el teléfono no podrá abrir Control hasta tener Wi‑Fi/Ethernet.",
    );
    return;
  }
  const primary = `http://${ips[0]}:${port}/control`;
  writeLanUrlFile(primary);
  console.log(`[dev] Teléfono (misma Wi‑Fi) → ${primary}`);
  for (const ip of ips.slice(1)) {
    console.log(`[dev]   también → http://${ip}:${port}/control`);
  }
}

clearReadyUrlFile();

/** Strip CSI / OSC ANSI so "Local" + ":" still match when Vite colors output. */
function stripAnsi(text) {
  return text.replace(/\u001b\[[0-9;]*[a-zA-Z]|\u001b\][^\u0007]*(?:\u0007|\u001b\\)/g, "");
}

// Vite: `Local:   http://…` (possibly with ANSI between tokens).
const LOCAL_URL_RE = /Local:\s*(https?:\/\/[^\s\u001b]+)/i;
const HUB_LISTENING_RE = /ProMedia hub listening on/i;
const HUB_EADDR_RE = /EADDRINUSE|puerto \d+ ya está en uso/i;

console.log("[dev] building packages…");
const build = spawnSync("pnpm", ["run", "build:packages"], {
  cwd: ROOT,
  env: process.env,
  stdio: "inherit",
});
if (build.status !== 0) {
  process.exit(build.status ?? 1);
}

console.log("[dev] starting hub + web (labels [hub] / [web])…");

const { result, commands } = concurrently(
  [
    {
      name: "hub",
      command: "pnpm --filter @promedia/hub dev",
      prefixColor: "cyan",
    },
    {
      name: "web",
      command: "pnpm --filter @promedia/web dev",
      prefixColor: "magenta",
    },
  ],
  {
    cwd: ROOT,
    prefix: "name",
    // Only kill siblings on failure. Do NOT kill on "success" — Vite can
    // exit 0 after an external SIGTERM; that must not tear down a healthy hub.
    killOthersOn: ["failure"],
    restartTries: 0,
  },
);

const hubCmd = commands.find((c) => c.name === "hub");
const webCmd = commands.find((c) => c.name === "web");

let hubExited = false;
let hubExitCode = null;
let hubListening = false;
let hubBindFailed = false;
let webLocalUrl = null;

function onWebLocalUrl(url) {
  const cleaned = url.replace(/\/$/, "");
  if (!cleaned || webLocalUrl === cleaned) return;
  webLocalUrl = cleaned;
  writeReadyUrlFile(`${cleaned}/control`);
}

function makeLineScanner(onLine) {
  let buf = "";
  return (chunk) => {
    buf += stripAnsi(chunk.toString());
    let nl;
    while ((nl = buf.indexOf("\n")) !== -1) {
      const line = buf.slice(0, nl);
      buf = buf.slice(nl + 1);
      onLine(line);
    }
    // Also scan the incomplete tail (Vite may flush without a trailing newline).
    if (buf) onLine(buf);
  };
}

const scanHubOutput = makeLineScanner((line) => {
  if (HUB_LISTENING_RE.test(line)) hubListening = true;
  if (HUB_EADDR_RE.test(line)) hubBindFailed = true;
});

const scanWebOutput = makeLineScanner((line) => {
  const m = line.match(LOCAL_URL_RE);
  if (m) onWebLocalUrl(m[1]);
});

if (hubCmd) {
  hubCmd.close.subscribe((ev) => {
    hubExited = true;
    hubExitCode = ev.exitCode;
  });
  hubCmd.stdout.subscribe(scanHubOutput);
  hubCmd.stderr.subscribe(scanHubOutput);
}

if (webCmd) {
  webCmd.stdout.subscribe(scanWebOutput);
  webCmd.stderr.subscribe(scanWebOutput);
}

function killAll() {
  for (const cmd of commands) {
    try {
      cmd.kill("SIGTERM");
    } catch {
      // ignore
    }
  }
}

function hubChildDead() {
  return (
    hubExited ||
    !hubCmd ||
    hubCmd.state === "exited" ||
    hubCmd.state === "errored"
  );
}

async function waitForHub(timeoutMs) {
  const start = Date.now();
  let lastErr = "";
  let sawStaleHealth = false;

  while (Date.now() - start < timeoutMs) {
    if (hubBindFailed) {
      console.error(
        "[dev] ERROR: el hub no pudo enlazar el puerto (EADDRINUSE).",
      );
      console.error(
        "[dev] Hay una instancia anterior en :8787. Cerrala o usá el launcher (start-mac / start-windows / start-linux).",
      );
      console.error(
        "[dev] Manual: lsof -nP -iTCP:8787 -sTCP:LISTEN  →  kill <PID>",
      );
      return false;
    }

    if (hubChildDead() && !hubListening) {
      console.error(
        `[dev] ERROR: el proceso hub salió antes de escuchar (code=${hubExitCode ?? hubCmd?.state}).`,
      );
      console.error(
        "[dev] No uses un /health de un hub viejo: esta instancia no arrancó.",
      );
      return false;
    }

    try {
      const res = await fetch(HEALTH, { cache: "no-store" });
      if (res.ok) {
        if (!hubListening) {
          // Old process still answering — our child has not bound yet.
          sawStaleHealth = true;
          lastErr =
            "HTTP 200 de un proceso viejo (nuestro hub aún no logueó listening)";
        } else if (hubChildDead() || hubBindFailed) {
          console.error(
            "[dev] ERROR: /health responde, pero nuestro hub hijo ya no está vivo.",
          );
          console.error(
            "[dev] Eso suele ser un hub ANTERIOR en :8787 (EADDRINUSE). Matá ese proceso y reintentá.",
          );
          return false;
        } else {
          const body = await res.json().catch(() => ({}));
          console.log(
            `[dev] hub healthy → ${HEALTH}` +
              (body?.service ? ` (${body.service})` : ""),
          );
          return true;
        }
      } else {
        lastErr = `HTTP ${res.status}`;
      }
    } catch (err) {
      lastErr = err instanceof Error ? err.message : String(err);
    }
    await delay(400);
  }

  console.error(
    `[dev] ERROR: hub did not become healthy within ${timeoutMs}ms.`,
  );
  console.error(`[dev] Last probe of ${HEALTH}: ${lastErr}`);
  if (sawStaleHealth && !hubListening) {
    console.error(
      "[dev] /health respondió, pero este hub nunca logueó “listening” — probable EADDRINUSE / proceso viejo.",
    );
  }
  console.error(
    `[dev] Two-terminal fallback:\n` +
      `  Terminal A: pnpm dev:hub\n` +
      `  Terminal B: pnpm dev:web\n` +
      `Then: curl -s ${HEALTH}`,
  );
  return false;
}

async function waitForWebLocalUrl(timeoutMs) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (webLocalUrl) return webLocalUrl;
    await delay(200);
  }
  return webLocalUrl;
}

/** If concurrently stdout missed "Local:", probe Vite ports for /control. */
async function probeViteControlUrl() {
  for (const port of VITE_PORTS) {
    const control = `http://127.0.0.1:${port}/control`;
    try {
      const res = await fetch(control, { cache: "no-store" });
      if (res.ok) {
        return `http://localhost:${port}`;
      }
    } catch {
      // try next
    }
  }
  return null;
}

const healthy = await waitForHub(HUB_WAIT_MS);
if (!healthy) {
  killAll();
  clearReadyUrlFile();
  process.exit(1);
}

let localUrl = await waitForWebLocalUrl(WEB_URL_WAIT_MS);
if (!localUrl) {
  localUrl = await probeViteControlUrl();
  if (localUrl) onWebLocalUrl(localUrl);
}

if (localUrl) {
  const control = `${localUrl.replace(/\/$/, "")}/control`;
  writeReadyUrlFile(control);
  console.log(`[dev] ready — Control ${control} · health ${HEALTH}`);
  advertiseLanControl(localUrl);
  if (!/:(5173)(\/|$)/.test(localUrl)) {
    console.log(
      `[dev] Nota: Vite no usó 5173 — abrí exactamente ${control} (y la URL LAN del mismo puerto).`,
    );
  }
} else {
  console.log(
    `[dev] ready — hub OK (${HEALTH}). Todavía no vi “Local:” de Vite.`,
  );
  console.log(
    `[dev] Buscá en los logs [web] la línea Local: http://localhost:PUERTO — NO asumas :5173.`,
  );
}

try {
  await result;
} catch {
  clearReadyUrlFile();
  process.exit(1);
}
clearReadyUrlFile();
