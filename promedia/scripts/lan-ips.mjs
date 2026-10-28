#!/usr/bin/env node
/**
 * Print non-internal IPv4 addresses (one per line) for launchers / docs.
 * Usage: node scripts/lan-ips.mjs
 */
import os from "node:os";

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
for (const ip of out) {
  console.log(ip);
}
