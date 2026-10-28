#!/usr/bin/env node
/**
 * Import Bible XML/XMM/PDF from the local operator folder into the hub library.
 *
 * Place files in: promedia/data/library/bibles/import/
 * (gitignored — do NOT commit copyrighted Bible texts)
 *
 * Usage (hub must be running, default :8787):
 *   node scripts/import-bibles.mjs
 *   HUB_URL=http://127.0.0.1:8787 node scripts/import-bibles.mjs
 */
import { mkdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const defaultImportDir = path.resolve(
  __dirname,
  "../data/library/bibles/import",
);
const importDir =
  process.env.BIBLE_IMPORT_DIR?.trim() || defaultImportDir;

// Ensure drop folder exists before the hub scans (Finder-friendly; no manual mkdir).
await mkdir(importDir, { recursive: true });

const hub = (process.env.HUB_URL || "http://127.0.0.1:8787").replace(/\/$/, "");
const url = `${hub}/api/v1/library/bibles/import-folder`;

const res = await fetch(url, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({}),
});
const body = await res.json().catch(() => ({}));
if (!res.ok) {
  console.error("Import failed:", body.message || body.error || res.status);
  process.exit(1);
}

console.log(`Folder: ${body.folder}`);
console.log(`Imported: ${body.importedCount}`);
for (const item of body.imported ?? []) {
  console.log(
    `  ✓ ${item.shortLabel || item.versionId} — ${item.title} (${item.verseCount ?? "?"} verses)`,
  );
}
if (body.skippedCount) {
  console.log(`Skipped: ${body.skippedCount}`);
  for (const s of body.skipped ?? []) {
    console.log(`  ✗ ${s.file}: ${s.message}`);
  }
}
