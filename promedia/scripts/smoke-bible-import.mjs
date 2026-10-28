#!/usr/bin/env node
/**
 * Smoke-test Bible parsers against optional local samples + the tiny fixture.
 * Does not write copyrighted texts into the repo.
 *
 *   node scripts/smoke-bible-import.mjs
 *   BIBLE_SAMPLES=/path/to/xmls node scripts/smoke-bible-import.mjs
 */
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const hubRoot = path.resolve(__dirname, "../apps/hub");
const require = createRequire(path.join(hubRoot, "package.json"));

// Prefer compiled dist if present; else tsx-register via dynamic import of source through tsx
async function loadImporter() {
  try {
    return await import(path.join(hubRoot, "dist/bible-import.js"));
  } catch {
    // run via: pnpm exec tsx scripts/smoke-bible-import.mjs — or import source
    return await import(path.join(hubRoot, "src/bible-import.ts"));
  }
}

const { importBibleFile } = await loadImporter();

const fixture = path.join(
  hubRoot,
  "src/bible/fixtures/sample-bcv.xml",
);
const samplesDir =
  process.env.BIBLE_SAMPLES ||
  "/cursor/stores/bc-1376f14d-6310-4ae9-aaf0-466e07c4b67f/internal/bible-uploads";

const files = [fixture];
try {
  const { readdir } = await import("node:fs/promises");
  const names = await readdir(samplesDir);
  for (const n of names.sort()) {
    if (/\.(xml|xmm|pdf)$/i.test(n)) {
      files.push(path.join(samplesDir, n));
    }
  }
} catch {
  console.log("(no sample dir; fixture only)");
}

let failed = 0;
for (const file of files) {
  const name = path.basename(file);
  const bytes = await readFile(file);
  const result = await importBibleFile({ bytes, originalName: name });
  if (!result.ok) {
    console.log(`FAIL ${name}: ${result.message}`);
    failed += 1;
    continue;
  }
  const c = result.canon;
  console.log(
    `OK   ${name}: ${c.shortLabel} (${c.versionId}) ${c.verseCount} verses / ${c.bookCount} books [${c.format}]`,
  );
}
process.exit(failed ? 1 : 0);
