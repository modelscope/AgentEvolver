#!/usr/bin/env node
/**
 * Smoke-test Quelea song parsers against tiny public-domain fixtures
 * and optionally a local .qsp pack (never writes copyrighted packs into git).
 *
 *   pnpm --filter @promedia/hub exec tsx ../../scripts/smoke-quelea-import.mjs
 *   QUELEA_QSP=/path/to/CancioneroFeb2026.qsp node … (via tsx)
 */
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { tmpdir } from "node:os";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const hubRoot = path.resolve(__dirname, "../apps/hub");

async function loadImporter() {
  try {
    return await import(path.join(hubRoot, "dist/quelea-import.js"));
  } catch {
    return await import(path.join(hubRoot, "src/quelea-import.ts"));
  }
}

const { parseQueleaSongXml, parseQueleaQsp } = await loadImporter();

const fixturesDir = path.join(hubRoot, "src/quelea/fixtures");
const files = [
  path.join(fixturesDir, "sample-amazing-grace.xml"),
  path.join(fixturesDir, "sample-doxology.xml"),
];

let failed = 0;

for (const file of files) {
  const name = path.basename(file);
  const xml = await readFile(file, "utf8");
  const result = parseQueleaSongXml(xml, {
    fallbackTitle: name.replace(/\.xml$/i, ""),
  });
  if (!result.ok) {
    console.log(`FAIL ${name}: ${result.message}`);
    failed += 1;
    continue;
  }
  console.log(
    `OK   ${name}: «${result.song.title}» · ${result.song.stanzas.length} stanza(s) · artist=${result.song.artist ?? "-"}`,
  );
}

const packFixture = path.join(fixturesDir, "sample-pack.qsp");
{
  const bytes = await readFile(packFixture);
  const pack = await parseQueleaQsp({ bytes, originalName: "sample-pack.qsp" });
  if (!pack.ok) {
    console.log(`FAIL sample-pack.qsp: ${pack.message}`);
    failed += 1;
  } else {
    console.log(
      `OK   sample-pack.qsp: ${pack.songs.length} song(s), skipped=${pack.skipped.length}`,
    );
  }
}

const qspPath =
  process.env.QUELEA_QSP ||
  "/cursor/stores/bc-1376f14d-6310-4ae9-aaf0-466e07c4b67f/media/CancioneroFeb2026.qsp";

try {
  const bytes = await readFile(qspPath);
  const pack = await parseQueleaQsp({
    bytes,
    originalName: path.basename(qspPath),
  });
  if (!pack.ok) {
    console.log(`FAIL ${path.basename(qspPath)}: ${pack.message}`);
    failed += 1;
  } else {
    console.log(
      `OK   ${path.basename(qspPath)}: ${pack.songs.length} song(s), skipped=${pack.skipped.length}`,
    );
    if (pack.songs.length < 100) {
      console.log(
        `WARN expected hundreds of songs from cancionero; got ${pack.songs.length}`,
      );
      failed += 1;
    }
  }
} catch (err) {
  console.log(
    `(optional cancionero missing at ${qspPath}: ${err instanceof Error ? err.message : err})`,
  );
}

// Optional: import into a temp library root (does not touch real data/)
if (process.env.QUELEA_IMPORT_TEMP === "1") {
  const { LibraryStore } = await import(
    path.join(hubRoot, "src/library-store.ts")
  ).catch(() => import(path.join(hubRoot, "dist/library-store.js")));
  const tmp = await mkdtemp(path.join(tmpdir(), "promedia-qsp-"));
  process.env.LIBRARY_ROOT = tmp;
  try {
    const store = new LibraryStore();
    const bytes = await readFile(qspPath);
    const result = await store.importQueleaUpload({
      bytes,
      originalName: path.basename(qspPath),
    });
    console.log(
      `TEMP import → ${tmp}: created=${result.imported.length} updated=${result.updated.length} skipped=${result.skipped.length}`,
    );
    await writeFile(
      path.join(tmp, "import-summary.json"),
      JSON.stringify(
        {
          importedCount: result.imported.length,
          updatedCount: result.updated.length,
          skippedCount: result.skipped.length,
          sample: result.imported.slice(0, 3).map((i) => i.title),
        },
        null,
        2,
      ),
    );
  } finally {
    // keep temp for inspection unless CLEAN=1
    if (process.env.CLEAN === "1") {
      await rm(tmp, { recursive: true, force: true });
    } else {
      console.log(`(temp library kept at ${tmp})`);
    }
  }
}

process.exit(failed ? 1 : 0);
