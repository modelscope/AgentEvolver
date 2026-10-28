#!/usr/bin/env node
/**
 * Smoke: generate a short local WAV (ffmpeg) and run hub ASR once.
 * Usage (from promedia/): node scripts/verify-subtitle-asr.mjs
 *
 * First run downloads Whisper tiny into data/models/ (gitignored).
 */
import { spawn } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const outDir = path.join(root, "data", "subtitles", "sample");
const wavPath = path.join(outDir, "sample-beep.wav");

function run(bin, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(bin, args, { stdio: "inherit" });
    child.on("error", reject);
    child.on("close", (code) =>
      code === 0 ? resolve() : reject(new Error(`${bin} exit ${code}`)),
    );
  });
}

await mkdir(outDir, { recursive: true });

// 3s mono 16 kHz tone — proves ffmpeg extract path; Whisper may return empty/no_speech.
console.log("[verify] generating sample WAV with ffmpeg…");
await run("ffmpeg", [
  "-y",
  "-f",
  "lavfi",
  "-i",
  "sine=frequency=440:duration=3",
  "-ac",
  "1",
  "-ar",
  "16000",
  wavPath,
]);

process.chdir(root);
process.env.TRANSFORMERS_CACHE =
  process.env.TRANSFORMERS_CACHE || path.join(root, "data", "models");
process.env.HF_HOME = process.env.HF_HOME || process.env.TRANSFORMERS_CACHE;

console.log("[verify] importing hub ASR module…");
const { pathToFileURL } = await import("node:url");
// Compile not required — use tsx if available, else expect dist.
let transcribeLocalFile;
try {
  const mod = await import(
    pathToFileURL(path.join(root, "apps/hub/dist/subtitle-asr.js")).href
  );
  transcribeLocalFile = mod.transcribeLocalFile;
} catch {
  console.log("[verify] dist missing — run via tsx source");
  const { register } = await import("node:module");
  // Fallback: spawn tsx
  const tsxBin = path.join(root, "node_modules", ".bin", "tsx");
  const runner = path.join(outDir, "run-asr.ts");
  await writeFile(
    runner,
    `
import { transcribeLocalFile } from "../../apps/hub/src/subtitle-asr.ts";
const r = await transcribeLocalFile({
  mediaPath: ${JSON.stringify(wavPath)},
  assetId: "sample-beep",
  translateToEn: false,
  targetLang: "es",
  onStatus: (s) => console.log("[asr]", s),
});
console.log(JSON.stringify({ cues: r.cues.length, sourceLang: r.sourceLang }, null, 2));
`,
    "utf8",
  );
  await run(tsxBin, [runner]);
  console.log("[verify] OK (tsx path)");
  process.exit(0);
}

const result = await transcribeLocalFile({
  mediaPath: wavPath,
  assetId: "sample-beep",
  translateToEn: false,
  targetLang: "es",
  onStatus: (s) => console.log("[asr]", s),
});
console.log(
  JSON.stringify(
    { cues: result.cues.length, sourceLang: result.sourceLang, sample: result.cues[0] },
    null,
    2,
  ),
);
console.log("[verify] OK");
