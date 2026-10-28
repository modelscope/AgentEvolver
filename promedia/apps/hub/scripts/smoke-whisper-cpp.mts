/**
 * Smoke: whisper.cpp parse + optional live ASR with whisper-cli.
 * Usage: pnpm --filter @promedia/hub exec tsx scripts/smoke-whisper-cpp.mts
 *
 * Set WHISPER_CPP_PATH to a local whisper-cli for a full WAV round-trip.
 */
import { spawn } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import {
  parseWhisperCppJson,
  parseWhisperCppVttOrSrt,
  whisperCliAvailable,
  whisperCppModelPath,
  ensureWhisperCppModel,
  transcribeWithWhisperCpp,
} from "../src/subtitle-whisper-cpp.ts";
import {
  clearCorruptModelCache,
  ensureSubtitleDirs,
  subtitleModelsRoot,
} from "../src/subtitle-paths.ts";
import { resolveWhisperCliBin } from "../src/subtitle-bins.ts";
import { extractWav16k } from "../src/subtitle-asr.ts";

function run(bin: string, args: string[]): Promise<void> {
  return new Promise((resolve, reject) => {
    const child = spawn(bin, args, { stdio: "inherit" });
    child.on("error", reject);
    child.on("close", (code) =>
      code === 0 ? resolve() : reject(new Error(`${bin} exit ${code}`)),
    );
  });
}

async function main(): Promise<void> {
  const sampleJson = JSON.stringify({
    transcription: [
      {
        text: " Hello world ",
        offsets: { from: 0, to: 1500 },
      },
      {
        text: "second",
        timestamps: { from: "00:00:01.500", to: "00:00:03.000" },
      },
    ],
  });
  const cuesJ = parseWhisperCppJson(sampleJson);
  if (cuesJ.length !== 2 || cuesJ[0]!.text !== "Hello world") {
    throw new Error(`json parse bad: ${JSON.stringify(cuesJ)}`);
  }
  if (cuesJ[1]!.startMs !== 1500 || cuesJ[1]!.endMs !== 3000) {
    throw new Error(`json clock bad: ${JSON.stringify(cuesJ[1])}`);
  }

  const vtt = `WEBVTT

00:00:00.000 --> 00:00:01.200
Alpha

00:00:01.200 --> 00:00:02.400
Beta
`;
  const cuesV = parseWhisperCppVttOrSrt(vtt);
  if (cuesV.length !== 2 || cuesV[0]!.text !== "Alpha") {
    throw new Error(`vtt parse bad: ${JSON.stringify(cuesV)}`);
  }

  // clearCorrupt must not claim wipe when Xenova dir is absent
  const dirs = await ensureSubtitleDirs();
  const cleared = await clearCorruptModelCache(
    dirs.models,
    "Xenova/whisper-tiny-smoke-missing",
  );
  if (cleared.removedModelDir) {
    throw new Error("clearCorrupt claimed modelDir wipe for missing tree");
  }
  console.log("parse+clear OK; models root", subtitleModelsRoot());
  console.log("whisper-cli resolve:", resolveWhisperCliBin());
  console.log("whisper-cli available:", await whisperCliAvailable());
  console.log("ggml path:", whisperCppModelPath(dirs.models));

  if (!(await whisperCliAvailable())) {
    console.log("SMOKE_OK (parse only; no whisper-cli on PATH)");
    return;
  }

  // Full round-trip: short WAV → whisper.cpp
  const sampleDir = path.join(dirs.wav, "smoke-wcpp");
  await mkdir(sampleDir, { recursive: true });
  const media = path.join(sampleDir, "tone.wav");
  // Generate with ffmpeg lavfi if media missing
  await run("ffmpeg", [
    "-y",
    "-f",
    "lavfi",
    "-i",
    "sine=frequency=440:duration=2",
    "-ac",
    "1",
    "-ar",
    "16000",
    media,
  ]);
  const wavPath = path.join(sampleDir, "tone-16k.wav");
  await extractWav16k(media, wavPath);

  await ensureWhisperCppModel((s) => console.log("[smoke]", s));
  const result = await transcribeWithWhisperCpp({
    wavPath,
    translateToEn: false,
    workDir: sampleDir,
    onStatus: (s) => console.log("[smoke]", s),
  });
  console.log("cues:", result.cues.length);
  // Tone may yield 0 cues (no speech) — that still proves the binary path works.
  await writeFile(
    path.join(sampleDir, "smoke-result.json"),
    JSON.stringify(result, null, 2),
  );
  console.log("SMOKE_OK (whisper-cli round-trip)");
}

main().catch((err) => {
  console.error("[smoke-whisper-cpp] FAILED:", err);
  process.exit(1);
});
