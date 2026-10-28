/**
 * Smoke: flexible VTT clocks + local sidecar basename helpers.
 * Usage: pnpm --filter @promedia/hub exec tsx scripts/smoke-local-tracks.mts
 */
import { mkdir, writeFile, rm } from "node:fs/promises";
import path from "node:path";
import { filterSpeechCues } from "../src/subtitle-cue-text.ts";
import {
  findLocalSidecarFiles,
  localSidecarBasenames,
  tryLocalSidecarCaptions,
} from "../src/subtitle-local-tracks.ts";
import { ensureSubtitleDirs } from "../src/subtitle-paths.ts";
import {
  parseVttClock,
  parseWhisperCppVttOrSrt,
} from "../src/subtitle-whisper-cpp.ts";

function assert(cond: unknown, msg: string): asserts cond {
  if (!cond) throw new Error(msg);
}

async function main(): Promise<void> {
  assert(parseVttClock("00:00:01.000") === 1000, "HH:MM:SS.mmm");
  assert(parseVttClock("0:00:01.500") === 1500, "H:MM:SS.mmm");
  assert(parseVttClock("01:02.050") === 62050, "MM:SS.mmm");
  assert(parseVttClock("1:02.5") === null, "reject 1-digit ms");
  assert(parseVttClock("01:02.50") === 62500, "2-digit ms → *10");

  const shortVtt = `WEBVTT

00:00.000 --> 00:02.000 align:start position:0%
[MUSIC]

00:02.000 --> 00:04.500
Hola mundo desde YouTube

0:00:05.000 --> 0:00:07.000
Segunda línea
`;
  const parsed = filterSpeechCues(parseWhisperCppVttOrSrt(shortVtt));
  assert(parsed.length === 2, `expected 2 cues, got ${parsed.length}`);
  assert(parsed[0]!.startMs === 2000, `start ${parsed[0]!.startMs}`);
  assert(parsed[0]!.text.includes("Hola mundo"), parsed[0]!.text);
  assert(parsed[1]!.startMs === 5000, `start2 ${parsed[1]!.startMs}`);

  const names = localSidecarBasenames({
    assetId: "abc123",
    mediaPath: "/tmp/store/abc123.mp4",
    originalName: "Mi Cancion.mp4",
  });
  assert(names.includes("abc123"), "asset id stem");
  assert(names.includes("Mi Cancion"), "original stem");

  const dirs = await ensureSubtitleDirs();
  const assetId = "smokeLocal01";
  const mediaPath = path.join(dirs.wav, `${assetId}.mp3`);
  await writeFile(mediaPath, Buffer.from([0]));
  const side = path.join(dirs.sidecars, `${assetId}.vtt`);
  await mkdir(dirs.sidecars, { recursive: true });
  await writeFile(
    side,
    `WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nPista local de prueba\n`,
  );

  const found = await findLocalSidecarFiles({
    assetId,
    mediaPath,
    originalName: null,
  });
  assert(found.some((f) => f.endsWith(`${assetId}.vtt`)), "find sidecar");

  const caps = await tryLocalSidecarCaptions({
    assetId,
    mediaPath,
    originalName: null,
  });
  assert(caps && caps.cues.length === 1, "sidecar cues");
  assert(caps!.origin === "sidecar", "origin sidecar");
  assert(caps!.cues[0]!.text.includes("Pista local"), caps!.cues[0]!.text);

  await rm(side, { force: true });
  await rm(mediaPath, { force: true });

  console.log("SMOKE_OK");
}

main().catch((err) => {
  console.error("[smoke-local-tracks] FAILED:", err);
  process.exit(1);
});
