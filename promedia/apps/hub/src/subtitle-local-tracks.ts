import { spawn } from "node:child_process";
import { access, mkdir, readdir, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import type { SubtitleCue } from "@promedia/shared";
import {
  ensureAugmentedPath,
  resolveFfmpegBin,
} from "./subtitle-bins.js";
import { filterSpeechCues } from "./subtitle-cue-text.js";
import { ensureSubtitleDirs, subtitleDataRoot } from "./subtitle-paths.js";
import { parseWhisperCppVttOrSrt } from "./subtitle-whisper-cpp.js";

/**
 * Practical alternate to Whisper on Mac: timed captions Julio already has.
 *
 * 1. Sidecar files in `promedia/data/subtitles/sidecars/` named after the
 *    upload’s original basename (`Canción.srt` / `Canción.vtt`) or asset id
 *    (`{assetId}.vtt`).
 * 2. Soft-sub / embedded subtitle streams inside the media via ffmpeg.
 *
 * Prefer these before ASR — Whisper on music/speech is still flaky without
 * brew `whisper-cli` + a full ggml, and sidecars are deterministic.
 */

const CAPTION_EXTS = [".vtt", ".srt"] as const;

export type LocalTrackResult = {
  cues: SubtitleCue[];
  sourceLang: string | null;
  /** How cues were obtained (for logs / hint). */
  origin: "sidecar" | "embedded";
  detail: string;
};

export function subtitleSidecarsDir(dataRoot?: string): string {
  return path.join(dataRoot ?? subtitleDataRoot(), "sidecars");
}

function stripExt(name: string): string {
  const base = path.basename(name);
  const dot = base.lastIndexOf(".");
  return dot > 0 ? base.slice(0, dot) : base;
}

function runCmd(
  bin: string,
  args: string[],
): Promise<{ code: number; stdout: string; stderr: string }> {
  ensureAugmentedPath();
  return new Promise((resolve, reject) => {
    const child = spawn(bin, args, {
      stdio: ["ignore", "pipe", "pipe"],
      env: process.env,
    });
    let stdout = "";
    let stderr = "";
    child.stdout?.on("data", (chunk: Buffer) => {
      stdout += chunk.toString("utf8");
    });
    child.stderr?.on("data", (chunk: Buffer) => {
      stderr += chunk.toString("utf8");
    });
    child.on("error", reject);
    child.on("close", (code) => {
      resolve({ code: code ?? 1, stdout, stderr });
    });
  });
}

async function parseCaptionFile(filePath: string): Promise<SubtitleCue[]> {
  let raw: string;
  try {
    raw = await readFile(filePath, "utf8");
  } catch {
    return [];
  }
  return filterSpeechCues(parseWhisperCppVttOrSrt(raw));
}

/**
 * Candidate sidecar basenames: asset id, original upload name, media file stem.
 */
export function localSidecarBasenames(options: {
  assetId: string;
  mediaPath: string;
  originalName?: string | null;
}): string[] {
  const out: string[] = [];
  const push = (v: string | null | undefined) => {
    const s = (v ?? "").trim();
    if (!s) return;
    const stem = stripExt(s);
    if (stem && !out.includes(stem)) out.push(stem);
  };
  push(options.assetId);
  push(options.originalName);
  push(path.basename(options.mediaPath));
  return out;
}

/** Resolve sidecars under data/subtitles/sidecars/ and next to the media file. */
export async function findLocalSidecarFiles(options: {
  assetId: string;
  mediaPath: string;
  originalName?: string | null;
}): Promise<string[]> {
  const dirs = await ensureSubtitleDirs();
  const sidecarRoot = subtitleSidecarsDir(dirs.data);
  await mkdir(sidecarRoot, { recursive: true });

  const basenames = localSidecarBasenames(options);
  const found: string[] = [];
  const mediaDir = path.dirname(options.mediaPath);

  for (const stem of basenames) {
    for (const ext of CAPTION_EXTS) {
      for (const dir of [sidecarRoot, mediaDir]) {
        const candidate = path.join(dir, `${stem}${ext}`);
        try {
          await access(candidate);
          if (!found.includes(candidate)) found.push(candidate);
        } catch {
          // miss
        }
      }
    }
  }

  // Also accept any `assetId.*.vtt` / `.srt` Julio drops in sidecars/.
  try {
    const entries = await readdir(sidecarRoot);
    for (const name of entries) {
      const lower = name.toLowerCase();
      if (
        !CAPTION_EXTS.some((ext) => lower.endsWith(ext)) ||
        !lower.startsWith(options.assetId.toLowerCase())
      ) {
        continue;
      }
      const full = path.join(sidecarRoot, name);
      if (!found.includes(full)) found.push(full);
    }
  } catch {
    // ignore
  }

  return found;
}

export async function tryLocalSidecarCaptions(options: {
  assetId: string;
  mediaPath: string;
  originalName?: string | null;
}): Promise<LocalTrackResult | null> {
  const files = await findLocalSidecarFiles(options);
  for (const filePath of files) {
    const cues = await parseCaptionFile(filePath);
    if (cues.length < 1) continue;
    console.log(
      `[subtitle] local sidecar captions cues=${cues.length} → ${filePath}`,
    );
    return {
      cues,
      sourceLang: null,
      origin: "sidecar",
      detail: path.basename(filePath),
    };
  }
  return null;
}

type FfprobeStream = {
  codec_type?: string;
  codec_name?: string;
  tags?: { language?: string; title?: string };
  index?: number;
};

/**
 * List subtitle stream indexes via ffprobe (bundled with ffmpeg on Mac brew).
 */
export async function listEmbeddedSubtitleStreams(
  mediaPath: string,
): Promise<{ index: number; language: string | null; codec: string }[]> {
  const ffmpeg = resolveFfmpegBin();
  // Prefer ffprobe next to ffmpeg; fall back to bare ffprobe on PATH.
  const probeCandidates = [
    ffmpeg.replace(/ffmpeg(\.exe)?$/i, "ffprobe$1"),
    "ffprobe",
  ];
  let stdout = "";
  let ok = false;
  for (const bin of probeCandidates) {
    try {
      const result = await runCmd(bin, [
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-select_streams",
        "s",
        mediaPath,
      ]);
      if (result.code === 0 && result.stdout.trim()) {
        stdout = result.stdout;
        ok = true;
        break;
      }
    } catch {
      // try next
    }
  }
  if (!ok) return [];

  let parsed: { streams?: FfprobeStream[] };
  try {
    parsed = JSON.parse(stdout) as { streams?: FfprobeStream[] };
  } catch {
    return [];
  }
  const out: { index: number; language: string | null; codec: string }[] = [];
  for (const s of parsed.streams ?? []) {
    if (s.codec_type !== "subtitle") continue;
    const index = typeof s.index === "number" ? s.index : out.length;
    out.push({
      index,
      language: s.tags?.language?.trim() || null,
      codec: s.codec_name ?? "unknown",
    });
  }
  return out;
}

/**
 * Extract the first usable embedded subtitle stream to WebVTT via ffmpeg.
 */
export async function tryEmbeddedSubtitleCaptions(options: {
  assetId: string;
  mediaPath: string;
}): Promise<LocalTrackResult | null> {
  const streams = await listEmbeddedSubtitleStreams(options.mediaPath);
  if (!streams.length) return null;

  const dirs = await ensureSubtitleDirs();
  const outBase = path.join(dirs.wav, `${options.assetId}_embedded`);
  const outVtt = `${outBase}.vtt`;
  await rm(outVtt, { force: true }).catch(() => undefined);

  const bin = resolveFfmpegBin();
  // Map by relative subtitle stream (0:s:0) — more portable than absolute index.
  for (let si = 0; si < streams.length; si++) {
    await rm(outVtt, { force: true }).catch(() => undefined);
    try {
      const result = await runCmd(bin, [
        "-hide_banner",
        "-nostdin",
        "-y",
        "-i",
        options.mediaPath,
        "-map",
        `0:s:${si}`,
        "-f",
        "webvtt",
        outVtt,
      ]);
      if (result.code !== 0) {
        continue;
      }
    } catch {
      continue;
    }
    const cues = await parseCaptionFile(outVtt);
    if (cues.length < 1) continue;
    const lang = streams[si]?.language ?? null;
    console.log(
      `[subtitle] embedded captions stream=${si} lang=${lang ?? "?"} cues=${cues.length}`,
    );
    // Keep a small meta note for operators.
    void writeFile(
      `${outBase}.meta.json`,
      JSON.stringify(
        {
          stream: si,
          language: lang,
          cues: cues.length,
          at: new Date().toISOString(),
        },
        null,
        2,
      ),
      "utf8",
    ).catch(() => undefined);
    return {
      cues,
      sourceLang: lang,
      origin: "embedded",
      detail: `stream:${si}${lang ? `:${lang}` : ""}`,
    };
  }
  return null;
}

/**
 * Prefer sidecar, then embedded softsubs. Returns null when neither yields
 * speech/lyric cues (caller falls through to Whisper).
 */
export async function tryLocalCaptionTracks(options: {
  assetId: string;
  mediaPath: string;
  originalName?: string | null;
}): Promise<LocalTrackResult | null> {
  const sidecar = await tryLocalSidecarCaptions(options);
  if (sidecar) return sidecar;
  return tryEmbeddedSubtitleCaptions(options);
}
