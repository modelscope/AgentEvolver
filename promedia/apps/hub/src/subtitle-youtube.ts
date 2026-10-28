import { mkdir, readdir, readFile, rm, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { spawn } from "node:child_process";
import type { SubtitleCue } from "@promedia/shared";
import {
  ensureAugmentedPath,
  missingBinHintFromError,
  resolveFfmpegBin,
  resolveYtDlpBin,
} from "./subtitle-bins.js";
import { filterSpeechCues } from "./subtitle-cue-text.js";
import {
  preserveNativeCaptionCues,
  scoreCaptionCueQuality,
} from "./subtitle-lyric-shape.js";
import { ensureSubtitleDirs } from "./subtitle-paths.js";
import { extractWav16k } from "./subtitle-asr.js";
import { parseWhisperCppVttOrSrt } from "./subtitle-whisper-cpp.js";

export type YoutubeAudioResult = {
  /** Absolute path to cached audio file (wav preferred; else raw bestaudio). */
  mediaPath: string;
  /** Synthetic asset id used for ASR cue cache (`yt_<videoId>`). */
  assetId: string;
  videoId: string;
};

export type YoutubeCaptionResult = {
  cues: SubtitleCue[];
  /** Language tag from the chosen caption file (e.g. `en`, `es`). */
  sourceLang: string | null;
  assetId: string;
  videoId: string;
  /** Manual vs auto-generated YouTube track. */
  trackKind: "manual" | "auto" | "unknown";
};

/** YouTube ids are 11 chars of [A-Za-z0-9_-]; reject anything else for path safety. */
function assertSafeVideoId(videoId: string): void {
  if (!/^[A-Za-z0-9_-]{6,20}$/.test(videoId)) {
    throw new Error("youtube_invalid_id");
  }
}

function runCmd(
  bin: string,
  args: string[],
): Promise<{ code: number; stderr: string; stdout: string }> {
  ensureAugmentedPath();
  return new Promise((resolve, reject) => {
    const child = spawn(bin, args, {
      stdio: ["ignore", "pipe", "pipe"],
      env: process.env,
    });
    let stderr = "";
    let stdout = "";
    child.stderr?.on("data", (chunk: Buffer) => {
      stderr += chunk.toString("utf8");
    });
    child.stdout?.on("data", (chunk: Buffer) => {
      stdout += chunk.toString("utf8");
    });
    child.on("error", reject);
    child.on("close", (code) => {
      resolve({ code: code ?? 1, stderr, stdout });
    });
  });
}

export async function ytDlpAvailable(): Promise<boolean> {
  try {
    const bin = resolveYtDlpBin();
    const { code } = await runCmd(bin, ["--version"]);
    return code === 0;
  } catch {
    return false;
  }
}

/**
 * Probe resolved ffmpeg (Homebrew PATH + FFMPEG_PATH) before any extract/remux.
 * Throws `ffmpeg_missing` so Control can show the brew hint — never a cache-path error.
 */
export async function ensureFfmpegForExtract(): Promise<string> {
  const bin = resolveFfmpegBin();
  try {
    const { code } = await runCmd(bin, ["-version"]);
    if (code === 0) return bin;
  } catch (err) {
    if (missingBinHintFromError(err, "ffmpeg")) {
      throw new Error("ffmpeg_missing");
    }
  }
  throw new Error("ffmpeg_missing");
}

/**
 * Map yt-dlp stderr to stable hints.
 * ffmpeg/Postprocessing ENOENT must NOT become `youtube_cache_path_error`
 * (that mislabel is what Julio saw when brew ffmpeg was missing mid `-x`).
 */
export function classifyYoutubeError(stderr: string): string {
  const s = (stderr || "").toLowerCase();

  const mentionsFfmpeg =
    s.includes("ffmpeg") ||
    s.includes("ffprobe") ||
    s.includes("postprocessing") ||
    s.includes("post-processing") ||
    s.includes("--ffmpeg-location");

  if (mentionsFfmpeg) {
    if (
      s.includes("not found") ||
      s.includes("no such file") ||
      s.includes("errno 2") ||
      s.includes("unable to") ||
      s.includes("is not installed") ||
      s.includes("install or provide") ||
      s.includes("please install")
    ) {
      return "ffmpeg_missing";
    }
    if (
      s.includes("conversion failed") ||
      s.includes("error while decoding") ||
      s.includes("invalid data found") ||
      s.includes("error opening input")
    ) {
      return "ffmpeg_extract_failed";
    }
    // Postprocessor failed for an unclear reason — still not a cache mkdir bug.
    return "ffmpeg_extract_failed";
  }

  if (
    s.includes("sign in to confirm your age") ||
    s.includes("age-restricted") ||
    s.includes("age restricted")
  ) {
    return "youtube_age_restricted";
  }
  if (
    s.includes("confirm you're not a bot") ||
    s.includes("confirm you’re not a bot") ||
    s.includes("not a bot") ||
    s.includes("--cookies-from-browser") ||
    s.includes("http error 403") ||
    s.includes("http error 429")
  ) {
    return "youtube_blocked";
  }
  if (
    s.includes("private video") ||
    s.includes("this video is private") ||
    s.includes("login required")
  ) {
    return "youtube_private";
  }
  if (
    s.includes("video unavailable") ||
    s.includes("is not available") ||
    s.includes("removed by the uploader")
  ) {
    return "youtube_unavailable";
  }

  // Real cache/path issues only when ffmpeg/postprocessing is not involved.
  if (
    s.includes("unable to rename") ||
    s.includes("no such file or directory") ||
    s.includes("errno 2")
  ) {
    return "youtube_cache_path_error";
  }

  const last = (stderr || "")
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean)
    .slice(-3)
    .join(" ");
  return last
    ? `youtube_download_failed: ${last.slice(0, 140)}`
    : "youtube_download_failed";
}

function isRetryableDownloadHint(hint: string): boolean {
  return (
    hint === "ffmpeg_extract_failed" ||
    hint === "youtube_cache_path_error" ||
    hint === "youtube_cache_missing_after_download" ||
    hint.startsWith("youtube_download_failed")
  );
}

/**
 * Remove temp + final artifacts for a video id (corrupt/partial downloads).
 */
export async function clearYoutubeCacheArtifacts(
  cacheDir: string,
  videoId: string,
): Promise<void> {
  let names: string[];
  try {
    names = await readdir(cacheDir);
  } catch {
    return;
  }
  const prefix = `${videoId}.`;
  await Promise.all(
    names
      .filter((name) => name.startsWith(prefix))
      .map((name) =>
        rm(path.join(cacheDir, name), { force: true }).catch(() => undefined),
      ),
  );
}

/**
 * Resolve the downloaded file under cacheDir by video id (any ext except
 * yt-dlp temp suffixes). Prefer wav, then m4a.
 */
export async function findCachedAudio(
  cacheDir: string,
  videoId: string,
): Promise<string | null> {
  let names: string[];
  try {
    names = await readdir(cacheDir);
  } catch {
    return null;
  }
  const prefix = `${videoId}.`;
  const matches: string[] = [];
  for (const name of names) {
    if (!name.startsWith(prefix)) continue;
    if (
      name.endsWith(".part") ||
      name.endsWith(".ytdl") ||
      name.endsWith(".temp")
    ) {
      continue;
    }
    const full = path.join(cacheDir, name);
    try {
      const st = await stat(full);
      if (!st.isFile() || st.size < 64) continue;
      matches.push(full);
    } catch {
      // continue
    }
  }
  if (!matches.length) return null;
  const wav = matches.find((p) => /\.wav$/i.test(p));
  if (wav) return wav;
  const m4a = matches.find((p) => /\.m4a$/i.test(p));
  return m4a ?? matches[0]!;
}

/** Absolute yt-dlp `-P` home + `-o` template (directory + `%(id)s.%(ext)s`). */
export function youtubeYtDlpOutputArgs(cacheDir: string): string[] {
  const absCache = path.resolve(cacheDir);
  return ["-P", absCache, "-o", "%(id)s.%(ext)s"];
}

/**
 * Download bestaudio only — no `-x` / postprocessor (avoids yt-dlp spawning
 * ffmpeg from a GUI PATH that lacks Homebrew).
 */
async function invokeYtDlpBestaudio(options: {
  bin: string;
  cacheDir: string;
  url: string;
}): Promise<{ code: number; stderr: string; stdout: string }> {
  const { bin, cacheDir, url } = options;
  await mkdir(cacheDir, { recursive: true });
  return runCmd(bin, [
    "--no-playlist",
    "--no-warnings",
    "-f",
    "bestaudio/best",
    ...youtubeYtDlpOutputArgs(cacheDir),
    "--",
    url,
  ]);
}

async function remuxCachedToWav(
  cacheDir: string,
  videoId: string,
  rawPath: string,
): Promise<string> {
  const wavPath = path.join(cacheDir, `${videoId}.wav`);
  if (/\.wav$/i.test(rawPath) && path.resolve(rawPath) === path.resolve(wavPath)) {
    return wavPath;
  }
  try {
    await extractWav16k(rawPath, wavPath);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    if (msg === "ffmpeg_missing" || msg.includes("ffmpeg_missing")) {
      throw new Error("ffmpeg_missing");
    }
    if (msg.startsWith("ffmpeg_extract_failed")) {
      throw new Error("ffmpeg_extract_failed");
    }
    throw new Error(
      `ffmpeg_extract_failed: ${msg.slice(0, 120)}`,
    );
  }
  // Prefer wav; drop raw container so next cache hit is clean.
  if (path.resolve(rawPath) !== path.resolve(wavPath)) {
    await rm(rawPath, { force: true }).catch(() => undefined);
  }
  return wavPath;
}

/**
 * Download YouTube audio into `promedia/data/subtitles/youtube-cache/`.
 * Strategy: require ffmpeg up front → yt-dlp bestaudio (no `-x`) → remux to
 * 16 kHz mono WAV with the resolved brew ffmpeg. One retry after clearing
 * partials (no tight loop).
 */
export async function ensureYoutubeAudio(options: {
  videoId: string;
  sourceUrl: string;
}): Promise<YoutubeAudioResult> {
  const { videoId, sourceUrl } = options;
  assertSafeVideoId(videoId);
  const assetId = `yt_${videoId}`;
  const dirs = await ensureSubtitleDirs();
  const cacheDir = path.resolve(dirs.youtubeCache);

  await mkdir(cacheDir, { recursive: true });

  const cached = await findCachedAudio(cacheDir, videoId);
  if (cached && /\.wav$/i.test(cached)) {
    return { mediaPath: cached, assetId, videoId };
  }

  const bin = resolveYtDlpBin();
  if (!(await ytDlpAvailable())) {
    throw new Error("yt_dlp_missing");
  }

  // Fail fast with brew hint — do not let a later postprocessor ENOENT
  // look like youtube_cache_path_error.
  await ensureFfmpegForExtract();

  // Existing raw bestaudio (m4a/webm/…) from older builds: remux in place.
  if (cached) {
    try {
      const wavPath = await remuxCachedToWav(cacheDir, videoId, cached);
      return { mediaPath: wavPath, assetId, videoId };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      if (msg === "ffmpeg_missing" || msg.includes("ffmpeg_missing")) {
        throw new Error("ffmpeg_missing");
      }
      // Corrupt raw — clear and re-download once below.
      await clearYoutubeCacheArtifacts(cacheDir, videoId);
    }
  }

  const url =
    sourceUrl?.trim() || `https://www.youtube.com/watch?v=${videoId}`;

  let lastHint = "youtube_download_failed";

  for (let attempt = 0; attempt < 2; attempt++) {
    if (attempt > 0) {
      await clearYoutubeCacheArtifacts(cacheDir, videoId);
    }

    let result: { code: number; stderr: string; stdout: string };
    try {
      result = await invokeYtDlpBestaudio({ bin, cacheDir, url });
    } catch (err) {
      if (missingBinHintFromError(err, "yt_dlp")) {
        throw new Error("yt_dlp_missing");
      }
      const msg = err instanceof Error ? err.message : String(err);
      throw new Error(`yt_dlp_spawn_failed: ${msg.slice(0, 120)}`);
    }

    if (result.code !== 0) {
      lastHint = classifyYoutubeError(result.stderr || result.stdout);
      if (lastHint === "ffmpeg_missing") {
        throw new Error("ffmpeg_missing");
      }
      if (
        lastHint === "youtube_age_restricted" ||
        lastHint === "youtube_blocked" ||
        lastHint === "youtube_private" ||
        lastHint === "youtube_unavailable"
      ) {
        throw new Error(lastHint);
      }
      if (!isRetryableDownloadHint(lastHint) || attempt === 1) {
        throw new Error(lastHint);
      }
      continue;
    }

    const rawPath = await findCachedAudio(cacheDir, videoId);
    if (!rawPath) {
      lastHint = "youtube_cache_missing_after_download";
      if (attempt === 1) throw new Error(lastHint);
      continue;
    }

    try {
      const wavPath = await remuxCachedToWav(cacheDir, videoId, rawPath);
      return { mediaPath: wavPath, assetId, videoId };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      if (msg === "ffmpeg_missing" || msg.includes("ffmpeg_missing")) {
        throw new Error("ffmpeg_missing");
      }
      lastHint = msg.startsWith("ffmpeg_extract_failed")
        ? "ffmpeg_extract_failed"
        : msg.slice(0, 180);
      await clearYoutubeCacheArtifacts(cacheDir, videoId);
      if (attempt === 1) {
        throw new Error(lastHint);
      }
    }
  }

  throw new Error(lastHint);
}

/** Caption sidecar extensions we accept after yt-dlp `--convert-subs vtt`. */
const CAPTION_EXTS = new Set([".vtt", ".srt"]);

export type YoutubeCaptionTrackKind = "manual" | "auto" | "unknown";

type CaptionCandidate = {
  filePath: string;
  lang: string;
  trackKind: YoutubeCaptionTrackKind;
  /** Lower rank = better (manual > auto; preferred language first). */
  rank: number;
};

/** A track advertised by yt-dlp JSON (`subtitles` vs `automatic_captions`). */
export type YoutubeCaptionTrackInfo = {
  lang: string;
  kind: "manual" | "auto";
};

/**
 * Parse `videoId.<lang>.vtt` (and `.en-orig`, `.es-US`, …) into a BCP-47-ish tag.
 * Returns null for the audio file itself (`videoId.wav` / `.m4a`).
 */
export function youtubeCaptionLangFromName(
  videoId: string,
  fileName: string,
): string | null {
  if (!fileName.startsWith(`${videoId}.`)) return null;
  const rest = fileName.slice(videoId.length + 1);
  const ext = path.extname(rest).toLowerCase();
  if (!CAPTION_EXTS.has(ext)) return null;
  const langPart = rest.slice(0, rest.length - ext.length).trim();
  if (!langPart) return null;
  // Skip obvious non-lang leftovers.
  if (/^(f\d+|temp|part)$/i.test(langPart)) return null;
  return langPart.toLowerCase();
}

export function captionLangBase(lang: string): string {
  return lang.split(/[-_.]/)[0] || lang;
}

/** Guess auto vs manual from yt-dlp filename conventions when meta is missing. */
export function youtubeCaptionKindFromLangTag(
  lang: string,
): YoutubeCaptionTrackKind {
  const l = lang.toLowerCase();
  // yt-dlp sometimes tags auto as `en-orig` / `…-auto` / leading `a.`.
  if (/(^|[.-])(orig|auto)([.-]|$)/i.test(l) || /^a\./i.test(l)) {
    return "auto";
  }
  return "unknown";
}

/**
 * Expand UI / target lang into yt-dlp `--sub-langs` preference order.
 * Spanish lyric videos often ship `es` or `es-419` manual tracks — both must
 * beat English auto when Control is ES.
 */
export function expandCaptionPreferLangs(preferLangs: string[]): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  const push = (raw: string) => {
    const l = raw.trim().toLowerCase();
    if (!l || seen.has(l)) return;
    seen.add(l);
    out.push(l);
  };
  for (const lang of preferLangs) {
    const base = captionLangBase(lang.toLowerCase());
    push(lang);
    push(base);
    if (base === "es") {
      push("es-419");
      push("es-es");
      push("es-mx");
      push("es-us");
    }
    if (base === "pt") {
      push("pt-br");
      push("pt-pt");
    }
    if (base === "en") {
      push("en-us");
      push("en-gb");
    }
  }
  // Callers that want an English fallback must pass "en" in preferLangs.
  // Native agent passes only the UI language so we never pick EN captions
  // (and Marian) when a Spanish track is missing — ASR `-l es` is preferred.
  return out;
}

/**
 * Rank caption tracks. Lower = better.
 * 1. Match app / preferLangs first (ES UI → Spanish track before English).
 * 2. Within the same language preference: **manual ≫ auto**.
 * 3. Prefer plain tags (`es` over `es-419`).
 */
export function rankYoutubeCaptionLang(
  lang: string,
  preferLangs: string[],
  trackKind: YoutubeCaptionTrackKind = "unknown",
): number {
  const base = captionLangBase(lang);
  const prefs = preferLangs.map((l) => captionLangBase(l.toLowerCase()));
  const prefIdx = prefs.findIndex((p) => p === base || lang.startsWith(`${p}-`));
  const prefScore = prefIdx >= 0 ? prefIdx : 100;
  // Manual uploaded tracks beat auto-generated ASR for the same language.
  // Gap is large so a matching-lang **manual** always beats matching-lang auto
  // even when the auto file is plain `es` and manual is `es-419`.
  const kindScore =
    trackKind === "manual" ? 0 : trackKind === "unknown" ? 8 : 20;
  // Prefer plain `en` / `es` over `en-orig` / regional when tied.
  const specificity = lang.includes("-") || lang.includes(".") ? 1 : 0;
  // Auto `*-orig` tags are YouTube ASR — push below plain auto.
  const origPenalty = /(^|[.-])orig([.-]|$)/i.test(lang) ? 3 : 0;
  return prefScore * 1000 + kindScore * 10 + specificity + origPenalty;
}

/**
 * True when a cached sidecar is good enough for the picked track.
 * Unknown/auto on disk must **not** satisfy a manual pick — that bug let stale
 * auto VTTs block official lyric downloads (same `videoId.es.vtt` name).
 */
export function cachedCaptionSatisfiesPick(
  candidate: Pick<CaptionCandidate, "lang" | "trackKind">,
  picked: YoutubeCaptionTrackInfo,
): boolean {
  if (captionLangBase(candidate.lang) !== captionLangBase(picked.lang)) {
    return false;
  }
  if (picked.kind === "manual") {
    return candidate.trackKind === "manual";
  }
  // Auto pick: accept auto or unknown (blind download) for the same lang.
  return (
    candidate.trackKind === "auto" || candidate.trackKind === "unknown"
  );
}

/**
 * Pick the best advertised track: manual matching preferLangs first, then
 * auto matching preferLangs, then any manual, then any auto.
 */
export function pickBestYoutubeCaptionTrack(
  tracks: YoutubeCaptionTrackInfo[],
  preferLangs: string[],
): YoutubeCaptionTrackInfo | null {
  if (!tracks.length) return null;
  const scored = tracks.map((t) => ({
    t,
    rank: rankYoutubeCaptionLang(t.lang, preferLangs, t.kind),
  }));
  scored.sort(
    (a, b) =>
      a.rank - b.rank ||
      a.t.lang.localeCompare(b.t.lang) ||
      (a.t.kind === "manual" ? -1 : 1),
  );
  return scored[0]?.t ?? null;
}

export async function findYoutubeCaptionFiles(
  cacheDir: string,
  videoId: string,
  preferLangs: string[],
  trackKindByLang?: Map<string, YoutubeCaptionTrackKind>,
): Promise<CaptionCandidate[]> {
  let names: string[];
  try {
    names = await readdir(cacheDir);
  } catch {
    return [];
  }
  const out: CaptionCandidate[] = [];
  for (const name of names) {
    const lang = youtubeCaptionLangFromName(videoId, name);
    if (!lang) continue;
    const filePath = path.join(cacheDir, name);
    try {
      const st = await stat(filePath);
      if (!st.isFile() || st.size < 16) continue;
    } catch {
      continue;
    }
    const base = captionLangBase(lang);
    const fromMeta =
      trackKindByLang?.get(lang) ||
      trackKindByLang?.get(base) ||
      null;
    const trackKind =
      fromMeta || youtubeCaptionKindFromLangTag(lang);
    out.push({
      filePath,
      lang,
      trackKind,
      rank: rankYoutubeCaptionLang(lang, preferLangs, trackKind),
    });
  }
  out.sort((a, b) => a.rank - b.rank || a.filePath.localeCompare(b.filePath));
  return out;
}

/**
 * Probe yt-dlp JSON for manual (`subtitles`) vs auto (`automatic_captions`).
 * Returns [] on failure — caller may still try a blind caption download.
 */
export async function listYoutubeCaptionTracks(options: {
  videoId: string;
  sourceUrl: string;
}): Promise<YoutubeCaptionTrackInfo[]> {
  const { videoId, sourceUrl } = options;
  assertSafeVideoId(videoId);
  if (!(await ytDlpAvailable())) return [];
  const bin = resolveYtDlpBin();
  const url =
    sourceUrl?.trim() || `https://www.youtube.com/watch?v=${videoId}`;
  let result: { code: number; stderr: string; stdout: string };
  try {
    result = await runCmd(bin, [
      "--no-playlist",
      "--no-warnings",
      "--skip-download",
      "-J",
      "--",
      url,
    ]);
  } catch {
    return [];
  }
  if (result.code !== 0 || !result.stdout.trim()) return [];
  let data: {
    subtitles?: Record<string, unknown>;
    automatic_captions?: Record<string, unknown>;
  };
  try {
    data = JSON.parse(result.stdout) as typeof data;
  } catch {
    return [];
  }
  const out: YoutubeCaptionTrackInfo[] = [];
  const seen = new Set<string>();
  for (const lang of Object.keys(data.subtitles || {})) {
    const key = `manual:${lang.toLowerCase()}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({ lang: lang.toLowerCase(), kind: "manual" });
  }
  for (const lang of Object.keys(data.automatic_captions || {})) {
    const key = `auto:${lang.toLowerCase()}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({ lang: lang.toLowerCase(), kind: "auto" });
  }
  return out;
}

export type YoutubeMeta = {
  videoId: string;
  /** Best display title (fulltitle || title). */
  title: string | null;
  track: string | null;
  artist: string | null;
  /** Duration in milliseconds when known. */
  durationMs: number;
};

/**
 * Lightweight yt-dlp `-J` probe for cancionero matching (title / track / artist / duration).
 * Does not download media. Returns null when yt-dlp is missing or the probe fails.
 */
export async function fetchYoutubeMeta(options: {
  videoId: string;
  sourceUrl: string;
}): Promise<YoutubeMeta | null> {
  const { videoId, sourceUrl } = options;
  assertSafeVideoId(videoId);
  if (!(await ytDlpAvailable())) return null;
  const bin = resolveYtDlpBin();
  const url =
    sourceUrl?.trim() || `https://www.youtube.com/watch?v=${videoId}`;
  let result: { code: number; stderr: string; stdout: string };
  try {
    result = await runCmd(bin, [
      "--no-playlist",
      "--no-warnings",
      "--skip-download",
      "-J",
      "--",
      url,
    ]);
  } catch {
    return null;
  }
  if (result.code !== 0 || !result.stdout.trim()) return null;
  let data: {
    title?: string;
    fulltitle?: string;
    track?: string;
    artist?: string;
    album_artist?: string;
    creator?: string;
    duration?: number;
  };
  try {
    data = JSON.parse(result.stdout) as typeof data;
  } catch {
    return null;
  }
  const title =
    (typeof data.fulltitle === "string" && data.fulltitle.trim()) ||
    (typeof data.title === "string" && data.title.trim()) ||
    null;
  const track =
    typeof data.track === "string" && data.track.trim() ? data.track.trim() : null;
  const artist =
    (typeof data.artist === "string" && data.artist.trim()) ||
    (typeof data.album_artist === "string" && data.album_artist.trim()) ||
    (typeof data.creator === "string" && data.creator.trim()) ||
    null;
  const durationSec =
    typeof data.duration === "number" && Number.isFinite(data.duration)
      ? data.duration
      : 0;
  return {
    videoId,
    title,
    track,
    artist,
    durationMs: durationSec > 0 ? Math.round(durationSec * 1000) : 0,
  };
}

async function invokeYtDlpCaptions(options: {
  bin: string;
  cacheDir: string;
  url: string;
  /** Exact lang tag to request (e.g. `es`, `en`). */
  lang: string;
  /** Manual uploaded vs auto-generated. */
  kind: "manual" | "auto";
}): Promise<{ code: number; stderr: string; stdout: string }> {
  const { bin, cacheDir, url, lang, kind } = options;
  await mkdir(cacheDir, { recursive: true });
  const base = captionLangBase(lang);
  const langArgs = [...new Set([`${base}.*`, base, lang])].filter(Boolean);
  const writeFlag =
    kind === "manual" ? "--write-subs" : "--write-auto-subs";
  return runCmd(bin, [
    "--no-playlist",
    "--no-warnings",
    "--skip-download",
    writeFlag,
    "--sub-langs",
    langArgs.join(","),
    "--sub-format",
    "vtt/srt/best",
    "--convert-subs",
    "vtt",
    ...youtubeYtDlpOutputArgs(cacheDir),
    "--",
    url,
  ]);
}

/** Legacy blind download (manual + auto) when JSON probe is unavailable. */
async function invokeYtDlpCaptionsBlind(options: {
  bin: string;
  cacheDir: string;
  url: string;
  preferLangs: string[];
}): Promise<{ code: number; stderr: string; stdout: string }> {
  const { bin, cacheDir, url, preferLangs } = options;
  await mkdir(cacheDir, { recursive: true });
  const langs = [
    ...preferLangs.map((l) => `${captionLangBase(l)}.*`),
    ...preferLangs.map((l) => captionLangBase(l)),
    "en.*",
    "en",
  ];
  const uniqueLangs = [...new Set(langs.filter(Boolean))];
  return runCmd(bin, [
    "--no-playlist",
    "--no-warnings",
    "--skip-download",
    "--write-subs",
    "--write-auto-subs",
    "--sub-langs",
    uniqueLangs.join(","),
    "--sub-format",
    "vtt/srt/best",
    "--convert-subs",
    "vtt",
    ...youtubeYtDlpOutputArgs(cacheDir),
    "--",
    url,
  ]);
}

/**
 * Prefer official / auto YouTube captions over Whisper-on-audio for music
 * videos — Whisper often emits only `[MUSIC]` / `[MÚSICA]` for songs.
 * Prefers **manual** tracks over auto, matching app language when possible.
 * Returns null when no usable speech/lyric cues are available (caller falls
 * through to audio ASR). Never throws for "no captions"; hard yt-dlp failures
 * are swallowed so the Whisper path can still run.
 */
export async function tryYoutubeCaptions(options: {
  videoId: string;
  sourceUrl: string;
  /** Preferred language codes (e.g. targetLang then `en`). */
  preferLangs: string[];
}): Promise<YoutubeCaptionResult | null> {
  const { videoId, sourceUrl } = options;
  const preferLangs = expandCaptionPreferLangs(options.preferLangs);
  assertSafeVideoId(videoId);
  const assetId = `yt_${videoId}`;
  const dirs = await ensureSubtitleDirs();
  const cacheDir = path.resolve(dirs.youtubeCache);
  await mkdir(cacheDir, { recursive: true });

  const url =
    sourceUrl?.trim() || `https://www.youtube.com/watch?v=${videoId}`;

  // Probe manual vs auto so we can prefer uploaded lyric tracks + UI lang.
  const advertised = await listYoutubeCaptionTracks({ videoId, sourceUrl });
  const kindByLang = new Map<string, YoutubeCaptionTrackKind>();
  for (const t of advertised) {
    const prev = kindByLang.get(t.lang);
    // Manual wins if both exist for the same tag.
    if (!prev || t.kind === "manual") {
      kindByLang.set(t.lang, t.kind);
      kindByLang.set(captionLangBase(t.lang), t.kind);
    }
  }

  // Read prior meta so a stale auto file is not treated as the official track.
  const metaPath = path.join(cacheDir, `${videoId}.captions-meta.json`);
  try {
    const metaRaw = await readFile(metaPath, "utf8");
    const meta = JSON.parse(metaRaw) as {
      lang?: string;
      trackKind?: YoutubeCaptionTrackKind;
    };
    if (meta.lang && meta.trackKind && meta.trackKind !== "unknown") {
      const prev = kindByLang.get(meta.lang.toLowerCase());
      if (!prev || meta.trackKind === "manual") {
        kindByLang.set(meta.lang.toLowerCase(), meta.trackKind);
        kindByLang.set(captionLangBase(meta.lang), meta.trackKind);
      }
    }
  } catch {
    // no meta yet
  }

  const picked = pickBestYoutubeCaptionTrack(advertised, preferLangs);

  // Reuse previously downloaded caption sidecars when present.
  let candidates = await findYoutubeCaptionFiles(
    cacheDir,
    videoId,
    preferLangs,
    kindByLang,
  );

  // If we know the best track and it's not on disk yet, fetch only that one.
  if (picked && (await ytDlpAvailable())) {
    const already = candidates.some((c) =>
      cachedCaptionSatisfiesPick(c, picked),
    );
    if (!already) {
      const bin = resolveYtDlpBin();
      try {
        await ensureFfmpegForExtract();
      } catch {
        // convert-subs may need ffmpeg; keep going — raw vtt often works.
      }
      // Drop same-lang sidecars so auto cannot shadow the manual download
      // (yt-dlp writes `videoId.es.vtt` for both kinds).
      for (const c of candidates) {
        if (captionLangBase(c.lang) === captionLangBase(picked.lang)) {
          await rm(c.filePath, { force: true }).catch(() => undefined);
        }
      }
      try {
        const result = await invokeYtDlpCaptions({
          bin,
          cacheDir,
          url,
          lang: picked.lang,
          kind: picked.kind,
        });
        if (result.code !== 0) {
          const hint = classifyYoutubeError(result.stderr || result.stdout);
          console.warn(
            `[subtitle] YouTube ${picked.kind} captions lang=${picked.lang} unavailable (${hint})`,
          );
        } else {
          // Stamp kind so the next run knows this file is the official track.
          kindByLang.set(picked.lang, picked.kind);
          kindByLang.set(captionLangBase(picked.lang), picked.kind);
        }
      } catch (err) {
        console.warn(
          "[subtitle] YouTube captions fetch failed:",
          err instanceof Error ? err.message : err,
        );
      }
      candidates = await findYoutubeCaptionFiles(
        cacheDir,
        videoId,
        preferLangs,
        kindByLang,
      );
    }
  }

  if (!candidates.length) {
    if (!(await ytDlpAvailable())) {
      return null;
    }
    const bin = resolveYtDlpBin();
    try {
      await ensureFfmpegForExtract();
    } catch {
      return null;
    }
    try {
      // Blind path: try **manual-only** first (gold-standard lyric tracks),
      // then fall back to auto if nothing landed.
      let result = await invokeYtDlpCaptions({
        bin,
        cacheDir,
        url,
        lang: preferLangs[0] || "en",
        kind: "manual",
      });
      candidates = await findYoutubeCaptionFiles(
        cacheDir,
        videoId,
        preferLangs,
        kindByLang,
      );
      if (!candidates.length) {
        result = await invokeYtDlpCaptionsBlind({
          bin,
          cacheDir,
          url,
          preferLangs,
        });
      }
      if (result.code !== 0 && !candidates.length) {
        const hint = classifyYoutubeError(result.stderr || result.stdout);
        console.warn(
          `[subtitle] YouTube captions unavailable (${hint}) — falling back to Whisper ASR`,
        );
        return null;
      }
    } catch (err) {
      console.warn(
        "[subtitle] YouTube captions fetch failed:",
        err instanceof Error ? err.message : err,
      );
      return null;
    }
    candidates = await findYoutubeCaptionFiles(
      cacheDir,
      videoId,
      preferLangs,
      kindByLang,
    );
  }

  // Prefer the probed track's language/kind when scoring local files.
  if (picked) {
    for (const c of candidates) {
      if (captionLangBase(c.lang) === captionLangBase(picked.lang)) {
        // Only upgrade unknown → picked.kind; never relabel a known auto as manual.
        if (c.trackKind === "unknown" || c.trackKind === picked.kind) {
          c.trackKind = picked.kind;
        }
        c.rank = rankYoutubeCaptionLang(c.lang, preferLangs, c.trackKind);
      }
    }
    candidates.sort(
      (a, b) => a.rank - b.rank || a.filePath.localeCompare(b.filePath),
    );
  }

  // Parse all viable files; pick by lang/kind rank + lyric-shaped quality
  // (official hymn lines beat ASR mush when both exist).
  type Scored = {
    cand: CaptionCandidate;
    cues: SubtitleCue[];
    trackKind: YoutubeCaptionTrackKind;
    quality: number;
  };
  const scored: Scored[] = [];
  for (const cand of candidates) {
    let raw: string;
    try {
      raw = await readFile(cand.filePath, "utf8");
    } catch {
      continue;
    }
    const parsed = filterSpeechCues(parseWhisperCppVttOrSrt(raw));
    if (parsed.length < 1) continue;
    const sourceLang = captionLangBase(cand.lang);
    const trackKind: YoutubeCaptionTrackKind =
      cand.trackKind !== "unknown"
        ? cand.trackKind
        : picked && captionLangBase(picked.lang) === sourceLang
          ? picked.kind
          : youtubeCaptionKindFromLangTag(cand.lang);
    const quality =
      scoreCaptionCueQuality(parsed, trackKind) - cand.rank * 10;
    scored.push({ cand, cues: parsed, trackKind, quality });
  }
  scored.sort(
    (a, b) =>
      b.quality - a.quality ||
      a.cand.rank - b.cand.rank ||
      a.cand.filePath.localeCompare(b.cand.filePath),
  );

  const best = scored[0];
  if (!best) return null;

  // Native agent: keep official VTT clocks/words — no lyric-shape drift.
  // (ASR fallback still shapes in subtitle-asr / finishWithCues.)
  const cues = preserveNativeCaptionCues(best.cues);
  const sourceLang = captionLangBase(best.cand.lang);
  console.log(
    `[subtitle] native captions kind=${best.trackKind} lang=${best.cand.lang} cues=${cues.length} quality=${best.quality} → ${best.cand.filePath}`,
  );
  void writeFile(
    metaPath,
    JSON.stringify(
      {
        agent: "native_v1",
        lang: best.cand.lang,
        sourceLang,
        trackKind: best.trackKind,
        cues: cues.length,
        quality: best.quality,
        file: path.basename(best.cand.filePath),
        at: new Date().toISOString(),
      },
      null,
      2,
    ),
    "utf8",
  ).catch(() => undefined);
  return {
    cues,
    sourceLang,
    assetId,
    videoId,
    trackKind: best.trackKind,
  };
}
