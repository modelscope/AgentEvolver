import { spawn } from "node:child_process";
import { createWriteStream, openSync, closeSync } from "node:fs";
import { access, mkdir, readFile, rename, rm, stat } from "node:fs/promises";
import path from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import type { SubtitleCue } from "@promedia/shared";
import {
  ensureAugmentedPath,
  missingBinHintFromError,
  resolveWhisperCliBin,
} from "./subtitle-bins.js";
import { cleanCueText, filterSpeechCues } from "./subtitle-cue-text.js";
import { shapeLyricLikeCues } from "./subtitle-lyric-shape.js";
import { ensureSubtitleDirs, subtitleModelsRoot } from "./subtitle-paths.js";

/**
 * Default ggml **base** multilingual (~142 MB). Tiny (~75 MB) often mangled
 * Spanish speech / lyrics (`[MUSIC]`-only); base is the accuracy/size sweet
 * spot for Mac local ASR. Override with `SUBTITLE_WHISPER_CPP_MODEL=ggml-tiny.bin`
 * (and matching URL / MIN_BYTES) if disk/bandwidth is tight.
 */
const DEFAULT_GGML_NAME = "ggml-base.bin";
const HF_GGML_BASE =
  "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";

/** Resolve download URL from model filename unless `SUBTITLE_WHISPER_CPP_MODEL_URL` is set. */
export function whisperCppDefaultModelUrl(modelName: string): string {
  const name = modelName.trim() || DEFAULT_GGML_NAME;
  return `${HF_GGML_BASE}/${name}`;
}

/**
 * Size floor by model family so a switch to base does not accept a leftover
 * tiny file, and an override to tiny still works without raising MIN_BYTES.
 */
export function whisperCppDefaultMinBytes(modelName: string): number {
  const n = (modelName || DEFAULT_GGML_NAME).toLowerCase();
  if (n.includes("tiny")) return 70 * 1024 * 1024; // ~75 MB
  if (n.includes("base")) return 140 * 1024 * 1024; // ~142 MB
  if (n.includes("small")) return 450 * 1024 * 1024; // ~466 MB
  if (n.includes("medium")) return 1400 * 1024 * 1024;
  if (n.includes("large")) return 2800 * 1024 * 1024;
  return 140 * 1024 * 1024;
}

/** In-flight downloads keyed by absolute model path (one fetch per path). */
const downloadInflight = new Map<string, Promise<string>>();

export function whisperCppModelsDir(modelsRoot?: string): string {
  return path.join(modelsRoot ?? subtitleModelsRoot(), "whisper");
}

export function whisperCppModelName(): string {
  return process.env.SUBTITLE_WHISPER_CPP_MODEL?.trim() || DEFAULT_GGML_NAME;
}

export function whisperCppModelPath(modelsRoot?: string): string {
  return path.join(whisperCppModelsDir(modelsRoot), whisperCppModelName());
}

/** Minimum accepted ggml size (override with SUBTITLE_WHISPER_CPP_MIN_BYTES for tests). */
export function whisperCppMinBytes(): number {
  const raw = process.env.SUBTITLE_WHISPER_CPP_MIN_BYTES?.trim();
  if (raw) {
    const n = Number(raw);
    if (Number.isFinite(n) && n > 0) return Math.floor(n);
  }
  return whisperCppDefaultMinBytes(whisperCppModelName());
}

/** @internal test helper — how many coalesced downloads are running. */
export function whisperCppDownloadInflightCount(): number {
  return downloadInflight.size;
}

/** @internal test helper — drop in-flight map (does not abort network). */
export function resetWhisperCppDownloadInflight(): void {
  downloadInflight.clear();
}

export async function whisperCliAvailable(): Promise<boolean> {
  try {
    const bin = resolveWhisperCliBin();
    // Absolute brew/override path that exists counts even if `-h` quirks.
    if (bin.includes("/") || bin.includes("\\")) {
      try {
        await access(bin);
      } catch {
        return false;
      }
    }
    // Absolute path preferred; bare name only counts if spawn succeeds.
    const { code } = await runCmd(bin, ["-h"]);
    // whisper-cli prints usage and may exit 0 or 1 depending on version.
    return code === 0 || code === 1;
  } catch {
    return false;
  }
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

export async function fileLooksComplete(filePath: string): Promise<boolean> {
  try {
    const st = await stat(filePath);
    return st.isFile() && st.size >= whisperCppMinBytes();
  } catch {
    return false;
  }
}

/**
 * Remove undersized / corrupt ggml leftovers so the next ensure re-downloads.
 * Returns true when a file was deleted.
 */
export async function removeIncompleteWhisperCppModel(
  filePath: string,
): Promise<boolean> {
  try {
    const st = await stat(filePath);
    if (!st.isFile()) return false;
    if (st.size >= whisperCppMinBytes()) return false;
    console.warn(
      `[subtitle] whisper.cpp removing incomplete ggml (${st.size} bytes) → ${filePath}`,
    );
    await rm(filePath, { force: true });
    return true;
  } catch {
    return false;
  }
}

/**
 * Cross-process advisory lock so two hub processes do not race the same ggml.
 * Best-effort: stale locks older than 30 min are broken.
 */
async function withModelDownloadLock<T>(
  modelPath: string,
  fn: () => Promise<T>,
): Promise<T> {
  const lockPath = `${modelPath}.lock`;
  const staleMs = 30 * 60 * 1000;
  const started = Date.now();
  // eslint-disable-next-line no-constant-condition
  while (true) {
    let fd: number | null = null;
    try {
      fd = openSync(lockPath, "wx");
      try {
        return await fn();
      } finally {
        try {
          closeSync(fd);
        } catch {
          /* ignore */
        }
        await rm(lockPath, { force: true }).catch(() => undefined);
      }
    } catch (err) {
      if (fd != null) {
        try {
          closeSync(fd);
        } catch {
          /* ignore */
        }
      }
      const code =
        err && typeof err === "object" && "code" in err
          ? String((err as { code?: unknown }).code)
          : "";
      if (code !== "EEXIST") {
        throw err;
      }
      // Another process holds the lock — wait, or break if stale.
      try {
        const st = await stat(lockPath);
        if (Date.now() - st.mtimeMs > staleMs) {
          await rm(lockPath, { force: true }).catch(() => undefined);
          continue;
        }
      } catch {
        // lock vanished — retry acquire
        continue;
      }
      if (Date.now() - started > staleMs) {
        throw new Error("model_download_failed: whisper.cpp lock timeout");
      }
      await new Promise((r) => setTimeout(r, 500));
      // Peer may have finished the file while we waited.
      if (await fileLooksComplete(modelPath)) {
        return modelPath as unknown as T;
      }
    }
  }
}

async function downloadWhisperCppModelOnce(modelPath: string): Promise<string> {
  await mkdir(path.dirname(modelPath), { recursive: true });

  // Drop undersized leftovers from older MIN thresholds / killed downloads.
  await removeIncompleteWhisperCppModel(modelPath);
  if (await fileLooksComplete(modelPath)) {
    return modelPath;
  }

  return withModelDownloadLock(modelPath, async () => {
    // Re-check under lock — peer may have finished.
    await removeIncompleteWhisperCppModel(modelPath);
    if (await fileLooksComplete(modelPath)) {
      return modelPath;
    }

    const url =
      process.env.SUBTITLE_WHISPER_CPP_MODEL_URL?.trim() ||
      whisperCppDefaultModelUrl(path.basename(modelPath));
    // Unique tmp so we never `rm` another job's in-progress stream.
    const tmpPath = `${modelPath}.part-${process.pid}-${Date.now()}-${Math.random().toString(16).slice(2)}.tmp`;
    await rm(tmpPath, { force: true }).catch(() => undefined);

    console.log(`[subtitle] whisper.cpp downloading ggml → ${modelPath}`);
    let res: Response;
    try {
      res = await fetch(url, { redirect: "follow" });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new Error(`model_download_failed: whisper.cpp ${msg}`.slice(0, 180));
    }
    if (!res.ok || !res.body) {
      throw new Error(
        `model_download_failed: whisper.cpp HTTP ${res.status}`.slice(0, 180),
      );
    }

    const contentLength = Number(res.headers.get("content-length") || 0);

    try {
      const nodeStream = Readable.fromWeb(
        res.body as import("node:stream/web").ReadableStream,
      );
      await pipeline(nodeStream, createWriteStream(tmpPath));
    } catch (err) {
      await rm(tmpPath, { force: true }).catch(() => undefined);
      const msg = err instanceof Error ? err.message : String(err);
      throw new Error(`model_download_failed: whisper.cpp ${msg}`.slice(0, 180));
    }

    let stSize = 0;
    try {
      stSize = (await stat(tmpPath)).size;
    } catch {
      await rm(tmpPath, { force: true }).catch(() => undefined);
      throw new Error("model_download_failed: whisper.cpp missing temp file");
    }

    if (contentLength > 0 && stSize !== contentLength) {
      await rm(tmpPath, { force: true }).catch(() => undefined);
      throw new Error(
        `model_download_failed: whisper.cpp size ${stSize} != Content-Length ${contentLength}`.slice(
          0,
          180,
        ),
      );
    }
    if (stSize < whisperCppMinBytes()) {
      await rm(tmpPath, { force: true }).catch(() => undefined);
      throw new Error(
        `model_download_failed: whisper.cpp incomplete download (${stSize} bytes)`.slice(
          0,
          180,
        ),
      );
    }

    await rm(modelPath, { force: true }).catch(() => undefined);
    await rename(tmpPath, modelPath);

    if (!(await fileLooksComplete(modelPath))) {
      await rm(modelPath, { force: true }).catch(() => undefined);
      throw new Error("model_download_failed: whisper.cpp rename incomplete");
    }

    console.log(
      `[subtitle] whisper.cpp ggml ready (${stSize} bytes) → ${modelPath}`,
    );
    return modelPath;
  });
}

/**
 * Download ggml (default **base**) once into `data/models/whisper/` (gitignored).
 * Skips when a complete file already exists. Concurrent callers share one
 * in-flight promise; cross-process callers serialize on a `.lock` file.
 * ASR must not start until this resolves with a size-valid model.
 */
export async function ensureWhisperCppModel(
  onStatus?: (status: "downloading") => void,
): Promise<string> {
  const dirs = await ensureSubtitleDirs();
  const modelPath = whisperCppModelPath(dirs.models);
  await mkdir(path.dirname(modelPath), { recursive: true });

  await removeIncompleteWhisperCppModel(modelPath);
  if (await fileLooksComplete(modelPath)) {
    return modelPath;
  }

  onStatus?.("downloading");

  let pending = downloadInflight.get(modelPath);
  if (!pending) {
    pending = downloadWhisperCppModelOnce(modelPath).finally(() => {
      // Only clear if we are still the mapped promise (avoid clobbering a retry).
      if (downloadInflight.get(modelPath) === pending) {
        downloadInflight.delete(modelPath);
      }
    });
    downloadInflight.set(modelPath, pending);
  }

  return pending;
}

export type WhisperCppParsed = {
  cues: SubtitleCue[];
  /** Detected spoken language from whisper-cli JSON (`result.language`), if any. */
  language: string | null;
};

/** Map ProMedia / BCP-47-ish tags to whisper-cli `-l` codes. */
export function whisperCliLanguageArg(
  languageHint?: string | null,
): string {
  // whisper-cli defaults to "en" when `-l` is omitted — that breaks Spanish ASR.
  // Always pass an explicit code; prefer auto-detect unless a clear hint exists.
  if (!languageHint) return "auto";
  const l = languageHint.toLowerCase();
  if (l === "auto") return "auto";
  if (l.startsWith("zh")) return "zh";
  if (l.startsWith("pt")) return "pt";
  if (l.startsWith("es")) return "es";
  if (l.startsWith("fr")) return "fr";
  if (l.startsWith("en")) return "en";
  // Unknown hint → auto rather than risk forcing English.
  return "auto";
}

/** Parse whisper-cli JSON (`-oj`) into cues + detected language. */
export function parseWhisperCppResult(raw: string): WhisperCppParsed {
  let data: unknown;
  try {
    data = JSON.parse(raw);
  } catch {
    return { cues: [], language: null };
  }
  const root = data as {
    result?: { language?: string };
    params?: { language?: string };
    transcription?: Array<{
      text?: string;
      offsets?: { from?: number; to?: number };
      timestamps?: { from?: string; to?: string };
    }>;
    segments?: Array<{
      text?: string;
      start?: number;
      end?: number;
      t0?: number;
      t1?: number;
    }>;
  };

  const languageRaw =
    (typeof root.result?.language === "string" && root.result.language) ||
    null;
  const language = languageRaw
    ? whisperCliLanguageArg(languageRaw) === "auto"
      ? languageRaw.slice(0, 8).toLowerCase()
      : whisperCliLanguageArg(languageRaw)
    : null;

  const cues: SubtitleCue[] = [];
  let i = 0;

  const push = (text: string, startMs: number, endMs: number) => {
    const t = cleanCueText(text);
    if (!t) return;
    let start = Math.max(0, Math.round(startMs));
    let end = Math.max(0, Math.round(endMs));
    if (end <= start) end = start + 1500;
    cues.push({ id: `c${i++}`, startMs: start, endMs: end, text: t });
  };

  if (Array.isArray(root.transcription)) {
    for (const seg of root.transcription) {
      const from = seg.offsets?.from;
      const to = seg.offsets?.to;
      if (typeof from === "number" && typeof to === "number") {
        push(seg.text ?? "", from, to);
        continue;
      }
      const startMs = parseVttClock(seg.timestamps?.from ?? "");
      const endMs = parseVttClock(seg.timestamps?.to ?? "");
      if (startMs != null && endMs != null) {
        push(seg.text ?? "", startMs, endMs);
      }
    }
  } else if (Array.isArray(root.segments)) {
    for (const seg of root.segments) {
      const start =
        typeof seg.start === "number"
          ? seg.start * 1000
          : typeof seg.t0 === "number"
            ? seg.t0 * 10 // centiseconds in some builds
            : 0;
      const end =
        typeof seg.end === "number"
          ? seg.end * 1000
          : typeof seg.t1 === "number"
            ? seg.t1 * 10
            : start + 2000;
      push(seg.text ?? "", start, end);
    }
  }

  return { cues, language };
}

/** Parse whisper-cli JSON (`-oj`) segments into cues. */
export function parseWhisperCppJson(raw: string): SubtitleCue[] {
  return parseWhisperCppResult(raw).cues;
}

/**
 * Parse WebVTT / SRT timestamp files (whisper-cli, yt-dlp, sidecars).
 * Accepts HH:MM:SS.mmm, H:MM:SS.mmm, and MM:SS.mmm (YouTube often omits hours
 * or uses a single-digit hour). Trailing cue settings (`align:start` …) are
 * ignored. Missing flexible clocks used to yield **zero cues** while the agent
 * still reported ready / fell through to Whisper — blank Preview/Output.
 */
export function parseWhisperCppVttOrSrt(raw: string): SubtitleCue[] {
  const cues: SubtitleCue[] = [];
  let i = 0;
  // Clock: optional hours, 1–2 digit hour, comma or dot millis (2–3 digits).
  const clock = String.raw`(?:\d{1,2}:)?\d{1,2}:\d{2}[,.]\d{2,3}`;
  const re = new RegExp(`(${clock})\\s*-->\\s*(${clock})`, "g");
  const lines = raw.replace(/\r\n/g, "\n").split("\n");
  for (let li = 0; li < lines.length; li++) {
    const line = lines[li] ?? "";
    // Strip cue settings after the end clock so the match stays clean.
    const timingLine = line.replace(/\s+(align|position|size|line):.+$/i, "");
    const m = re.exec(timingLine);
    re.lastIndex = 0;
    if (!m) continue;
    const startMs = parseVttClock(m[1]!.replace(",", "."));
    const endMs = parseVttClock(m[2]!.replace(",", "."));
    if (startMs == null || endMs == null) continue;
    const textLines: string[] = [];
    for (let j = li + 1; j < lines.length; j++) {
      const textLine = lines[j] ?? "";
      if (!textLine.trim() || /^\d+$/.test(textLine.trim()) || re.test(textLine)) {
        re.lastIndex = 0;
        if (/^\d+$/.test(textLine.trim())) continue;
        if (!textLine.trim()) break;
        if (/-->/.test(textLine)) break;
      }
      re.lastIndex = 0;
      if (/^(WEBVTT|NOTE|STYLE|REGION)\b/i.test(textLine)) continue;
      textLines.push(textLine);
    }
    const text = cleanCueText(textLines.join(" "));
    if (!text) continue;
    let end = endMs;
    if (end <= startMs) end = startMs + 1500;
    cues.push({
      id: `c${i++}`,
      startMs: startMs,
      endMs: end,
      text,
    });
  }
  return cues;
}

/** @internal exported for smokes — ms from VTT/SRT clock strings. */
export function parseVttClock(clock: string): number | null {
  const t = clock.trim().replace(",", ".");
  // HH:MM:SS.mmm or H:MM:SS.mm(m)
  let m = /^(\d{1,2}):(\d{1,2}):(\d{2})\.(\d{2,3})$/.exec(t);
  if (m) {
    const h = Number(m[1]);
    const min = Number(m[2]);
    const s = Number(m[3]);
    const frac = m[4]!;
    const ms = Number(frac.length === 2 ? `${frac}0` : frac);
    if (![h, min, s, ms].every((n) => Number.isFinite(n))) return null;
    return ((h * 60 + min) * 60 + s) * 1000 + ms;
  }
  // MM:SS.mmm (no hours — common in YouTube WebVTT)
  m = /^(\d{1,2}):(\d{2})\.(\d{2,3})$/.exec(t);
  if (m) {
    const min = Number(m[1]);
    const s = Number(m[2]);
    const frac = m[3]!;
    const ms = Number(frac.length === 2 ? `${frac}0` : frac);
    if (![min, s, ms].every((n) => Number.isFinite(n))) return null;
    return (min * 60 + s) * 1000 + ms;
  }
  return null;
}

/**
 * Transcribe a 16 kHz mono WAV with brew `whisper-cli` / whisper.cpp.
 * Throws `whisper_cpp_missing` when the binary is absent.
 * Waits for a size-valid ggml before spawning ASR (no parallel incomplete downloads).
 */
export async function transcribeWithWhisperCpp(options: {
  wavPath: string;
  translateToEn: boolean;
  /** Spoken-language hint for `-l` (e.g. targetLang when same-language captions). */
  languageHint?: string | null;
  /**
   * Optional initial prompt (karaoke timing path): bias Whisper toward
   * official cancionero vocabulary. Clocks only — cue text is discarded
   * after forced align. Older whisper-cli builds may reject `-prompt`.
   */
  prompt?: string | null;
  workDir: string;
  onStatus?: (status: "downloading" | "transcribing") => void;
}): Promise<{
  cues: SubtitleCue[];
  whisperTranslatedToEn: boolean;
  sourceLang: string | null;
}> {
  const bin = resolveWhisperCliBin();
  // Must resolve a complete model before any ASR spawn.
  const modelPath = await ensureWhisperCppModel((st) => options.onStatus?.(st));
  if (!(await fileLooksComplete(modelPath))) {
    throw new Error("model_download_failed: whisper.cpp model not ready");
  }

  options.onStatus?.("transcribing");
  await mkdir(options.workDir, { recursive: true });
  const outBase = path.join(options.workDir, "whisper-cpp-out");
  // Clean previous outputs
  for (const ext of [".json", ".vtt", ".srt", ".txt"]) {
    await rm(`${outBase}${ext}`, { force: true }).catch(() => undefined);
  }

  // whisper-cli defaults `-l en` — must pass auto/es/… or Spanish audio yields empty/garbage.
  const lang = whisperCliLanguageArg(options.languageHint ?? "auto");

  // Prefer suppressing non-speech tokens (`-sns`) so music videos yield lyrics
  // instead of `[MUSIC]` / `[MÚSICA]`. `-ml` + `-sow` keep segments at
  // cancionero line length (gold-standard hymn timing) instead of late
  // paragraphs. Older whisper-cli builds may reject optional flags.
  const lyricMaxLen = Number(
    process.env.SUBTITLE_WHISPER_MAX_LEN?.trim() || "48",
  );
  const promptText = options.prompt?.trim() || "";

  const buildArgs = (optional: {
    np: boolean;
    sns: boolean;
    lyricSeg: boolean;
    prompt: boolean;
  }): string[] => {
    const args = [
      "-m",
      modelPath,
      "-f",
      options.wavPath,
      "-of",
      outBase,
      "-oj",
      "-ovtt",
      "-osrt",
      "-nt", // no timestamps in stdout (cleaner)
      "-l",
      lang,
    ];
    if (optional.np) args.push("-np");
    if (optional.sns) args.push("-sns");
    if (
      optional.lyricSeg &&
      Number.isFinite(lyricMaxLen) &&
      lyricMaxLen > 0
    ) {
      args.push("-ml", String(Math.floor(lyricMaxLen)), "-sow");
    }
    if (optional.prompt && promptText) {
      args.push("-prompt", promptText);
    }
    if (options.translateToEn) {
      args.push("-tr");
    }
    return args;
  };

  let code: number;
  let stderr: string;
  let args = buildArgs({
    np: true,
    sns: true,
    lyricSeg: true,
    prompt: Boolean(promptText),
  });
  try {
    ({ code, stderr } = await runCmd(bin, args));
  } catch (err) {
    if (missingBinHintFromError(err, "whisper_cpp")) {
      throw new Error("whisper_cpp_missing");
    }
    const msg = err instanceof Error ? err.message : String(err);
    if (/enoent|not found/i.test(msg)) {
      throw new Error("whisper_cpp_missing");
    }
    throw err;
  }

  if (
    code !== 0 &&
    /-(?:np|sns|ml|sow|prompt)|max-len|split-on-word|unknown|unrecognized/i.test(
      stderr,
    )
  ) {
    // Older builds may reject optional flags — peel them off in stages.
    const retries = [
      { np: true, sns: true, lyricSeg: true, prompt: false },
      { np: true, sns: true, lyricSeg: false, prompt: false },
      { np: false, sns: false, lyricSeg: false, prompt: false },
    ] as const;
    for (const opt of retries) {
      args = buildArgs(opt);
      const next = await runCmd(bin, args);
      code = next.code;
      stderr = next.stderr;
      if (code === 0) break;
      if (
        !/-(?:np|sns|ml|sow|prompt)|max-len|split-on-word|unknown|unrecognized/i.test(
          stderr,
        )
      ) {
        break;
      }
    }
  }

  if (code !== 0) {
    if (
      /not found|no such file|enoent|unable to find|cannot find/i.test(stderr) ||
      (/whisper-cli|whisper\.cpp/i.test(stderr) && /not found/i.test(stderr))
    ) {
      throw new Error("whisper_cpp_missing");
    }
    const tail = stderr.split("\n").slice(-8).join(" ").trim();
    throw new Error(
      `whisper_cpp_failed: ${tail || `exit ${code}`}`.slice(0, 180),
    );
  }

  let cues: SubtitleCue[] = [];
  let sourceLang: string | null = null;
  try {
    const jsonRaw = await readFile(`${outBase}.json`, "utf8");
    const parsed = parseWhisperCppResult(jsonRaw);
    cues = parsed.cues;
    sourceLang = parsed.language;
  } catch {
    // fall through
  }
  if (!cues.length) {
    try {
      const vtt = await readFile(`${outBase}.vtt`, "utf8");
      cues = parseWhisperCppVttOrSrt(vtt);
    } catch {
      // fall through
    }
  }
  if (!cues.length) {
    try {
      const srt = await readFile(`${outBase}.srt`, "utf8");
      cues = parseWhisperCppVttOrSrt(srt);
    } catch {
      // fall through
    }
  }
  if (!cues.length) {
    try {
      const txt = (await readFile(`${outBase}.txt`, "utf8")).trim();
      if (txt) {
        cues = [{ id: "c0", startMs: 0, endMs: 4000, text: txt }];
      }
    } catch {
      // empty
    }
  }

  if (options.translateToEn) {
    sourceLang = "en";
  }

  // Drop `[MUSIC]` / sound labels, then shape toward official lyric density
  // (short lines, punctuation, no late gaps vs previous cue). Adelante lead
  // and intro-hold stay on the overlay path — we only fix cue payload here.
  cues = shapeLyricLikeCues(filterSpeechCues(cues));

  return {
    cues,
    whisperTranslatedToEn: options.translateToEn,
    sourceLang,
  };
}

export async function assertWhisperCliOrThrow(): Promise<string> {
  const bin = resolveWhisperCliBin();
  try {
    await access(bin);
  } catch {
    // bare name may still be on PATH after augment — probe via -h
  }
  try {
    const { code, stderr } = await runCmd(bin, ["-h"]);
    if (code !== 0 && code !== 1 && /not found|enoent/i.test(stderr)) {
      throw new Error("whisper_cpp_missing");
    }
    // If spawn worked at all, treat as present (usage exits 0/1).
    return bin;
  } catch (err) {
    if (missingBinHintFromError(err, "whisper_cpp")) {
      throw new Error("whisper_cpp_missing");
    }
    const msg = err instanceof Error ? err.message : String(err);
    if (/enoent|not found/i.test(msg)) {
      throw new Error("whisper_cpp_missing");
    }
    throw new Error("whisper_cpp_missing");
  }
}
