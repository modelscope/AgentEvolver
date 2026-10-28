import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import { access, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import wavefile from "wavefile";
import type { SubtitleCue, SubtitleTargetLang } from "@promedia/shared";
import {
  ensureAugmentedPath,
  missingBinHintFromError,
  resolveFfmpegBin,
} from "./subtitle-bins.js";
import {
  applyTransformersCacheEnv,
  clearCorruptModelCache,
  ensureSubtitleDirs,
  subtitleModelsRoot,
} from "./subtitle-paths.js";
import { filterSpeechCues } from "./subtitle-cue-text.js";
import { shapeLyricLikeCues } from "./subtitle-lyric-shape.js";
import {
  assertWhisperCliOrThrow,
  transcribeWithWhisperCpp,
  whisperCliAvailable,
} from "./subtitle-whisper-cpp.js";

export { whisperCliAvailable };

const { WaveFile } = wavefile as unknown as {
  WaveFile: new (buf?: Buffer | Uint8Array) => {
    fmt: object;
    toBitDepth: (depth: string) => void;
    toSampleRate: (rate: number) => void;
    getSamples: (
      interleaved: boolean,
      Container?: Float32ArrayConstructor,
    ) => unknown;
  };
};

export type AsrResult = {
  cues: SubtitleCue[];
  sourceLang: string | null;
  /** True when Whisper ran with task=translate (always English). */
  whisperTranslatedToEn: boolean;
};

type CachedAsr = {
  assetId: string;
  sourceHash: string;
  sourceLang: string | null;
  whisperTranslatedToEn: boolean;
  cues: SubtitleCue[];
  createdAt: string;
};

function runCmd(
  bin: string,
  args: string[],
): Promise<{ code: number; stderr: string }> {
  ensureAugmentedPath();
  return new Promise((resolve, reject) => {
    const child = spawn(bin, args, {
      stdio: ["ignore", "ignore", "pipe"],
      env: process.env,
    });
    let stderr = "";
    child.stderr?.on("data", (chunk: Buffer) => {
      stderr += chunk.toString("utf8");
    });
    child.on("error", reject);
    child.on("close", (code) => {
      resolve({ code: code ?? 1, stderr });
    });
  });
}

export async function ffmpegAvailable(): Promise<boolean> {
  try {
    const bin = resolveFfmpegBin();
    const { code } = await runCmd(bin, ["-version"]);
    return code === 0;
  } catch {
    return false;
  }
}

/** Extract 16 kHz mono WAV for Whisper (local ffmpeg — no cloud). */
export async function extractWav16k(
  mediaPath: string,
  outWavPath: string,
): Promise<void> {
  const bin = resolveFfmpegBin();
  await mkdir(path.dirname(outWavPath), { recursive: true });
  // Drop any previous partial/corrupt output before remux.
  await rm(outWavPath, { force: true }).catch(() => undefined);

  let code: number;
  let stderr: string;
  try {
    // Safer remux: audio-only, mono, 16 kHz PCM (Whisper). Avoid copy/codec
    // passthrough that fails on partial AAC from incomplete YouTube downloads.
    ({ code, stderr } = await runCmd(bin, [
      "-hide_banner",
      "-nostdin",
      "-y",
      "-i",
      mediaPath,
      "-vn",
      "-ac",
      "1",
      "-ar",
      "16000",
      "-c:a",
      "pcm_s16le",
      "-f",
      "wav",
      outWavPath,
    ]));
  } catch (err) {
    await rm(outWavPath, { force: true }).catch(() => undefined);
    if (missingBinHintFromError(err, "ffmpeg")) {
      throw new Error("ffmpeg_missing");
    }
    throw err;
  }
  if (code !== 0) {
    await rm(outWavPath, { force: true }).catch(() => undefined);
    const tail = (stderr || "").split("\n").slice(-6).join(" ").trim();
    // Only treat as missing binary when ffmpeg itself is absent — not when
    // the input media is corrupt (AAC decode / Conversion failed).
    if (
      /ffmpeg.*(not found|no such file)|unable to (find|run) .*ffmpeg|cannot find.*ffmpeg/i.test(
        tail,
      ) ||
      (/^\s*$/.test(tail) && /enoent|not found/i.test(String(code)))
    ) {
      throw new Error("ffmpeg_missing");
    }
    throw new Error(
      `ffmpeg_extract_failed: ${tail || "conversion failed"}`.slice(0, 180),
    );
  }
}

async function fileSha256(filePath: string): Promise<string> {
  const buf = await readFile(filePath);
  return createHash("sha256").update(buf).digest("hex");
}

/**
 * Cache key v6: native agent — whisper `-l <targetLang>` (not auto) and no
 * translate-from-English retry for non-EN UI. Busts v5 auto/Marian-era caches.
 */
function cacheKey(
  assetId: string,
  targetLang: SubtitleTargetLang,
  translateToEn: boolean,
  prompt?: string | null,
): string {
  const base = `${assetId}_${translateToEn ? "en" : "src"}_${targetLang}_v7`;
  const p = prompt?.trim();
  if (!p) return base;
  const hash = createHash("sha256").update(p).digest("hex").slice(0, 8);
  return `${base}_p${hash}`;
}

function normalizeChunks(
  chunks: Array<{ text?: string; timestamp?: [number, number] | number }>,
): SubtitleCue[] {
  const cues: SubtitleCue[] = [];
  let i = 0;
  for (const chunk of chunks) {
    const text = (chunk.text ?? "").replace(/\s+/g, " ").trim();
    if (!text) continue;
    let startSec = 0;
    let endSec = 0;
    const ts = chunk.timestamp;
    if (Array.isArray(ts)) {
      startSec = typeof ts[0] === "number" ? ts[0] : 0;
      endSec = typeof ts[1] === "number" ? ts[1] : startSec + 2;
    } else if (typeof ts === "number") {
      startSec = ts;
      endSec = ts + 2;
    } else {
      startSec = cues.length ? cues[cues.length - 1]!.endMs / 1000 : 0;
      endSec = startSec + 2;
    }
    if (endSec <= startSec) {
      endSec = startSec + 1.5;
    }
    cues.push({
      id: `c${i++}`,
      startMs: Math.max(0, Math.round(startSec * 1000)),
      endMs: Math.max(0, Math.round(endSec * 1000)),
      text,
    });
  }
  return filterSpeechCues(cues);
}

type WhisperPipeline = (
  input: Float32Array | Float32Array[],
  options?: Record<string, unknown>,
) => Promise<{
  text?: string;
  chunks?: Array<{ text?: string; timestamp?: [number, number] | number }>;
}>;

/** Decode 16 kHz mono WAV → Float32Array for transformers.js (no AudioContext in Node). */
async function wavToFloat32(wavPath: string): Promise<Float32Array> {
  const buf = await readFile(wavPath);
  const wav = new WaveFile(buf);
  wav.toBitDepth("32f");
  const fmt = wav.fmt as { sampleRate?: number };
  if (Number(fmt.sampleRate) !== 16000) {
    wav.toSampleRate(16000);
  }
  const raw = wav.getSamples(false, Float32Array) as unknown;
  if (Array.isArray(raw)) {
    const first = raw[0] as Float32Array | undefined;
    return first ?? new Float32Array(0);
  }
  if (raw instanceof Float32Array) {
    return raw;
  }
  // wavefile may return Float64Array depending on bit depth helpers
  const asNum = raw as ArrayLike<number>;
  const out = new Float32Array(asNum.length);
  for (let i = 0; i < asNum.length; i++) {
    out[i] = Number(asNum[i]);
  }
  return out;
}

/** Default Xenova whisper-tiny; pin revision so ONNX layout stays stable. */
const DEFAULT_WHISPER_MODEL = "Xenova/whisper-tiny";
/** Known-good revision with onnx/ weights (override: SUBTITLE_WHISPER_REVISION). */
const DEFAULT_WHISPER_REVISION = "main";

let whisperPromise: Promise<WhisperPipeline> | null = null;

function whisperModelId(): string {
  return process.env.SUBTITLE_WHISPER_MODEL?.trim() || DEFAULT_WHISPER_MODEL;
}

function whisperRevision(): string {
  return (
    process.env.SUBTITLE_WHISPER_REVISION?.trim() || DEFAULT_WHISPER_REVISION
  );
}

/**
 * Map transformers / download failures to Control-facing hint codes.
 * - model_download_failed: network / HF hub unreachable
 * - model_path_error: cache path / buffer / corrupt local files
 */
export function classifyWhisperLoadError(err: unknown): string {
  const msg = err instanceof Error ? err.message : String(err ?? "");
  const lower = msg.toLowerCase();
  if (
    /fetch failed|econnrefused|enotfound|etimedout|econnreset|network|socket|getaddrinfo|und_err|certificate|ssl|tls|huggingface\.co|status (?:4\d\d|5\d\d)|403|429|502|503|504|download/i.test(
      msg,
    ) ||
    /network|offline|dns|timeout/.test(lower)
  ) {
    return "model_download_failed";
  }
  if (
    /unable to get model file path or buffer|local_files_only|unable to load from local path|ENOENT|EACCES|EPERM|EISDIR|ENOSPC/i.test(
      msg,
    ) ||
    /model file path|file path or buffer/i.test(lower)
  ) {
    return "model_path_error";
  }
  return msg.slice(0, 180) || "asr_failed";
}

async function loadWhisperPipeline(attempt: number): Promise<WhisperPipeline> {
  const dirs = await ensureSubtitleDirs();
  const modelsRoot = dirs.models || subtitleModelsRoot();
  const modelId = whisperModelId();
  const revision = whisperRevision();

  // Dynamic import keeps `tsc` / cold start light.
  const transformers = await import("@huggingface/transformers");
  applyTransformersCacheEnv(transformers.env, modelsRoot);

  console.log(
    `[subtitle] Whisper cacheDir=${transformers.env.cacheDir} localModelPath=${transformers.env.localModelPath} model=${modelId}@${revision} attempt=${attempt}`,
  );

  const pipe = await transformers.pipeline(
    "automatic-speech-recognition",
    modelId,
    {
      revision,
      // tiny quantized ONNX is the Xenova default and stays CPU-friendly
      dtype: "q8",
    },
  );
  return pipe as unknown as WhisperPipeline;
}

async function getWhisper(): Promise<WhisperPipeline> {
  if (!whisperPromise) {
    whisperPromise = (async () => {
      try {
        return await loadWhisperPipeline(1);
      } catch (firstErr) {
        const firstMsg =
          firstErr instanceof Error ? firstErr.message : String(firstErr);
        console.warn(
          `[subtitle] Whisper load failed (will clear corrupt cache + retry once): ${firstMsg}`,
        );
        try {
          const cleared = await clearCorruptModelCache(
            subtitleModelsRoot(),
            whisperModelId(),
          );
          console.warn(
            `[subtitle] cleared model cache tmp=${cleared.removedTmp} modelDir=${cleared.removedModelDir}`,
          );
        } catch (clearErr) {
          console.warn(
            "[subtitle] cache clear failed:",
            clearErr instanceof Error ? clearErr.message : clearErr,
          );
        }
        try {
          return await loadWhisperPipeline(2);
        } catch (secondErr) {
          // Reset so a later job can try again after network/path is fixed.
          whisperPromise = null;
          throw new Error(classifyWhisperLoadError(secondErr));
        }
      }
    })();
  }
  return whisperPromise;
}

type AsrBackend = "auto" | "transformers" | "whisper_cpp";

function asrBackendPref(): AsrBackend {
  const raw = (process.env.SUBTITLE_ASR_BACKEND || "auto").trim().toLowerCase();
  if (raw === "transformers" || raw === "whisper_cpp" || raw === "auto") {
    return raw;
  }
  return "auto";
}

/** Exported for smoke tests — documents the exclusive whisper.cpp contract. */
export function asrBackendPreference(): AsrBackend {
  return asrBackendPref();
}

async function transcribeWithTransformers(options: {
  wavPath: string;
  translateToEn: boolean;
  languageHint?: string | null;
  onStatus?: (status: "downloading" | "transcribing") => void;
}): Promise<AsrResult> {
  console.log(
    "[subtitle] ASR backend=transformers (SUBTITLE_ASR_BACKEND=transformers)",
  );
  options.onStatus?.("downloading");
  const whisper = await getWhisper();
  const audio = await wavToFloat32(options.wavPath);

  options.onStatus?.("transcribing");
  const language = options.translateToEn
    ? "auto"
    : options.languageHint || "auto";
  const result = await whisper(audio, {
    return_timestamps: true,
    chunk_length_s: 30,
    stride_length_s: 5,
    task: options.translateToEn ? "translate" : "transcribe",
    language,
    sampling_rate: 16000,
  });

  let cues = normalizeChunks(result.chunks ?? []);
  if (!cues.length && result.text?.trim()) {
    cues = [
      {
        id: "c0",
        startMs: 0,
        endMs: 4000,
        text: result.text.trim(),
      },
    ];
  }

  return {
    cues,
    sourceLang: options.translateToEn
      ? "en"
      : options.languageHint && options.languageHint !== "auto"
        ? options.languageHint
        : null,
    whisperTranslatedToEn: options.translateToEn,
  };
}

async function transcribeViaWhisperCpp(options: {
  wavPath: string;
  translateToEn: boolean;
  languageHint?: string | null;
  /** Karaoke timing: bias clocks toward official lyrics (text discarded). */
  prompt?: string | null;
  assetId: string;
  onStatus?: (status: "downloading" | "transcribing") => void;
}): Promise<AsrResult> {
  const bin = await assertWhisperCliOrThrow();
  console.log(`[subtitle] ASR backend=whisper_cpp bin=${bin}`);
  const dirs = await ensureSubtitleDirs();
  const workDir = path.join(dirs.wav, `wcpp_${options.assetId}`);
  const result = await transcribeWithWhisperCpp({
    wavPath: options.wavPath,
    translateToEn: options.translateToEn,
    languageHint: options.languageHint,
    prompt: options.prompt,
    workDir,
    onStatus: options.onStatus,
  });
  return {
    cues: result.cues,
    sourceLang: result.sourceLang,
    whisperTranslatedToEn: result.whisperTranslatedToEn,
  };
}

/**
 * Local ASR. Default (`auto`): whisper.cpp exclusively when `whisper-cli` is
 * on PATH (Homebrew Mac). Never loads Xenova/ONNX in that mode — Mac hits
 * `model_path_error` on transformers.js. Missing CLI → `whisper_cpp_missing`
 * with brew install hint. Opt-in only: `SUBTITLE_ASR_BACKEND=transformers`.
 */
export async function transcribeLocalFile(options: {
  mediaPath: string;
  assetId: string;
  /** When true, Whisper task=translate → English cues. */
  translateToEn: boolean;
  targetLang: SubtitleTargetLang;
  /**
   * Karaoke aligner: optional Whisper prompt from cancionero lines.
   * Improves timing clocks only — callers must never display ASR text.
   */
  prompt?: string | null;
  onStatus?: (status: "downloading" | "extracting" | "transcribing") => void;
}): Promise<AsrResult> {
  const { mediaPath, assetId, translateToEn, targetLang, onStatus } = options;
  const prompt = options.prompt?.trim() || null;
  const dirs = await ensureSubtitleDirs();
  const sourceHash = await fileSha256(mediaPath);
  const key = cacheKey(assetId, targetLang, translateToEn, prompt);
  const cachePath = path.join(dirs.cache, `${key}.json`);

  try {
    const raw = await readFile(cachePath, "utf8");
    const cached = JSON.parse(raw) as CachedAsr;
    if (
      cached &&
      cached.sourceHash === sourceHash &&
      Array.isArray(cached.cues)
    ) {
      const cues = shapeLyricLikeCues(filterSpeechCues(cached.cues));
      // Ignore stale music-label-only caches (showed as `[MÚSICA]` only).
      if (cues.length > 0) {
        return {
          cues,
          sourceLang: cached.sourceLang,
          whisperTranslatedToEn: cached.whisperTranslatedToEn,
        };
      }
    }
  } catch {
    // miss
  }

  onStatus?.("extracting");
  const wavPath = path.join(dirs.wav, `${assetId}.wav`);
  await extractWav16k(mediaPath, wavPath);

  const backend = asrBackendPref();
  let asr: AsrResult;

  // Native agent: transcribe in the UI/target language (`-l es` for Spanish),
  // never force English then Marian. whisper-cli defaults to `-l en` when
  // omitted — always pass an explicit code.
  const languageHint = translateToEn ? "en" : targetLang;

  async function runAsr(forceTranslateToEn: boolean): Promise<AsrResult> {
    const hint = forceTranslateToEn ? "en" : languageHint;
    if (backend === "transformers") {
      try {
        return await transcribeWithTransformers({
          wavPath,
          translateToEn: forceTranslateToEn,
          languageHint: hint,
          onStatus,
        });
      } catch (tfErr) {
        throw new Error(classifyWhisperLoadError(tfErr));
      }
    }
    // auto | whisper_cpp: whisper-cli exclusively. Never fall into Xenova.
    const cliPresent = await whisperCliAvailable();
    if (!cliPresent) {
      console.warn(
        "[subtitle] ASR backend=whisper_cpp required but whisper-cli not found — install: brew install whisper-cpp (or set WHISPER_CPP_PATH). Not loading Xenova/ONNX.",
      );
      throw new Error("whisper_cpp_missing");
    }
    try {
      return await transcribeViaWhisperCpp({
        wavPath,
        translateToEn: forceTranslateToEn,
        languageHint: hint,
        prompt,
        assetId,
        onStatus,
      });
    } catch (cppErr) {
      const missing = missingBinHintFromError(cppErr, "whisper_cpp");
      if (missing) {
        throw new Error("whisper_cpp_missing");
      }
      const cppMsg = cppErr instanceof Error ? cppErr.message : String(cppErr);
      throw new Error(cppMsg.slice(0, 180));
    }
  }

  asr = await runAsr(translateToEn);

  // No `-tr` (translate→EN) retry for non-English UI — that path fed Marian
  // and mangled Spanish lyrics. Prefer captions; ASR stays `-l <targetLang>`.

  asr = {
    ...asr,
    cues: shapeLyricLikeCues(filterSpeechCues(asr.cues)),
  };

  // Only cache when we have real speech/lyric cues — never persist `[MUSIC]`-only.
  if (asr.cues.length > 0) {
    const payload: CachedAsr = {
      assetId,
      sourceHash,
      sourceLang: asr.sourceLang,
      whisperTranslatedToEn: asr.whisperTranslatedToEn,
      cues: asr.cues,
      createdAt: new Date().toISOString(),
    };
    await writeFile(cachePath, JSON.stringify(payload, null, 2), "utf8");
  }
  return asr;
}

export async function mediaFileExists(filePath: string): Promise<boolean> {
  try {
    await access(filePath);
    return true;
  } catch {
    return false;
  }
}
