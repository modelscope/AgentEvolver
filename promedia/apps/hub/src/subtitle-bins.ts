import { accessSync, constants, existsSync } from "node:fs";
import os from "node:os";
import path from "node:path";

/** Homebrew + Intel Mac prefixes often missing from GUI/launcher PATH. */
export const COMMON_MAC_BIN_DIRS = [
  "/opt/homebrew/bin",
  "/usr/local/bin",
] as const;

/** Linux package-manager prefixes (apt/dnf/pacman). */
export const COMMON_LINUX_BIN_DIRS = [
  "/usr/bin",
  "/usr/local/bin",
] as const;

const MISSING_BIN_HINTS = new Set([
  "yt_dlp_missing",
  "ffmpeg_missing",
  "whisper_cpp_missing",
]);

let pathAugmented = false;

/** Extra dirs to prepend so GUI launchers find brew / system tools. */
export function commonExtraBinDirs(): string[] {
  const platform = os.platform();
  if (platform === "darwin") {
    return [...COMMON_MAC_BIN_DIRS];
  }
  if (platform === "linux") {
    return [...COMMON_LINUX_BIN_DIRS];
  }
  // Windows: rely on PATH (Scoop/Chocolatey/winget usually register there).
  // Also try common user Scoop shims if present.
  if (platform === "win32") {
    const home = os.homedir();
    const scoop = path.join(home, "scoop", "shims");
    const dirs: string[] = [];
    if (existsSync(scoop)) dirs.push(scoop);
    const local = process.env.LOCALAPPDATA;
    if (local) {
      const winget = path.join(local, "Microsoft", "WinGet", "Links");
      if (existsSync(winget)) dirs.push(winget);
    }
    return dirs;
  }
  return [];
}

/** Prepend common OS bin dirs to `process.env.PATH` once (idempotent). */
export function ensureAugmentedPath(): void {
  if (pathAugmented) return;
  pathAugmented = true;
  const sep = path.delimiter;
  const parts = (process.env.PATH || "").split(sep).filter(Boolean);
  const extra = commonExtraBinDirs().filter((d) => !parts.includes(d));
  if (extra.length > 0) {
    process.env.PATH = [...extra, ...parts].join(sep);
  }
}

function isExecutable(filePath: string): boolean {
  try {
    // On Windows X_OK is unreliable; existence + F_OK is enough for .exe/.cmd.
    if (os.platform() === "win32") {
      accessSync(filePath, constants.F_OK);
      return true;
    }
    accessSync(filePath, constants.X_OK);
    return true;
  } catch {
    return false;
  }
}

/** Candidate filenames for a tool (Windows adds .exe / .cmd). */
function toolNameVariants(bareName: string): string[] {
  if (os.platform() !== "win32") return [bareName];
  const lower = bareName.toLowerCase();
  if (
    lower.endsWith(".exe") ||
    lower.endsWith(".cmd") ||
    lower.endsWith(".bat")
  ) {
    return [bareName];
  }
  return [bareName, `${bareName}.exe`, `${bareName}.cmd`];
}

/**
 * Resolve a CLI tool: env override → PATH (augmented) → common OS absolute paths.
 * Returns an absolute path when found; otherwise the bare name for spawn to try.
 */
export function resolveToolBin(envKey: string, bareName: string): string {
  ensureAugmentedPath();
  const override = process.env[envKey]?.trim();
  const candidates: string[] = [];
  if (override) {
    candidates.push(override);
  }
  for (const v of toolNameVariants(bareName)) {
    if (!candidates.includes(v)) candidates.push(v);
  }

  const pathDirs = (process.env.PATH || "").split(path.delimiter).filter(Boolean);
  const searchDirs = [...pathDirs];
  for (const d of commonExtraBinDirs()) {
    if (!searchDirs.includes(d)) searchDirs.push(d);
  }

  for (const name of candidates) {
    if (name.includes("/") || name.includes("\\")) {
      if (isExecutable(name)) return name;
      continue;
    }
    for (const dir of searchDirs) {
      const full = path.join(dir, name);
      if (isExecutable(full)) return full;
    }
  }

  return override || bareName;
}

export function resolveYtDlpBin(): string {
  return resolveToolBin("YT_DLP_PATH", "yt-dlp");
}

export function resolveFfmpegBin(): string {
  return resolveToolBin("FFMPEG_PATH", "ffmpeg");
}

/**
 * brew `whisper-cpp` installs `whisper-cli` (preferred). Older builds may
 * expose `whisper-cpp` or `whisper`. Override: `WHISPER_CPP_PATH`.
 */
export function resolveWhisperCliBin(): string {
  ensureAugmentedPath();
  const override = process.env.WHISPER_CPP_PATH?.trim();
  if (override) {
    if (isExecutable(override)) return override;
    return override;
  }

  const pathDirs = (process.env.PATH || "")
    .split(path.delimiter)
    .filter(Boolean);
  // Prefer common OS dirs first so GUI launchers without a shell PATH
  // still find brew/apt/scoop whisper-cli before bare PATH lookup.
  const searchDirs: string[] = [];
  for (const d of [...commonExtraBinDirs(), ...pathDirs]) {
    if (!searchDirs.includes(d)) searchDirs.push(d);
  }

  for (const bare of ["whisper-cli", "whisper-cpp", "whisper"] as const) {
    for (const name of toolNameVariants(bare)) {
      for (const dir of searchDirs) {
        const full = path.join(dir, name);
        if (isExecutable(full)) return full;
      }
    }
  }
  return "whisper-cli";
}

export function isMissingBinHint(hint: string | null | undefined): boolean {
  return Boolean(hint && MISSING_BIN_HINTS.has(hint));
}

/**
 * Map thrown errors to stable subtitleAgent hint codes.
 * Callers should throw `yt_dlp_missing` / `ffmpeg_missing` /
 * `whisper_cpp_missing` for ENOENT; this also accepts those substrings.
 * Bare ENOENT without a typed code is mapped with `kind` when provided.
 */
export function missingBinHintFromError(
  err: unknown,
  kind?: "yt_dlp" | "ffmpeg" | "whisper_cpp",
): string | null {
  const msg = err instanceof Error ? err.message : String(err);
  const lower = msg.toLowerCase();
  if (msg === "yt_dlp_missing" || lower.includes("yt_dlp_missing")) {
    return "yt_dlp_missing";
  }
  if (msg === "ffmpeg_missing" || lower.includes("ffmpeg_missing")) {
    return "ffmpeg_missing";
  }
  if (
    msg === "whisper_cpp_missing" ||
    lower.includes("whisper_cpp_missing")
  ) {
    return "whisper_cpp_missing";
  }
  if (
    kind &&
    (lower.includes("enoent") || lower.includes("not found"))
  ) {
    if (kind === "yt_dlp") return "yt_dlp_missing";
    if (kind === "ffmpeg") return "ffmpeg_missing";
    return "whisper_cpp_missing";
  }
  return null;
}
