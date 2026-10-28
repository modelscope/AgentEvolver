import { isSupportedTextImport } from "./textImport";

export type MediaFileKind = "image" | "video" | "audio" | "text" | "unsupported";

const IMAGE_EXTS = new Set([
  ".jpg",
  ".jpeg",
  ".png",
  ".webp",
  ".gif",
]);
const VIDEO_EXTS = new Set([".mp4", ".webm", ".mov"]);
const AUDIO_EXTS = new Set([
  ".mp3",
  ".m4a",
  ".aac",
  ".wav",
  ".flac",
  ".ogg",
  ".oga",
  ".opus",
]);

function extOf(name: string): string {
  const i = name.lastIndexOf(".");
  return i >= 0 ? name.slice(i).toLowerCase() : "";
}

/** Detect playlist type for a dropped/picked file (same rules as legacy pickers). */
export function classifyMediaFile(file: File): MediaFileKind {
  const mime = (file.type || "").toLowerCase();
  const ext = extOf(file.name);

  if (mime.startsWith("image/") || IMAGE_EXTS.has(ext)) {
    return "image";
  }
  if (mime.startsWith("video/") || VIDEO_EXTS.has(ext)) {
    return "video";
  }
  if (mime.startsWith("audio/") || AUDIO_EXTS.has(ext)) {
    return "audio";
  }
  if (isSupportedTextImport(file)) {
    return "text";
  }
  return "unsupported";
}

/** Accept attribute for the unified multi-file picker. */
export const UNIFIED_MEDIA_ACCEPT = [
  "image/jpeg",
  "image/png",
  "image/webp",
  "image/gif",
  ".jpg",
  ".jpeg",
  ".png",
  ".webp",
  ".gif",
  "video/mp4",
  "video/webm",
  "video/quicktime",
  ".mp4",
  ".webm",
  ".mov",
  "audio/mpeg",
  "audio/mp4",
  "audio/aac",
  "audio/wav",
  "audio/flac",
  "audio/ogg",
  "audio/*",
  ".mp3",
  ".m4a",
  ".aac",
  ".wav",
  ".flac",
  ".ogg",
  ".oga",
  ".opus",
  ".pdf",
  ".xml",
  ".txt",
  ".md",
  ".text",
  ".markdown",
  "application/pdf",
  "text/plain",
  "text/markdown",
  "application/xml",
  "text/xml",
].join(",");

/** Accept attribute for the Guion Audio solapa picker. */
export const AUDIO_MEDIA_ACCEPT = [
  "audio/*",
  "audio/mpeg",
  "audio/mp4",
  "audio/aac",
  "audio/wav",
  "audio/flac",
  "audio/ogg",
  ".mp3",
  ".m4a",
  ".aac",
  ".wav",
  ".flac",
  ".ogg",
  ".oga",
  ".opus",
].join(",");
