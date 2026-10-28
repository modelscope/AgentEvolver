import type { MediaAsset } from "@promedia/shared";

export type { MediaAsset };

export type PutMeta = {
  mime: string;
  originalName?: string;
  size?: number;
};

export type PutResult = {
  storageKey: string;
};

export type StorageAdapter = {
  readonly name: string;
  put(
    sessionId: string,
    assetId: string,
    bytes: Uint8Array | Buffer,
    meta: PutMeta,
  ): Promise<PutResult>;
  open(storageKey: string): Promise<{ path: string; mimeHint?: string }>;
  resolvePath(storageKey: string): string;
  delete?(storageKey: string): Promise<void>;
};

export const ALLOWED_IMAGE_MIMES = [
  "image/jpeg",
  "image/png",
  "image/webp",
  "image/gif",
  "image/svg+xml",
] as const;

export type AllowedImageMime = (typeof ALLOWED_IMAGE_MIMES)[number];

export function isAllowedImageMime(mime: string): mime is AllowedImageMime {
  return (ALLOWED_IMAGE_MIMES as readonly string[]).includes(mime);
}

/** Q1: mp4/webm allowed; quicktime optional with UI warning. */
export const ALLOWED_VIDEO_MIMES = [
  "video/mp4",
  "video/webm",
  "video/quicktime",
] as const;

export type AllowedVideoMime = (typeof ALLOWED_VIDEO_MIMES)[number];

export function isAllowedVideoMime(mime: string): mime is AllowedVideoMime {
  return (ALLOWED_VIDEO_MIMES as readonly string[]).includes(mime);
}

/** Common music/audio uploads (Guion Audio tab + Biblioteca). */
export const ALLOWED_AUDIO_MIMES = [
  "audio/mpeg",
  "audio/mp3",
  "audio/mp4",
  "audio/x-m4a",
  "audio/aac",
  "audio/wav",
  "audio/wave",
  "audio/x-wav",
  "audio/flac",
  "audio/x-flac",
  "audio/ogg",
  "audio/opus",
  "audio/webm",
] as const;

export type AllowedAudioMime = (typeof ALLOWED_AUDIO_MIMES)[number];

export function isAllowedAudioMime(mime: string): mime is AllowedAudioMime {
  return (ALLOWED_AUDIO_MIMES as readonly string[]).includes(mime);
}

export function isQuicktimeMime(mime: string): boolean {
  return mime === "video/quicktime";
}

/** Infer MIME from filename when the browser sends empty / octet-stream. */
export function mimeFromFileName(name: string): string | undefined {
  const lower = name.toLowerCase();
  if (lower.endsWith(".jpg") || lower.endsWith(".jpeg")) return "image/jpeg";
  if (lower.endsWith(".png")) return "image/png";
  if (lower.endsWith(".webp")) return "image/webp";
  if (lower.endsWith(".gif")) return "image/gif";
  if (lower.endsWith(".svg")) return "image/svg+xml";
  if (lower.endsWith(".mp4")) return "video/mp4";
  if (lower.endsWith(".webm")) return "video/webm";
  if (lower.endsWith(".mov")) return "video/quicktime";
  if (lower.endsWith(".mp3")) return "audio/mpeg";
  if (lower.endsWith(".m4a")) return "audio/mp4";
  if (lower.endsWith(".aac")) return "audio/aac";
  if (lower.endsWith(".wav")) return "audio/wav";
  if (lower.endsWith(".flac")) return "audio/flac";
  if (lower.endsWith(".ogg") || lower.endsWith(".oga")) return "audio/ogg";
  if (lower.endsWith(".opus")) return "audio/opus";
  return undefined;
}

export function resolveUploadMime(
  mime: string | undefined,
  originalName?: string,
): string {
  const raw = (mime || "").toLowerCase().trim();
  if (
    raw &&
    raw !== "application/octet-stream" &&
    raw !== "binary/octet-stream"
  ) {
    return raw;
  }
  return mimeFromFileName(originalName || "") || raw || "application/octet-stream";
}

export function extensionForMime(mime: string): string {
  switch (mime) {
    case "image/jpeg":
      return ".jpg";
    case "image/png":
      return ".png";
    case "image/webp":
      return ".webp";
    case "image/gif":
      return ".gif";
    case "image/svg+xml":
      return ".svg";
    case "video/mp4":
      return ".mp4";
    case "video/webm":
      return ".webm";
    case "video/quicktime":
      return ".mov";
    case "audio/mpeg":
    case "audio/mp3":
      return ".mp3";
    case "audio/mp4":
    case "audio/x-m4a":
      return ".m4a";
    case "audio/aac":
      return ".aac";
    case "audio/wav":
    case "audio/wave":
    case "audio/x-wav":
      return ".wav";
    case "audio/flac":
    case "audio/x-flac":
      return ".flac";
    case "audio/ogg":
      return ".ogg";
    case "audio/opus":
      return ".opus";
    case "audio/webm":
      return ".webm";
    default:
      return ".bin";
  }
}
