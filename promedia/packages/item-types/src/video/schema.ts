import type {
  VideoFilePayload,
  VideoFit,
  VideoPayload,
  VideoPlaylistItem,
  VideoYoutubePayload,
} from "@promedia/shared";
import { isYouTubeVideoId, parseYouTubeUrl } from "./youtube.js";

const FITS: ReadonlySet<VideoFit> = new Set(["contain", "cover"]);

function isBoolOrUndef(v: unknown): boolean {
  return v === undefined || typeof v === "boolean";
}

function isCommonFields(p: Record<string, unknown>): boolean {
  if (p.title !== undefined && typeof p.title !== "string") {
    return false;
  }
  if (!isBoolOrUndef(p.autoplay) || !isBoolOrUndef(p.loop) || !isBoolOrUndef(p.muted)) {
    return false;
  }
  if (p.fit !== undefined && !FITS.has(p.fit as VideoFit)) {
    return false;
  }
  return true;
}

export function isVideoFilePayload(value: unknown): value is VideoFilePayload {
  if (!value || typeof value !== "object") {
    return false;
  }
  const p = value as Record<string, unknown>;
  return (
    p.sourceKind === "file" &&
    typeof p.assetId === "string" &&
    p.assetId.length > 0 &&
    isCommonFields(p)
  );
}

export function isVideoYoutubePayload(value: unknown): value is VideoYoutubePayload {
  if (!value || typeof value !== "object") {
    return false;
  }
  const p = value as Record<string, unknown>;
  return (
    p.sourceKind === "youtube" &&
    typeof p.videoId === "string" &&
    isYouTubeVideoId(p.videoId) &&
    typeof p.sourceUrl === "string" &&
    p.sourceUrl.length > 0 &&
    isCommonFields(p)
  );
}

export function isVideoPayload(value: unknown): value is VideoPayload {
  return isVideoFilePayload(value) || isVideoYoutubePayload(value);
}

export function isVideoPlaylistItem(value: unknown): value is VideoPlaylistItem {
  if (!value || typeof value !== "object") {
    return false;
  }
  const item = value as Record<string, unknown>;
  return (
    typeof item.id === "string" &&
    item.id.length > 0 &&
    item.type === "video" &&
    typeof item.order === "number" &&
    Number.isFinite(item.order) &&
    isVideoPayload(item.payload)
  );
}

export function normalizeVideoPayload(payload: VideoPayload): VideoPayload {
  const common = {
    title: payload.title?.trim() ? payload.title.trim() : undefined,
    autoplay: payload.autoplay !== false,
    loop: payload.loop === true,
    muted: payload.muted === true,
    fit: payload.fit ?? "contain",
  };
  if (payload.sourceKind === "file") {
    return {
      sourceKind: "file",
      assetId: payload.assetId,
      ...common,
    };
  }
  return {
    sourceKind: "youtube",
    videoId: payload.videoId,
    sourceUrl: payload.sourceUrl,
    ...common,
  };
}

export { parseYouTubeUrl, isYouTubeVideoId };
