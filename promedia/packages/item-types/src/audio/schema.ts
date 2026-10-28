import type { AudioPayload, AudioPlaylistItem } from "@promedia/shared";

export function isAudioPayload(value: unknown): value is AudioPayload {
  if (!value || typeof value !== "object") {
    return false;
  }
  const p = value as Record<string, unknown>;
  if (typeof p.assetId !== "string" || p.assetId.length === 0) {
    return false;
  }
  if (p.title !== undefined && typeof p.title !== "string") {
    return false;
  }
  if (p.autoplay !== undefined && typeof p.autoplay !== "boolean") {
    return false;
  }
  if (p.loop !== undefined && typeof p.loop !== "boolean") {
    return false;
  }
  return true;
}

export function isAudioPlaylistItem(value: unknown): value is AudioPlaylistItem {
  if (!value || typeof value !== "object") {
    return false;
  }
  const item = value as Record<string, unknown>;
  return (
    typeof item.id === "string" &&
    item.id.length > 0 &&
    item.type === "audio" &&
    typeof item.order === "number" &&
    Number.isFinite(item.order) &&
    isAudioPayload(item.payload)
  );
}

export function normalizeAudioPayload(payload: AudioPayload): AudioPayload {
  return {
    assetId: payload.assetId,
    title: payload.title?.trim() ? payload.title.trim() : undefined,
    autoplay: payload.autoplay !== false,
    loop: payload.loop === true,
  };
}
