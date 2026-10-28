import type { ImageFit, ImagePayload, ImagePlaylistItem } from "@promedia/shared";

const FITS: ReadonlySet<ImageFit> = new Set(["contain", "cover", "fill"]);

export function isImagePayload(value: unknown): value is ImagePayload {
  if (!value || typeof value !== "object") {
    return false;
  }
  const p = value as Record<string, unknown>;
  if (typeof p.assetId !== "string" || p.assetId.length === 0) {
    return false;
  }
  if (typeof p.fit !== "string" || !FITS.has(p.fit as ImageFit)) {
    return false;
  }
  if (p.title !== undefined && typeof p.title !== "string") {
    return false;
  }
  return true;
}

export function isImagePlaylistItem(value: unknown): value is ImagePlaylistItem {
  if (!value || typeof value !== "object") {
    return false;
  }
  const item = value as Record<string, unknown>;
  return (
    typeof item.id === "string" &&
    item.id.length > 0 &&
    item.type === "image" &&
    typeof item.order === "number" &&
    Number.isFinite(item.order) &&
    isImagePayload(item.payload)
  );
}

export function normalizeImagePayload(payload: ImagePayload): ImagePayload {
  return {
    assetId: payload.assetId,
    fit: payload.fit ?? "contain",
    title: payload.title?.trim() ? payload.title.trim() : undefined,
  };
}
