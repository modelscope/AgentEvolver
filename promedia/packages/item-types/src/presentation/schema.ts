import type {
  ImageFit,
  PresentationPayload,
  PresentationPlaylistItem,
  PresentationSlide,
} from "@promedia/shared";

const FITS: ReadonlySet<ImageFit> = new Set(["contain", "cover", "fill"]);

export function isPresentationSlide(
  value: unknown,
): value is PresentationSlide {
  if (!value || typeof value !== "object") {
    return false;
  }
  const s = value as Record<string, unknown>;
  return (
    typeof s.id === "string" &&
    s.id.length > 0 &&
    typeof s.assetId === "string" &&
    s.assetId.length > 0
  );
}

export function isPresentationPayload(
  value: unknown,
): value is PresentationPayload {
  if (!value || typeof value !== "object") {
    return false;
  }
  const p = value as Record<string, unknown>;
  if (!Array.isArray(p.slides) || p.slides.length < 1) {
    return false;
  }
  if (!p.slides.every(isPresentationSlide)) {
    return false;
  }
  if (typeof p.fit !== "string" || !FITS.has(p.fit as ImageFit)) {
    return false;
  }
  if (p.title !== undefined && typeof p.title !== "string") {
    return false;
  }
  if (p.originalName !== undefined && typeof p.originalName !== "string") {
    return false;
  }
  if (p.libraryItemId !== undefined && typeof p.libraryItemId !== "string") {
    return false;
  }
  return true;
}

export function isPresentationPlaylistItem(
  value: unknown,
): value is PresentationPlaylistItem {
  if (!value || typeof value !== "object") {
    return false;
  }
  const item = value as Record<string, unknown>;
  return (
    typeof item.id === "string" &&
    item.id.length > 0 &&
    item.type === "presentation" &&
    typeof item.order === "number" &&
    Number.isFinite(item.order) &&
    isPresentationPayload(item.payload)
  );
}

export function normalizePresentationPayload(
  payload: PresentationPayload,
): PresentationPayload {
  return {
    title: payload.title?.trim() ? payload.title.trim() : undefined,
    fit: payload.fit ?? "contain",
    slides: payload.slides.map((s) => ({
      id: s.id,
      assetId: s.assetId,
    })),
    originalName: payload.originalName?.trim()
      ? payload.originalName.trim()
      : undefined,
    libraryItemId: payload.libraryItemId?.trim()
      ? payload.libraryItemId.trim()
      : undefined,
  };
}
