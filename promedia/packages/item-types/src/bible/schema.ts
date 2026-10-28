import type { BiblePayload, BiblePlaylistItem, TextAlign } from "@promedia/shared";
import { projectionTextSlides } from "@promedia/shared";
import { isTextSlide } from "../text/schema.js";

const ALIGNS: ReadonlySet<TextAlign> = new Set(["left", "center", "right"]);

export function isBiblePayload(value: unknown): value is BiblePayload {
  if (!value || typeof value !== "object") {
    return false;
  }
  const p = value as Record<string, unknown>;
  if (typeof p.versionId !== "string" || !p.versionId.trim()) {
    return false;
  }
  if (typeof p.versionLabel !== "string" || !p.versionLabel.trim()) {
    return false;
  }
  if (typeof p.bookId !== "string" || !p.bookId.trim()) {
    return false;
  }
  if (typeof p.bookName !== "string" || !p.bookName.trim()) {
    return false;
  }
  if (typeof p.chapter !== "number" || !Number.isFinite(p.chapter) || p.chapter < 1) {
    return false;
  }
  if (
    typeof p.verseStart !== "number" ||
    !Number.isFinite(p.verseStart) ||
    p.verseStart < 1
  ) {
    return false;
  }
  if (
    typeof p.verseEnd !== "number" ||
    !Number.isFinite(p.verseEnd) ||
    p.verseEnd < p.verseStart
  ) {
    return false;
  }
  if (typeof p.reference !== "string" || !p.reference.trim()) {
    return false;
  }
  if (!Array.isArray(p.slides) || p.slides.length < 1) {
    return false;
  }
  if (!p.slides.every(isTextSlide)) {
    return false;
  }
  if (p.copyright !== undefined && typeof p.copyright !== "string") {
    return false;
  }
  if (p.fromCache !== undefined && typeof p.fromCache !== "boolean") {
    return false;
  }
  if (p.align !== undefined && !ALIGNS.has(p.align as TextAlign)) {
    return false;
  }
  return true;
}

export function isBiblePlaylistItem(value: unknown): value is BiblePlaylistItem {
  if (!value || typeof value !== "object") {
    return false;
  }
  const item = value as Record<string, unknown>;
  return (
    typeof item.id === "string" &&
    item.id.length > 0 &&
    item.type === "bible" &&
    typeof item.order === "number" &&
    Number.isFinite(item.order) &&
    isBiblePayload(item.payload)
  );
}

export function normalizeBiblePayload(payload: BiblePayload): BiblePayload {
  return {
    versionId: payload.versionId.trim(),
    versionLabel: payload.versionLabel.trim(),
    bookId: payload.bookId.trim().toLowerCase(),
    bookName: payload.bookName.trim(),
    chapter: Math.trunc(payload.chapter),
    verseStart: Math.trunc(payload.verseStart),
    verseEnd: Math.trunc(payload.verseEnd),
    reference: payload.reference.trim(),
    copyright: payload.copyright?.trim() || undefined,
    fromCache: payload.fromCache === true ? true : undefined,
    align: payload.align ?? "center",
    slides: projectionTextSlides(
      payload.slides.map((slide) => ({
        id: slide.id,
        lines: slide.lines.map((line) => String(line)),
      })),
    ),
  };
}
