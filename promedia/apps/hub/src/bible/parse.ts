import { findBook, type BibleBook } from "./books.js";

export type ParsedBibleRef = {
  book: BibleBook;
  chapter: number;
  verseStart: number;
  verseEnd: number;
};

/**
 * Parse "Juan 3:16-17", "Jn 3:16", "1 Corintios 13:4", "John 3".
 * Chapter-only → verseStart=1, verseEnd=1 (caller may expand); we require a verse for MVP add.
 */
export function parseBibleReference(raw: string): ParsedBibleRef | null {
  const cleaned = raw.trim().replace(/\s+/g, " ");
  if (!cleaned) {
    return null;
  }

  // "Book Chapter:Start-End" or "Book Chapter:Start" or "Book Chapter"
  const m =
    /^(.+?)\s+(\d+)\s*(?::\s*(\d+)(?:\s*[-–—]\s*(\d+))?)?$/u.exec(cleaned);
  if (!m) {
    return null;
  }
  const book = findBook(m[1]!);
  if (!book) {
    return null;
  }
  const chapter = Number(m[2]);
  if (!Number.isFinite(chapter) || chapter < 1 || chapter > book.chapters) {
    return null;
  }
  const verseStart = m[3] ? Number(m[3]) : 1;
  const verseEnd = m[4] ? Number(m[4]) : verseStart;
  if (
    !Number.isFinite(verseStart) ||
    !Number.isFinite(verseEnd) ||
    verseStart < 1 ||
    verseEnd < verseStart
  ) {
    return null;
  }
  return { book, chapter, verseStart, verseEnd };
}

export function formatReferenceEs(
  book: BibleBook,
  chapter: number,
  verseStart: number,
  verseEnd: number,
): string {
  if (verseStart === verseEnd) {
    return `${book.nameEs} ${chapter}:${verseStart}`;
  }
  return `${book.nameEs} ${chapter}:${verseStart}-${verseEnd}`;
}
