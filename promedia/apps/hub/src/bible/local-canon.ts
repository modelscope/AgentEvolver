import { readFile } from "node:fs/promises";
import type { LibraryBibleVersionMeta, LibraryItem } from "@promedia/shared";
import type { BibleCanonBooks, ImportedBibleCanon } from "../bible-import.js";

export type LocalBibleVersion = {
  versionId: string;
  label: string;
  shortLabel: string;
  language: string;
  license: string;
  source: "local";
  libraryItemId: string;
  bookCount: number;
  verseCount: number;
  note?: string;
  canonAbsolutePath: string;
};

type CanonFile = {
  versionId: string;
  label: string;
  shortLabel: string;
  language?: string;
  copyright?: string;
  books: BibleCanonBooks;
};

const canonCache = new Map<string, CanonFile>();

export function clearLocalCanonCache(versionId?: string): void {
  if (versionId) {
    canonCache.delete(versionId.toLowerCase());
    return;
  }
  canonCache.clear();
}

export function libraryItemToLocalVersion(
  item: LibraryItem,
  resolvePath: (rel: string) => string,
): LocalBibleVersion | null {
  const meta = item.bibleVersion;
  if (!meta?.versionId || !meta.canonRelativePath) {
    return null;
  }
  return {
    versionId: meta.versionId,
    label: meta.label,
    shortLabel: meta.shortLabel,
    language: meta.language || "es",
    license: meta.copyright || "Importación local",
    source: "local",
    libraryItemId: item.id,
    bookCount: meta.bookCount,
    verseCount: meta.verseCount,
    note: meta.note,
    canonAbsolutePath: resolvePath(meta.canonRelativePath),
  };
}

export async function loadCanonFile(absPath: string): Promise<CanonFile | null> {
  const cached = [...canonCache.entries()].find(
    ([, v]) => (v as CanonFile & { __path?: string }).__path === absPath,
  );
  if (cached) {
    return cached[1];
  }
  try {
    const raw = await readFile(absPath, "utf8");
    const parsed = JSON.parse(raw) as CanonFile;
    if (!parsed?.versionId || !parsed.books) {
      return null;
    }
    canonCache.set(parsed.versionId.toLowerCase(), parsed);
    return parsed;
  } catch {
    return null;
  }
}

export async function lookupLocalVerses(opts: {
  canonAbsolutePath: string;
  versionId: string;
  bookId: string;
  chapter: number;
  verseStart: number;
  verseEnd: number;
}): Promise<Array<{ verse: number; text: string }> | null> {
  let canon = canonCache.get(opts.versionId.toLowerCase());
  if (!canon) {
    canon = (await loadCanonFile(opts.canonAbsolutePath)) ?? undefined;
  }
  if (!canon) {
    // try load by path even if version id differs
    canon = (await loadCanonFile(opts.canonAbsolutePath)) ?? undefined;
  }
  if (!canon) {
    return null;
  }
  const chapter = canon.books[opts.bookId]?.[String(opts.chapter)];
  if (!chapter) {
    return null;
  }
  const verses: Array<{ verse: number; text: string }> = [];
  for (let v = opts.verseStart; v <= opts.verseEnd; v += 1) {
    const text = chapter[String(v)];
    if (typeof text === "string" && text.trim()) {
      verses.push({ verse: v, text: text.trim() });
    }
  }
  return verses.length > 0 ? verses : null;
}

export function canonToDiskJson(canon: ImportedBibleCanon): string {
  return JSON.stringify(
    {
      versionId: canon.versionId,
      label: canon.label,
      shortLabel: canon.shortLabel,
      language: canon.language,
      copyright: canon.copyright,
      format: canon.format,
      bookCount: canon.bookCount,
      verseCount: canon.verseCount,
      note: canon.note,
      books: canon.books,
    },
    null,
    0,
  );
}

export function metaFromCanon(
  canon: ImportedBibleCanon,
  canonRelativePath: string,
): LibraryBibleVersionMeta {
  return {
    versionId: canon.versionId,
    label: canon.label,
    shortLabel: canon.shortLabel,
    language: canon.language,
    copyright: canon.copyright,
    bookCount: canon.bookCount,
    verseCount: canon.verseCount,
    format: canon.format,
    canonRelativePath,
    note: canon.note,
  };
}
