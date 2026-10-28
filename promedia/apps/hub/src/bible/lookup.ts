import { randomBytes } from "node:crypto";
import { projectionTextSlides } from "@promedia/shared";
import { getBookById, type BibleBook } from "./books.js";
import { lookupFallback } from "./fallback.js";
import { lookupLocalVerses } from "./local-canon.js";
import { formatReferenceEs } from "./parse.js";
import {
  defaultVersionId,
  getVersion,
  type BibleVersion,
} from "./versions.js";
import { libraryStore } from "../library-store.js";

export type BibleVerseResult = {
  verse: number;
  text: string;
};

export type BiblePassageResult = {
  versionId: string;
  versionLabel: string;
  bookId: string;
  bookName: string;
  chapter: number;
  verseStart: number;
  verseEnd: number;
  reference: string;
  copyright?: string;
  fromCache?: boolean;
  verses: BibleVerseResult[];
};

type CacheEntry = {
  at: number;
  result: BiblePassageResult;
};

const CACHE_TTL_MS = 1000 * 60 * 60; // 1h
const cache = new Map<string, CacheEntry>();

function cacheKey(
  versionId: string,
  bookId: string,
  chapter: number,
  verseStart: number,
  verseEnd: number,
): string {
  return `${versionId}:${bookId}:${chapter}:${verseStart}-${verseEnd}`;
}

function newSlideId(): string {
  return randomBytes(8).toString("hex");
}

async function fetchJson(url: string, timeoutMs = 12_000): Promise<unknown> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      signal: ctrl.signal,
      headers: { Accept: "application/json" },
    });
    if (!res.ok) {
      throw new Error(`http_${res.status}`);
    }
    return (await res.json()) as unknown;
  } finally {
    clearTimeout(timer);
  }
}

async function fetchMidvash(
  version: BibleVersion,
  book: BibleBook,
  chapter: number,
  verseStart: number,
  verseEnd: number,
): Promise<BiblePassageResult> {
  const range =
    verseStart === verseEnd
      ? String(verseStart)
      : `${verseStart}-${verseEnd}`;
  const url = `https://api.midvash.com/v1/${encodeURIComponent(version.apiId)}/${encodeURIComponent(book.id)}/${chapter}/${range}`;
  const raw = (await fetchJson(url)) as {
    data?: { verses?: string[]; text?: string; verse?: number; verseEnd?: number };
    meta?: { copyright?: string };
  };
  const texts = raw.data?.verses;
  if (!Array.isArray(texts) || texts.length === 0) {
    throw new Error("empty_passage");
  }
  const verses: BibleVerseResult[] = texts.map((text, i) => ({
    verse: verseStart + i,
    text: String(text).trim().replace(/\s+/g, " "),
  }));
  return {
    versionId: version.id,
    versionLabel: version.shortLabel,
    bookId: book.id,
    bookName: book.nameEs,
    chapter,
    verseStart,
    verseEnd,
    reference: formatReferenceEs(book, chapter, verseStart, verseEnd),
    copyright: raw.meta?.copyright?.trim() || version.license,
    verses,
  };
}

async function fetchBibleApi(
  version: BibleVersion,
  book: BibleBook,
  chapter: number,
  verseStart: number,
  verseEnd: number,
): Promise<BiblePassageResult> {
  // bible-api.com: book name with + for spaces, then chapter:verse
  const pathRef =
    verseStart === verseEnd
      ? `${encodeURIComponent(book.nameEn)}+${chapter}:${verseStart}`
      : `${encodeURIComponent(book.nameEn)}+${chapter}:${verseStart}-${verseEnd}`;
  const finalUrl = `https://bible-api.com/${pathRef}?translation=${encodeURIComponent(version.apiId)}`;
  const raw = (await fetchJson(finalUrl)) as {
    verses?: Array<{ verse?: number; text?: string }>;
    translation_note?: string;
    error?: string;
  };
  if (!Array.isArray(raw.verses) || raw.verses.length === 0) {
    throw new Error(raw.error || "empty_passage");
  }
  const verses: BibleVerseResult[] = raw.verses.map((v) => ({
    verse: Number(v.verse) || 0,
    text: String(v.text ?? "")
      .trim()
      .replace(/\s+/g, " "),
  })).filter((v) => v.verse >= verseStart && v.verse <= verseEnd && v.text);
  if (verses.length === 0) {
    throw new Error("empty_passage");
  }
  return {
    versionId: version.id,
    versionLabel: version.shortLabel,
    bookId: book.id,
    bookName: book.nameEs,
    chapter,
    verseStart: verses[0]!.verse,
    verseEnd: verses[verses.length - 1]!.verse,
    reference: formatReferenceEs(
      book,
      chapter,
      verses[0]!.verse,
      verses[verses.length - 1]!.verse,
    ),
    copyright: raw.translation_note || version.license,
    verses,
  };
}

function fromFallback(
  version: BibleVersion,
  book: BibleBook,
  chapter: number,
  verseStart: number,
  verseEnd: number,
): BiblePassageResult | null {
  const hit = lookupFallback({
    versionId: version.id,
    bookId: book.id,
    chapter,
    verseStart,
    verseEnd,
  });
  if (!hit) {
    return null;
  }
  return {
    versionId: version.id,
    versionLabel: version.shortLabel,
    bookId: book.id,
    bookName: book.nameEs,
    chapter,
    verseStart: hit.verseStart,
    verseEnd: hit.verseEnd,
    reference: formatReferenceEs(book, chapter, hit.verseStart, hit.verseEnd),
    copyright: hit.copyright,
    fromCache: true,
    verses: hit.verses,
  };
}

async function resolveVersion(
  versionId: string | undefined,
): Promise<BibleVersion | undefined> {
  const id = (versionId ?? defaultVersionId()).trim().toLowerCase();
  const local = await libraryStore.getLocalBibleVersion(id);
  if (local) {
    return {
      id: local.versionId,
      label: local.label,
      shortLabel: local.shortLabel,
      language: local.language,
      provider: "local",
      apiId: local.versionId,
      license: local.license,
      source: "local",
      libraryItemId: local.libraryItemId,
      note: local.note,
    };
  }
  return getVersion(id);
}

export async function lookupPassage(opts: {
  versionId?: string;
  bookId: string;
  chapter: number;
  verseStart: number;
  verseEnd: number;
}): Promise<
  | { ok: true; result: BiblePassageResult }
  | { ok: false; error: string; detail?: string; message?: string }
> {
  const version = await resolveVersion(opts.versionId);
  if (!version) {
    return { ok: false, error: "unknown_version" };
  }
  const book = getBookById(opts.bookId);
  if (!book) {
    return { ok: false, error: "unknown_book" };
  }
  const chapter = Math.trunc(opts.chapter);
  const verseStart = Math.trunc(opts.verseStart);
  let verseEnd = Math.trunc(opts.verseEnd);
  if (chapter < 1 || chapter > book.chapters) {
    return { ok: false, error: "invalid_chapter" };
  }
  if (verseStart < 1 || verseEnd < verseStart) {
    return { ok: false, error: "invalid_verse_range" };
  }
  // Cap range size for MVP (one chapter slice)
  if (verseEnd - verseStart > 49) {
    verseEnd = verseStart + 49;
  }

  const key = cacheKey(version.id, book.id, chapter, verseStart, verseEnd);
  const cached = cache.get(key);
  if (cached && Date.now() - cached.at < CACHE_TTL_MS) {
    return { ok: true, result: cached.result };
  }

  if (version.provider === "local") {
    const local = await libraryStore.getLocalBibleVersion(version.id);
    if (!local) {
      return {
        ok: false,
        error: "local_missing",
        message: "La versión local ya no está en la biblioteca.",
      };
    }
    const verses = await lookupLocalVerses({
      canonAbsolutePath: local.canonAbsolutePath,
      versionId: version.id,
      bookId: book.id,
      chapter,
      verseStart,
      verseEnd,
    });
    if (!verses) {
      return {
        ok: false,
        error: "passage_not_found",
        message:
          "Ese pasaje no está en la Biblia local (libro/capítulo incompleto o vacío).",
      };
    }
    const result: BiblePassageResult = {
      versionId: version.id,
      versionLabel: version.shortLabel,
      bookId: book.id,
      bookName: book.nameEs,
      chapter,
      verseStart: verses[0]!.verse,
      verseEnd: verses[verses.length - 1]!.verse,
      reference: formatReferenceEs(
        book,
        chapter,
        verses[0]!.verse,
        verses[verses.length - 1]!.verse,
      ),
      copyright: version.license,
      verses,
    };
    cache.set(key, { at: Date.now(), result });
    return { ok: true, result };
  }

  try {
    const result =
      version.provider === "midvash"
        ? await fetchMidvash(version, book, chapter, verseStart, verseEnd)
        : await fetchBibleApi(version, book, chapter, verseStart, verseEnd);
    cache.set(key, { at: Date.now(), result });
    return { ok: true, result };
  } catch (err) {
    const fallback = fromFallback(version, book, chapter, verseStart, verseEnd);
    if (fallback) {
      return { ok: true, result: fallback };
    }
    return {
      ok: false,
      error: "api_failed",
      detail: err instanceof Error ? err.message : String(err),
    };
  }
}

/** Build playlist-ready slides (one verse per slide; multi-line body → 5-line chunks). */
export function passageToSlides(result: BiblePassageResult): Array<{
  id: string;
  lines: string[];
}> {
  const raw = result.verses.map((v) => {
    const bodyLines = String(v.text ?? "")
      .replace(/\r\n/g, "\n")
      .split("\n")
      .map((line) => line.trimEnd());
    return {
      id: newSlideId(),
      lines: [
        `${result.bookName} ${result.chapter}:${v.verse}`,
        ...(bodyLines.length > 0 ? bodyLines : [""]),
      ],
    };
  });
  return projectionTextSlides(raw);
}
