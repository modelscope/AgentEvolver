/**
 * Tiny offline subset used when remote APIs fail.
 * Not a full canon — clear errors remain the primary failure path.
 */

export type FallbackVerse = {
  verse: number;
  text: string;
};

export type FallbackPassage = {
  versionId: string;
  bookId: string;
  chapter: number;
  verseStart: number;
  verseEnd: number;
  copyright: string;
  verses: FallbackVerse[];
};

const FALLBACK: FallbackPassage[] = [
  {
    versionId: "rvr1909",
    bookId: "john",
    chapter: 3,
    verseStart: 16,
    verseEnd: 17,
    copyright: "Reina-Valera 1909 — dominio público (subconjunto local ProMedia).",
    verses: [
      {
        verse: 16,
        text: "Porque de tal manera amó Dios al mundo, que ha dado á su Hijo unigénito, para que todo aquel que en él cree, no se pierda, mas tenga vida eterna.",
      },
      {
        verse: 17,
        text: "Porque no envió Dios á su Hijo al mundo para que condene al mundo, mas para que el mundo sea salvo por él.",
      },
    ],
  },
  {
    versionId: "kjv",
    bookId: "john",
    chapter: 3,
    verseStart: 16,
    verseEnd: 17,
    copyright: "King James Version — public domain (ProMedia local subset).",
    verses: [
      {
        verse: 16,
        text: "For God so loved the world, that he gave his only begotten Son, that whosoever believeth in him should not perish, but have everlasting life.",
      },
      {
        verse: 17,
        text: "For God sent not his Son into the world to condemn the world; but that the world through him might be saved.",
      },
    ],
  },
];

export function lookupFallback(opts: {
  versionId: string;
  bookId: string;
  chapter: number;
  verseStart: number;
  verseEnd: number;
}): FallbackPassage | null {
  for (const entry of FALLBACK) {
    if (
      entry.versionId !== opts.versionId ||
      entry.bookId !== opts.bookId ||
      entry.chapter !== opts.chapter
    ) {
      continue;
    }
    if (
      opts.verseStart >= entry.verseStart &&
      opts.verseEnd <= entry.verseEnd
    ) {
      const verses = entry.verses.filter(
        (v) => v.verse >= opts.verseStart && v.verse <= opts.verseEnd,
      );
      if (verses.length === 0) {
        return null;
      }
      return {
        ...entry,
        verseStart: opts.verseStart,
        verseEnd: opts.verseEnd,
        verses,
      };
    }
  }
  return null;
}
