export type BibleVersionInfo = {
  id: string;
  label: string;
  shortLabel: string;
  language: "es" | "en" | string;
  license: string;
  default?: boolean;
  source?: "local" | "remote";
  note?: string;
  verseCount?: number;
  bookCount?: number;
};

export type BibleBookInfo = {
  id: string;
  nameEs: string;
  nameEn: string;
  chapters: number;
};

export type BiblePassageResponse = {
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
  verses: Array<{ verse: number; text: string }>;
  slides: Array<{ id: string; lines: string[] }>;
  error?: string;
  message?: string;
  detail?: string;
};

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(path);
  const data = (await res.json()) as T & {
    error?: string;
    message?: string;
    detail?: string;
  };
  if (!res.ok) {
    const err = new Error(
      data.message || data.error || `bible_http_${res.status}`,
    ) as Error & { code?: string; detail?: string };
    err.code = data.error;
    err.detail = data.detail;
    throw err;
  }
  return data;
}

export async function fetchBibleVersions(): Promise<{
  defaultVersionId: string;
  versions: BibleVersionInfo[];
  note?: string;
}> {
  return getJson("/api/v1/bible/versions");
}

export async function fetchBibleBooks(): Promise<{ books: BibleBookInfo[] }> {
  return getJson("/api/v1/bible/books");
}

export async function fetchBiblePassage(opts: {
  version: string;
  book?: string;
  chapter?: number;
  verseStart?: number;
  verseEnd?: number;
  q?: string;
}): Promise<BiblePassageResponse> {
  const params = new URLSearchParams();
  params.set("version", opts.version);
  if (opts.q?.trim()) {
    params.set("q", opts.q.trim());
  } else {
    if (opts.book) {
      params.set("book", opts.book);
    }
    if (opts.chapter != null) {
      params.set("chapter", String(opts.chapter));
    }
    if (opts.verseStart != null) {
      params.set("verseStart", String(opts.verseStart));
    }
    if (opts.verseEnd != null) {
      params.set("verseEnd", String(opts.verseEnd));
    }
  }
  return getJson(`/api/v1/bible/passage?${params.toString()}`);
}
