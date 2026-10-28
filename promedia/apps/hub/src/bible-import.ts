import { BIBLE_BOOKS, findBook } from "./bible/books.js";
import { parsePdfBuffer } from "./text-import.js";

export const BIBLE_IMPORT_MAX_BYTES = 40 * 1024 * 1024;
/** Reject empty/corrupt files; tiny fixtures (2+ verses) are OK for tests. */
export const BIBLE_IMPORT_MIN_VERSES = 2;

export type BibleCanonBooks = Record<
  string,
  Record<string, Record<string, string>>
>;

export type ImportedBibleCanon = {
  versionId: string;
  label: string;
  shortLabel: string;
  language: "es" | "en";
  copyright?: string;
  format: "bible-bcv" | "bible-numbered" | "pdf-nt";
  books: BibleCanonBooks;
  bookCount: number;
  verseCount: number;
  note?: string;
};

export type BibleImportResult =
  | { ok: true; canon: ImportedBibleCanon }
  | { ok: false; error: string; message: string };

type TranslationGuess = {
  versionId: string;
  label: string;
  shortLabel: string;
};

const NAME_RULES: Array<{
  test: RegExp;
  versionId: string;
  label: string;
  shortLabel: string;
}> = [
  {
    test: /las[_\s-]*americas|lbla|biblia_de_las_americas/i,
    versionId: "lbla",
    label: "La Biblia de las Américas",
    shortLabel: "LBLA",
  },
  {
    test: /nuevo[_\s-]*testamento[_\s-]*nvi|nvi[_\s-]*nt/i,
    versionId: "nvi-nt",
    label: "Nueva Versión Internacional (NT PDF)",
    shortLabel: "NVI-NT",
  },
  {
    test: /nueva[_\s-]*version[_\s-]*internacional|\bnvi\b/i,
    versionId: "nvi",
    label: "Nueva Versión Internacional",
    shortLabel: "NVI",
  },
  {
    test: /rv\s*1960|rvr\s*1960|reina[_\s-]*valera[_\s-]*1960/i,
    versionId: "rv1960",
    label: "Reina-Valera 1960",
    shortLabel: "RV1960",
  },
  {
    test: /spanish\s*ntv|nueva[_\s-]*traduccio[_]?n[_\s-]*viviente|nueva[_\s-]*traducci[oó]n[_\s-]*viviente|\bntv\b/i,
    versionId: "ntv",
    label: "Nueva Traducción Viviente",
    shortLabel: "NTV",
  },
  {
    test: /spanish\s*tla|traducci[oó]n[_\s-]*en[_\s-]*lenguaje[_\s-]*actual|\btla\b/i,
    versionId: "tla",
    label: "Traducción en Lenguaje Actual",
    shortLabel: "TLA",
  },
  {
    test: /latinoamericana|blh/i,
    versionId: "blh",
    label: "Biblia Latinoamericana de Hoy",
    shortLabel: "BLH",
  },
  {
    test: /nueva[_\s-]*biblia[_\s-]*de[_\s-]*los[_\s-]*hispanos|\bnbh\b/i,
    versionId: "nbh",
    label: "Nueva Biblia de los Hispanos",
    shortLabel: "NBH",
  },
  {
    test: /palabra[_\s-]*de[_\s-]*dios[_\s-]*para[_\s-]*tod|\bpdt\b/i,
    versionId: "pdt",
    label: "Palabra de Dios para Todos",
    shortLabel: "PDT",
  },
  {
    test: /lenguaje[_\s-]*sencillo|\bbls\b/i,
    versionId: "bls",
    label: "Biblia en Lenguaje Sencillo",
    shortLabel: "BLS",
  },
  {
    test: /dios[_\s-]*habla[_\s-]*hoy|\bdhh\b/i,
    versionId: "dhh",
    label: "Dios Habla Hoy",
    shortLabel: "DHH",
  },
];

function extOf(name: string): string {
  const i = name.lastIndexOf(".");
  return i >= 0 ? name.slice(i).toLowerCase() : "";
}

function decodeXmlEntities(raw: string): string {
  return raw
    .replace(/<!\[CDATA\[([\s\S]*?)\]\]>/g, "$1")
    .replace(/<[^>]+>/g, "")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&#(\d+);/g, (_, n) => String.fromCharCode(Number(n)))
    .replace(/&#x([0-9a-f]+);/gi, (_, h) =>
      String.fromCharCode(parseInt(h, 16)),
    )
    .replace(/\[[0-9]+\]/g, "")
    .replace(/\*/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function cleanVerseText(raw: string): string {
  const text = decodeXmlEntities(raw);
  if (!text) {
    return "";
  }
  if (/^\(TEXT\s+OMITTED\)$/i.test(text) || /^TEXT\s+OMITTED$/i.test(text)) {
    return "";
  }
  return text;
}

function decodeBibleBytes(bytes: Buffer): string {
  if (bytes.length >= 3 && bytes[0] === 0xef && bytes[1] === 0xbb && bytes[2] === 0xbf) {
    return bytes.toString("utf8");
  }
  const head = bytes.subarray(0, Math.min(bytes.length, 800)).toString("latin1");
  const encMatch = /encoding\s*=\s*["']?\s*(utf-8|utf8|iso-8859-1|latin1|windows-1252)/i.exec(
    head,
  );
  const enc = (encMatch?.[1] ?? "utf-8").toLowerCase();
  if (enc === "iso-8859-1" || enc === "latin1" || enc === "windows-1252") {
    const latin1 = bytes.toString("latin1");
    // Many Quelea exports declare latin1 but are UTF-8; prefer UTF-8 if valid Spanish.
    const utf8 = bytes.toString("utf8");
    const badUtf8 = (utf8.match(/\uFFFD/g) ?? []).length;
    const hasSpanishUtf8 = /[áéíóúñüÁÉÍÓÚÑÜ]/.test(utf8);
    if (badUtf8 === 0 && hasSpanishUtf8) {
      return utf8;
    }
    return latin1;
  }
  return bytes.toString("utf8");
}

export function guessTranslationFromName(originalName: string): TranslationGuess {
  const base = originalName.replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ");
  for (const rule of NAME_RULES) {
    if (rule.test.test(originalName) || rule.test.test(base)) {
      return {
        versionId: rule.versionId,
        label: rule.label,
        shortLabel: rule.shortLabel,
      };
    }
  }
  const slug = base
    .normalize("NFD")
    .replace(/\p{M}/gu, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 40) || "local";
  return {
    versionId: `local-${slug}`,
    label: base.trim() || "Biblia local",
    shortLabel: slug.slice(0, 12).toUpperCase() || "LOCAL",
  };
}

function countVerses(books: BibleCanonBooks): { bookCount: number; verseCount: number } {
  let bookCount = 0;
  let verseCount = 0;
  for (const chapters of Object.values(books)) {
    let has = false;
    for (const verses of Object.values(chapters)) {
      const n = Object.keys(verses).length;
      if (n > 0) {
        has = true;
        verseCount += n;
      }
    }
    if (has) {
      bookCount += 1;
    }
  }
  return { bookCount, verseCount };
}

function putVerse(
  books: BibleCanonBooks,
  bookId: string,
  chapter: number,
  verse: number,
  text: string,
): void {
  if (!text || chapter < 1 || verse < 1) {
    return;
  }
  const ch = String(chapter);
  const v = String(verse);
  if (!books[bookId]) {
    books[bookId] = {};
  }
  if (!books[bookId]![ch]) {
    books[bookId]![ch] = {};
  }
  books[bookId]![ch]![v] = text;
}

function parseFormatBcv(xml: string): BibleCanonBooks {
  const books: BibleCanonBooks = {};
  const bookRe = /<b\s+n="([^"]*)"\s*>([\s\S]*?)<\/b>/gi;
  let bm: RegExpExecArray | null;
  while ((bm = bookRe.exec(xml))) {
    const bookName = (bm[1] ?? "").trim();
    const book = findBook(bookName);
    if (!book) {
      continue;
    }
    const body = bm[2] ?? "";
    const chapRe = /<c\s+n="(\d+)"\s*>([\s\S]*?)<\/c>/gi;
    let cm: RegExpExecArray | null;
    while ((cm = chapRe.exec(body))) {
      const chapter = Number(cm[1]);
      const chapBody = cm[2] ?? "";
      const verseRe = /<v\s+n="(\d+)"\s*>([\s\S]*?)<\/v>/gi;
      let vm: RegExpExecArray | null;
      while ((vm = verseRe.exec(chapBody))) {
        const verse = Number(vm[1]);
        const text = cleanVerseText(vm[2] ?? "");
        putVerse(books, book.id, chapter, verse, text);
      }
    }
  }
  return books;
}

function parseFormatNumbered(xml: string): BibleCanonBooks {
  const books: BibleCanonBooks = {};
  const bookRe = /<book\s+number="(\d+)"\s*>([\s\S]*?)<\/book>/gi;
  let bm: RegExpExecArray | null;
  while ((bm = bookRe.exec(xml))) {
    const num = Number(bm[1]);
    const book = BIBLE_BOOKS[num - 1];
    if (!book || !Number.isFinite(num) || num < 1) {
      continue;
    }
    const body = bm[2] ?? "";
    const chapRe = /<chapter\s+number="(\d+)"\s*>([\s\S]*?)<\/chapter>/gi;
    let cm: RegExpExecArray | null;
    while ((cm = chapRe.exec(body))) {
      const chapter = Number(cm[1]);
      const chapBody = cm[2] ?? "";
      const verseRe = /<verse\s+number="(\d+)"\s*>([\s\S]*?)<\/verse>/gi;
      let vm: RegExpExecArray | null;
      while ((vm = verseRe.exec(chapBody))) {
        const verse = Number(vm[1]);
        const text = cleanVerseText(vm[2] ?? "");
        putVerse(books, book.id, chapter, verse, text);
      }
    }
  }
  return books;
}

function detectXmlFormat(xml: string): "bible-bcv" | "bible-numbered" | null {
  if (/<book\s+number=/i.test(xml) && /<verse\s+number=/i.test(xml)) {
    return "bible-numbered";
  }
  if (/<b\s+n=/i.test(xml) && /<v\s+n=/i.test(xml)) {
    return "bible-bcv";
  }
  return null;
}

function extractBibleAttr(xml: string, attr: string): string | undefined {
  const re = new RegExp(`<bible[^>]*\\b${attr}\\s*=\\s*"([^"]*)"`, "i");
  const m = re.exec(xml);
  const v = m?.[1]?.trim();
  return v || undefined;
}

/** Best-effort NT PDF: book headers + numbered verses. */
function parseNtPdfPages(pages: string[]): BibleCanonBooks {
  const books: BibleCanonBooks = {};
  const ntBooks = BIBLE_BOOKS.slice(39); // Matthew…Revelation
  let currentBook = ntBooks[0];
  let currentChapter = 1;

  const bookHeader = new RegExp(
    `^(?:${ntBooks
      .map((b) =>
        b.nameEs
          .normalize("NFD")
          .replace(/\p{M}/gu, "")
          .replace(/[.*+?^${}()|[\]\\]/g, "\\$&"),
      )
      .join("|")}|${ntBooks.map((b) => b.nameEn.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})(?:\\s+(\\d+))?\\s*$`,
    "i",
  );

  for (const page of pages) {
    const lines = page.split(/\n+/).map((l) => l.replace(/\s+/g, " ").trim()).filter(Boolean);
    for (const line of lines) {
      const norm = line
        .normalize("NFD")
        .replace(/\p{M}/gu, "")
        .toLowerCase();
      const header = bookHeader.exec(line);
      if (header) {
        const name = header[0].replace(/\s+\d+\s*$/, "").trim();
        const found = findBook(name);
        if (found && found.id !== "genesis") {
          // Prefer NT matches
          const nt = ntBooks.find((b) => b.id === found.id);
          if (nt) {
            currentBook = nt;
            currentChapter = header[1] ? Number(header[1]) : 1;
            continue;
          }
        }
      }
      // "Mateo 3" / "Juan 3:" style
      for (const b of ntBooks) {
        const es = b.nameEs
          .normalize("NFD")
          .replace(/\p{M}/gu, "")
          .toLowerCase();
        const en = b.nameEn.toLowerCase();
        const m = new RegExp(`^(?:${es}|${en})\\s+(\\d+)\\b`, "i").exec(norm);
        if (m) {
          currentBook = b;
          currentChapter = Number(m[1]);
          break;
        }
      }

      const verseLine = /^(\d{1,3})\s+(.+)$/.exec(line);
      if (verseLine && currentBook) {
        const verse = Number(verseLine[1]);
        const text = cleanVerseText(verseLine[2] ?? "");
        if (text.length >= 8) {
          putVerse(books, currentBook.id, currentChapter, verse, text);
        }
      }
    }
  }
  return books;
}

export function looksLikeBibleFile(originalName: string, bytes?: Buffer): boolean {
  const lower = originalName.toLowerCase();
  if (lower.endsWith(".xmm")) {
    return true;
  }
  if (lower.endsWith(".xml") || lower.endsWith(".pdf")) {
    if (
      /biblia|bible|nvi|ntv|tla|rv1960|rvr|lbla|dhh|pdt|blh|nbh|bls|testamento/i.test(
        originalName,
      )
    ) {
      return true;
    }
  }
  if (!bytes || bytes.length < 32) {
    return lower.endsWith(".xml");
  }
  const head = bytes.subarray(0, Math.min(bytes.length, 4000)).toString("utf8");
  return /<bible[\s>]/i.test(head) || /<b\s+n=/i.test(head) || /<book\s+number=/i.test(head);
}

export async function importBibleFile(opts: {
  bytes: Buffer;
  originalName: string;
  mime?: string;
}): Promise<BibleImportResult> {
  const originalName = opts.originalName || "biblia";
  if (opts.bytes.byteLength === 0) {
    return {
      ok: false,
      error: "empty_file",
      message: "El archivo está vacío.",
    };
  }
  if (opts.bytes.byteLength > BIBLE_IMPORT_MAX_BYTES) {
    return {
      ok: false,
      error: "too_large",
      message: `El archivo supera el límite de ${Math.round(BIBLE_IMPORT_MAX_BYTES / (1024 * 1024))} MB.`,
    };
  }

  const ext = extOf(originalName);
  const guess = guessTranslationFromName(originalName);

  if (ext === ".pdf" || opts.mime === "application/pdf") {
    try {
      const parsed = await parsePdfBuffer(opts.bytes, { maxPages: 400 });
      if (parsed.pages.length === 0) {
        return {
          ok: false,
          error: "pdf_empty",
          message:
            "No se pudo leer texto del PDF. Preferí un XML/XMM de Biblia completo.",
        };
      }
      // NT PDFs can exceed the generic text-import page cap; still use what we got.
      const books = parseNtPdfPages(parsed.pages);
      const { bookCount, verseCount } = countVerses(books);
      if (verseCount < BIBLE_IMPORT_MIN_VERSES) {
        return {
          ok: false,
          error: "pdf_unstructured",
          message:
            "No se reconocieron versículos en el PDF. Usá un archivo XML/XMM (formato Biblia) o subí el Nuevo Testamento en ese formato.",
        };
      }
      return {
        ok: true,
        canon: {
          versionId: guess.versionId,
          label: guess.label,
          shortLabel: guess.shortLabel,
          language: "es",
          copyright: "Texto importado localmente (PDF). Respetá la licencia del editor.",
          format: "pdf-nt",
          books,
          bookCount,
          verseCount,
          note: "Importación PDF aproximada (suele ser solo NT). Preferí XML/XMM si está disponible.",
        },
      };
    } catch (err) {
      return {
        ok: false,
        error: "pdf_parse_failed",
        message: `No se pudo analizar el PDF: ${err instanceof Error ? err.message : String(err)}`,
      };
    }
  }

  if (ext !== ".xml" && ext !== ".xmm") {
    return {
      ok: false,
      error: "unsupported_bible_type",
      message:
        "Formato no soportado para Biblias. Usá .xml, .xmm o un PDF de Nuevo Testamento.",
    };
  }

  const xml = decodeBibleBytes(opts.bytes);
  if (!/<bible[\s>]/i.test(xml) && !/<b\s+n=/i.test(xml) && !/<book\s+number=/i.test(xml)) {
    return {
      ok: false,
      error: "not_bible_xml",
      message:
        "El XML no parece una Biblia (faltan etiquetas <bible>/<b>/<v> o <book>/<verse>).",
    };
  }

  const format = detectXmlFormat(xml);
  if (!format) {
    return {
      ok: false,
      error: "unknown_bible_xml",
      message:
        "Estructura XML de Biblia no reconocida. Se admiten <b n>/<c n>/<v n> o <book number>/<verse number>.",
    };
  }

  const books =
    format === "bible-numbered" ? parseFormatNumbered(xml) : parseFormatBcv(xml);
  const { bookCount, verseCount } = countVerses(books);
  if (verseCount < BIBLE_IMPORT_MIN_VERSES) {
    return {
      ok: false,
      error: "too_few_verses",
      message: `El archivo casi no tiene versículos con texto (${verseCount}). Puede estar vacío o corrupto.`,
    };
  }

  const translationAttr = extractBibleAttr(xml, "translation");
  const statusAttr = extractBibleAttr(xml, "status");
  const copyright =
    statusAttr ||
    translationAttr ||
    "Texto importado localmente. No se redistribuye; respetá la licencia del editor.";

  const note =
    bookCount < 39
      ? `Solo ${bookCount} libro(s) con texto (p. ej. Nuevo Testamento).`
      : undefined;

  return {
    ok: true,
    canon: {
      versionId: guess.versionId,
      label: translationAttr?.slice(0, 120) || guess.label,
      shortLabel: guess.shortLabel,
      language: "es",
      copyright: copyright.slice(0, 500),
      format,
      books,
      bookCount,
      verseCount,
      note,
    },
  };
}
