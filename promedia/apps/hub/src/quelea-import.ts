import { randomBytes } from "node:crypto";
import JSZip from "jszip";
import type { LibrarySongPayload, LibrarySongStanza } from "@promedia/shared";

export const QUELEA_IMPORT_MAX_BYTES = 80 * 1024 * 1024;
export const QUELEA_IMPORT_MAX_SONGS = 5000;

export type ParsedQueleaSong = {
  song: LibrarySongPayload;
  sourceName?: string;
};

export type QueleaSongParseResult =
  | { ok: true; song: LibrarySongPayload }
  | { ok: false; error: string; message: string };

export type QueleaPackParseResult =
  | {
      ok: true;
      songs: ParsedQueleaSong[];
      skipped: Array<{ file: string; error: string; message: string }>;
    }
  | { ok: false; error: string; message: string };

function newStanzaId(): string {
  return randomBytes(6).toString("hex");
}

function decodeXmlEntities(raw: string): string {
  return raw
    .replace(/<!\[CDATA\[([\s\S]*?)\]\]>/g, "$1")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&#(\d+);/g, (_, n) => String.fromCodePoint(Number(n)))
    .replace(/&#x([0-9a-fA-F]+);/g, (_, h) =>
      String.fromCodePoint(Number.parseInt(h, 16)),
    );
}

function stripTags(raw: string): string {
  return decodeXmlEntities(raw.replace(/<[^>]+>/g, ""));
}

function firstTagText(xml: string, tag: string): string {
  const re = new RegExp(`<${tag}\\b[^>]*>([\\s\\S]*?)</${tag}>`, "i");
  const m = xml.match(re);
  return m ? stripTags(m[1]).trim() : "";
}

/**
 * Quelea stanza body: each non-empty newline is one projected row; blank lines
 * and a lone "." (Quelea blank placeholder) are skipped. Multi-line blobs are
 * already split here so song→text uses the same 5-line projection path.
 */
export function parseQueleaLyricBody(raw: string): string[] {
  const text = decodeXmlEntities(raw.replace(/\r\n/g, "\n").replace(/\r/g, "\n"));
  const lines: string[] = [];
  for (const part of text.split("\n")) {
    const trimmed = part.trim();
    if (trimmed === "" || trimmed === ".") {
      continue;
    }
    lines.push(trimmed);
  }
  return lines;
}

function sectionInnerLyrics(sectionXml: string): string | null {
  // Prefer the last <lyrics>…</lyrics> inside the section (outer pack wraps
  // sections; each section has its own inner lyrics with verse text).
  const matches = [
    ...sectionXml.matchAll(/<lyrics\b[^>]*>([\s\S]*?)<\/lyrics>/gi),
  ];
  if (matches.length === 0) {
    return null;
  }
  return matches[matches.length - 1]![1] ?? "";
}

export function looksLikeQueleaSongXml(
  originalName: string,
  bytes?: Buffer,
): boolean {
  const lower = originalName.toLowerCase();
  if (lower.endsWith(".qsp")) {
    return true;
  }
  if (!lower.endsWith(".xml")) {
    return false;
  }
  if (!bytes || bytes.length < 16) {
    return /song|quelea|cancion/i.test(originalName);
  }
  const head = bytes.subarray(0, Math.min(bytes.length, 8000)).toString("utf8");
  return /<song[\s>]/i.test(head) && /<lyrics[\s>]/i.test(head);
}

export function parseQueleaSongXml(
  xml: string,
  opts?: { fallbackTitle?: string },
): QueleaSongParseResult {
  const trimmed = xml.trim();
  if (!trimmed) {
    return {
      ok: false,
      error: "empty_file",
      message: "El archivo de canción está vacío.",
    };
  }
  if (!/<song[\s>]/i.test(trimmed)) {
    return {
      ok: false,
      error: "not_quelea_song",
      message: "No parece una canción Quelea (<song>…).",
    };
  }

  const title =
    firstTagText(trimmed, "title") ||
    opts?.fallbackTitle?.trim() ||
    "";
  if (!title) {
    return {
      ok: false,
      error: "missing_title",
      message: "La canción Quelea no tiene título.",
    };
  }

  const author = firstTagText(trimmed, "author");
  const sections = [...trimmed.matchAll(/<section\b[^>]*>([\s\S]*?)<\/section>/gi)];
  const stanzas: LibrarySongStanza[] = [];

  for (const match of sections) {
    const body = sectionInnerLyrics(match[1] ?? "");
    if (body == null) {
      continue;
    }
    const lines = parseQueleaLyricBody(body);
    if (lines.length === 0) {
      continue;
    }
    stanzas.push({ id: newStanzaId(), lines });
  }

  // Fallback: some minimal songs may only have a flat lyrics block.
  if (stanzas.length === 0) {
    const flat = trimmed.match(/<lyrics\b[^>]*>([\s\S]*?)<\/lyrics>/i);
    if (flat) {
      const inner = flat[1] ?? "";
      if (!/<section\b/i.test(inner)) {
        const lines = parseQueleaLyricBody(inner);
        if (lines.length > 0) {
          stanzas.push({ id: newStanzaId(), lines });
        }
      }
    }
  }

  if (stanzas.length === 0) {
    return {
      ok: false,
      error: "no_lyrics",
      message: `«${title}» no tiene estrofas con letra.`,
    };
  }

  const song: LibrarySongPayload = {
    title,
    artist: author || undefined,
    stanzas,
  };
  return { ok: true, song };
}

export async function parseQueleaSongFile(opts: {
  bytes: Buffer;
  originalName: string;
}): Promise<QueleaSongParseResult> {
  const originalName = opts.originalName || "song.xml";
  if (opts.bytes.byteLength === 0) {
    return {
      ok: false,
      error: "empty_file",
      message: "El archivo está vacío.",
    };
  }
  if (opts.bytes.byteLength > QUELEA_IMPORT_MAX_BYTES) {
    return {
      ok: false,
      error: "too_large",
      message: `Archivo demasiado grande (máx. ${Math.round(QUELEA_IMPORT_MAX_BYTES / (1024 * 1024))} MB).`,
    };
  }
  const lower = originalName.toLowerCase();
  if (!lower.endsWith(".xml") && !lower.endsWith(".qsp")) {
    return {
      ok: false,
      error: "unsupported_type",
      message: "Usá un pack Quelea (.qsp) o un XML de canción Quelea.",
    };
  }
  if (lower.endsWith(".qsp")) {
    return {
      ok: false,
      error: "use_pack_parser",
      message: "Usá el importador de packs .qsp para este archivo.",
    };
  }
  const xml = opts.bytes.toString("utf8");
  const fallbackTitle = originalName.replace(/\.xml$/i, "").trim();
  return parseQueleaSongXml(xml, { fallbackTitle });
}

export async function parseQueleaQsp(opts: {
  bytes: Buffer;
  originalName?: string;
}): Promise<QueleaPackParseResult> {
  if (opts.bytes.byteLength === 0) {
    return {
      ok: false,
      error: "empty_file",
      message: "El pack .qsp está vacío.",
    };
  }
  if (opts.bytes.byteLength > QUELEA_IMPORT_MAX_BYTES) {
    return {
      ok: false,
      error: "too_large",
      message: `Pack demasiado grande (máx. ${Math.round(QUELEA_IMPORT_MAX_BYTES / (1024 * 1024))} MB).`,
    };
  }

  let zip: JSZip;
  try {
    zip = await JSZip.loadAsync(opts.bytes);
  } catch {
    return {
      ok: false,
      error: "invalid_zip",
      message: "No se pudo abrir el .qsp (ZIP de canciones Quelea).",
    };
  }

  const songs: ParsedQueleaSong[] = [];
  const skipped: Array<{ file: string; error: string; message: string }> = [];
  const names = Object.keys(zip.files).sort((a, b) =>
    a.localeCompare(b, "es", { sensitivity: "base" }),
  );

  for (const name of names) {
    const entry = zip.files[name];
    if (!entry || entry.dir) {
      continue;
    }
    const base = name.split("/").pop() || name;
    if (base.startsWith(".") || base.startsWith("__MACOSX")) {
      continue;
    }
    if (!base.toLowerCase().endsWith(".xml")) {
      continue;
    }
    if (songs.length >= QUELEA_IMPORT_MAX_SONGS) {
      skipped.push({
        file: base,
        error: "too_many",
        message: `Límite de ${QUELEA_IMPORT_MAX_SONGS} canciones por pack.`,
      });
      continue;
    }
    let xml: string;
    try {
      xml = await entry.async("string");
    } catch (err) {
      skipped.push({
        file: base,
        error: "read_failed",
        message: err instanceof Error ? err.message : String(err),
      });
      continue;
    }
    const parsed = parseQueleaSongXml(xml, {
      fallbackTitle: base.replace(/\.xml$/i, ""),
    });
    if (!parsed.ok) {
      skipped.push({
        file: base,
        error: parsed.error,
        message: parsed.message,
      });
      continue;
    }
    songs.push({ song: parsed.song, sourceName: base });
  }

  if (songs.length === 0) {
    return {
      ok: false,
      error: "no_songs",
      message:
        skipped[0]?.message ||
        "El pack .qsp no contiene canciones Quelea válidas.",
    };
  }

  return { ok: true, songs, skipped };
}

/** Normalize title+author for idempotent match. */
export function songIdentityKey(title: string, artist?: string): string {
  const t = title.trim().toLowerCase().replace(/\s+/g, " ");
  const a = (artist ?? "").trim().toLowerCase().replace(/\s+/g, " ");
  return `${t}\0${a}`;
}
