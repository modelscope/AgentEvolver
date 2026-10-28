import type { TextSlide } from "@promedia/shared";
import { projectionTextSlides, TEXT_PROJECTION_LINES } from "@promedia/shared";
import { cryptoRandomId } from "@promedia/item-types";
import { hubHttpBase } from "./hub";

/** Soft limits shown in UI; oversized files are rejected. */
export const TEXT_IMPORT_MAX_BYTES = 8 * 1024 * 1024;
export const TEXT_IMPORT_MAX_SLIDES = 200;
export const TEXT_IMPORT_MAX_CHARS_PER_SLIDE = 900;
/** Projection shows this many lines per slide (←/→ advances chunks). */
export { TEXT_PROJECTION_LINES };

export type TextImportResult = {
  title?: string;
  slides: TextSlide[];
  truncated: boolean;
};

const TEXT_EXTS = new Set([".txt", ".md", ".text", ".markdown"]);
const XML_EXTS = new Set([".xml"]);
const PDF_EXTS = new Set([".pdf"]);

function extOf(name: string): string {
  const i = name.lastIndexOf(".");
  return i >= 0 ? name.slice(i).toLowerCase() : "";
}

export function isSupportedTextImport(file: File): boolean {
  const ext = extOf(file.name);
  return TEXT_EXTS.has(ext) || XML_EXTS.has(ext) || PDF_EXTS.has(ext);
}

/** Split plain text into slide-sized chunks (paragraphs, then length). */
export function chunkTextToSlides(
  raw: string,
  opts?: { maxSlides?: number; maxChars?: number },
): { slides: TextSlide[]; truncated: boolean } {
  const maxSlides = opts?.maxSlides ?? TEXT_IMPORT_MAX_SLIDES;
  const maxChars = opts?.maxChars ?? TEXT_IMPORT_MAX_CHARS_PER_SLIDE;
  const normalized = raw.replace(/\r\n/g, "\n").replace(/\r/g, "\n").trim();
  if (!normalized) {
    return {
      slides: [{ id: cryptoRandomId(), lines: [""] }],
      truncated: false,
    };
  }

  const paragraphs = normalized
    .split(/\n{2,}/)
    .map((p) => p.trim())
    .filter(Boolean);

  const chunks: string[] = [];
  for (const para of paragraphs) {
    if (para.length <= maxChars) {
      chunks.push(para);
      continue;
    }
    // Long paragraph / page: split on newlines, then hard-wrap
    const lines = para.split("\n");
    let buf = "";
    const flush = () => {
      if (buf.trim()) {
        chunks.push(buf.trim());
      }
      buf = "";
    };
    for (const line of lines) {
      const next = buf ? `${buf}\n${line}` : line;
      if (next.length > maxChars && buf) {
        flush();
        if (line.length <= maxChars) {
          buf = line;
        } else {
          for (let i = 0; i < line.length; i += maxChars) {
            chunks.push(line.slice(i, i + maxChars));
          }
        }
      } else if (next.length > maxChars) {
        for (let i = 0; i < next.length; i += maxChars) {
          chunks.push(next.slice(i, i + maxChars));
        }
        buf = "";
      } else {
        buf = next;
      }
    }
    flush();
  }

  const rawSlides = chunks.map((text) => ({
    id: cryptoRandomId(),
    lines: text.split("\n"),
  }));
  // Enforce ≤5 lines per slide so import matches Output/Live ←/→ chunks.
  const projected = projectionTextSlides(rawSlides);
  const truncated = projected.length > maxSlides || chunks.length > maxSlides;
  const slides = truncated ? projected.slice(0, maxSlides) : projected;
  return { slides, truncated };
}

function extractXmlText(xml: string): string {
  try {
    const doc = new DOMParser().parseFromString(xml, "application/xml");
    const err = doc.querySelector("parsererror");
    if (err) {
      // Fallback: strip tags roughly
      return xml
        .replace(/<\?xml[\s\S]*?\?>/i, "")
        .replace(/<!\[CDATA\[([\s\S]*?)\]\]>/g, "$1")
        .replace(/<[^>]+>/g, "\n")
        .replace(/[ \t]+\n/g, "\n")
        .replace(/\n{3,}/g, "\n\n");
    }
    const walk = (node: Node): string[] => {
      if (node.nodeType === Node.TEXT_NODE) {
        const t = (node.textContent ?? "").trim();
        return t ? [t] : [];
      }
      if (node.nodeType !== Node.ELEMENT_NODE) {
        return [];
      }
      const el = node as Element;
      const tag = el.tagName.toLowerCase();
      if (tag === "script" || tag === "style") {
        return [];
      }
      const parts: string[] = [];
      for (const child of Array.from(el.childNodes)) {
        parts.push(...walk(child));
      }
      return parts;
    };
    return walk(doc.documentElement ?? doc).join("\n\n");
  } catch {
    return xml.replace(/<[^>]+>/g, "\n");
  }
}

async function parsePdfViaHub(file: File): Promise<{
  pages: string[];
  title?: string;
}> {
  const form = new FormData();
  form.append("file", file, file.name);
  const res = await fetch(`${hubHttpBase()}/api/v1/parse/document`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const body = (await res.json().catch(() => ({}))) as { error?: string };
    throw new Error(body.error ?? `parse_failed_${res.status}`);
  }
  return (await res.json()) as { pages: string[]; title?: string };
}

/**
 * Import PDF / XML / plain text into Text slides.
 * PDF is parsed on the hub (pdf-parse); XML/txt/md in the browser.
 */
export async function importFileToTextSlides(
  file: File,
): Promise<TextImportResult> {
  if (file.size > TEXT_IMPORT_MAX_BYTES) {
    throw new Error("file_too_large");
  }
  const ext = extOf(file.name);
  const baseTitle = file.name.replace(/\.[^.]+$/, "").slice(0, 80);

  if (PDF_EXTS.has(ext) || file.type === "application/pdf") {
    const { pages, title } = await parsePdfViaHub(file);
    const joined = pages
      .map((p) => p.trim())
      .filter(Boolean)
      .join("\n\n");
    const { slides, truncated } = chunkTextToSlides(joined);
    return { title: title || baseTitle, slides, truncated };
  }

  const text = await file.text();
  if (XML_EXTS.has(ext) || file.type.includes("xml")) {
    const extracted = extractXmlText(text);
    const { slides, truncated } = chunkTextToSlides(extracted);
    return { title: baseTitle, slides, truncated };
  }

  if (TEXT_EXTS.has(ext) || file.type.startsWith("text/")) {
    const { slides, truncated } = chunkTextToSlides(text);
    return { title: baseTitle, slides, truncated };
  }

  throw new Error("unsupported_type");
}
