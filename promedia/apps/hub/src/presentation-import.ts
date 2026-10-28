import JSZip from "jszip";
import { randomBytes } from "node:crypto";
import { parsePdfBuffer } from "./text-import.js";

export const PRESENTATION_IMPORT_MAX_BYTES = 80 * 1024 * 1024;
export const PRESENTATION_IMPORT_MAX_SLIDES = 120;

const MACRO_EXTS = new Set([".pptm", ".ppsm"]);

const OFFICE_ZIP_EXTS = new Set([
  ".pptx",
  ".ppsx",
  ".potx",
  ".odp",
]);

export type RenderedSlide = {
  id: string;
  mime: string;
  bytes: Buffer;
  fileName: string;
};

export type PresentationImportResult =
  | {
      ok: true;
      title?: string;
      slides: RenderedSlide[];
      truncated: boolean;
    }
  | { ok: false; error: string; message: string };

function extOf(name: string): string {
  const i = name.lastIndexOf(".");
  return i >= 0 ? name.slice(i).toLowerCase() : "";
}

function newSlideId(): string {
  return randomBytes(8).toString("hex");
}

function escapeXml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/** Dark full-bleed SVG page with centered white lines (projection-friendly). */
export function linesToSvgSlide(
  lines: string[],
  opts?: { title?: string },
): Buffer {
  const usable = lines
    .map((l) => l.replace(/\s+/g, " ").trim())
    .filter(Boolean)
    .slice(0, 16);
  const display =
    usable.length > 0
      ? usable
      : opts?.title
        ? [opts.title]
        : ["(diapositiva vacía)"];
  const fontSize = display.length <= 4 ? 48 : display.length <= 8 ? 36 : 28;
  const lineHeight = fontSize * 1.25;
  const blockHeight = display.length * lineHeight;
  const startY = Math.max(80, (1080 - blockHeight) / 2 + fontSize);
  const textNodes = display
    .map((line, i) => {
      const y = startY + i * lineHeight;
      return `<text x="960" y="${y}" text-anchor="middle" fill="#f4f7fb" font-family="Segoe UI, system-ui, sans-serif" font-size="${fontSize}">${escapeXml(line)}</text>`;
    })
    .join("\n");
  const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="1920" height="1080" viewBox="0 0 1920 1080">
  <rect width="1920" height="1080" fill="#0c1016"/>
  ${textNodes}
</svg>
`;
  return Buffer.from(svg, "utf8");
}

function collectTextFromXml(xml: string): string[] {
  const lines: string[] = [];
  const re = /<(?:a:t|text:p|text:span)[^>]*>([^<]*)<\/(?:a:t|text:p|text:span)>/gi;
  let m: RegExpExecArray | null;
  while ((m = re.exec(xml))) {
    const t = (m[1] ?? "").replace(/\s+/g, " ").trim();
    if (t) {
      lines.push(t);
    }
  }
  return lines;
}

function mimeForEmbedded(name: string): string | null {
  const lower = name.toLowerCase();
  if (lower.endsWith(".png")) {
    return "image/png";
  }
  if (lower.endsWith(".jpg") || lower.endsWith(".jpeg")) {
    return "image/jpeg";
  }
  if (lower.endsWith(".gif")) {
    return "image/gif";
  }
  if (lower.endsWith(".webp")) {
    return "image/webp";
  }
  if (lower.endsWith(".svg")) {
    return "image/svg+xml";
  }
  return null;
}

async function importPdf(
  bytes: Buffer,
  originalName: string,
): Promise<PresentationImportResult> {
  const parsed = await parsePdfBuffer(bytes);
  const pages = parsed.pages.length > 0 ? parsed.pages : [""];
  const truncated = pages.length > PRESENTATION_IMPORT_MAX_SLIDES;
  const slice = truncated
    ? pages.slice(0, PRESENTATION_IMPORT_MAX_SLIDES)
    : pages;
  const slides: RenderedSlide[] = slice.map((pageText, index) => {
    const lines = pageText
      .split(/\n+/)
      .map((l) => l.trim())
      .filter(Boolean);
    const id = newSlideId();
    return {
      id,
      mime: "image/svg+xml",
      bytes: linesToSvgSlide(lines, {
        title: parsed.title || originalName,
      }),
      fileName: `slide-${String(index + 1).padStart(3, "0")}.svg`,
    };
  });
  return {
    ok: true,
    title: parsed.title || originalName.replace(/\.[^.]+$/, ""),
    slides,
    truncated: truncated || parsed.truncated,
  };
}

async function importPptxLike(
  bytes: Buffer,
  originalName: string,
): Promise<PresentationImportResult> {
  const zip = await JSZip.loadAsync(bytes);
  const slidePaths = Object.keys(zip.files)
    .filter(
      (p) =>
        /^ppt\/slides\/slide\d+\.xml$/i.test(p) ||
        /^content\.xml$/i.test(p),
    )
    .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));

  // ODP: single content.xml with draw:page nodes
  if (slidePaths.length === 1 && /content\.xml$/i.test(slidePaths[0]!)) {
    return importOdp(zip, originalName);
  }

  if (slidePaths.length === 0) {
    return {
      ok: false,
      error: "no_slides",
      message:
        "No se encontraron diapositivas en el archivo. Probá exportar a .pptx o .pdf.",
    };
  }

  const truncated = slidePaths.length > PRESENTATION_IMPORT_MAX_SLIDES;
  const paths = truncated
    ? slidePaths.slice(0, PRESENTATION_IMPORT_MAX_SLIDES)
    : slidePaths;
  const slides: RenderedSlide[] = [];

  for (let i = 0; i < paths.length; i++) {
    const path = paths[i]!;
    const xml = await zip.file(path)!.async("string");
    const lines = collectTextFromXml(xml);

    // Prefer a large embedded image from this slide's relationships if present.
    const slideNum = /slide(\d+)\.xml/i.exec(path)?.[1];
    let imageSlide: RenderedSlide | null = null;
    if (slideNum) {
      const relPath = `ppt/slides/_rels/slide${slideNum}.xml.rels`;
      const relFile = zip.file(relPath);
      if (relFile) {
        const relXml = await relFile.async("string");
        const targets = [
          ...relXml.matchAll(/Target="([^"]+)"/gi),
        ].map((m) => m[1] ?? "");
        for (const target of targets) {
          const mediaPath = target.startsWith("../")
            ? `ppt/${target.replace(/^\.\.\//, "")}`
            : target.startsWith("ppt/")
              ? target
              : `ppt/slides/${target}`;
          const normalized = mediaPath.replace(/\\/g, "/");
          const mime = mimeForEmbedded(normalized);
          const media = mime ? zip.file(normalized) : null;
          if (mime && media) {
            const imgBytes = Buffer.from(await media.async("uint8array"));
            if (imgBytes.byteLength > 8_000) {
              imageSlide = {
                id: newSlideId(),
                mime,
                bytes: imgBytes,
                fileName: `slide-${String(i + 1).padStart(3, "0")}${extOf(normalized) || ".bin"}`,
              };
              break;
            }
          }
        }
      }
    }

    if (imageSlide && lines.length < 2) {
      slides.push(imageSlide);
      continue;
    }

    slides.push({
      id: newSlideId(),
      mime: "image/svg+xml",
      bytes: linesToSvgSlide(lines, { title: originalName }),
      fileName: `slide-${String(i + 1).padStart(3, "0")}.svg`,
    });
  }

  if (slides.length === 0) {
    return {
      ok: false,
      error: "no_slides",
      message: "No se pudieron renderizar diapositivas de la presentación.",
    };
  }

  return {
    ok: true,
    title: originalName.replace(/\.[^.]+$/, ""),
    slides,
    truncated,
  };
}

async function importOdp(
  zip: JSZip,
  originalName: string,
): Promise<PresentationImportResult> {
  const content = await zip.file("content.xml")!.async("string");
  const pages = content.split(/<draw:page\b/i).slice(1);
  if (pages.length === 0) {
    const lines = collectTextFromXml(content);
    return {
      ok: true,
      title: originalName.replace(/\.[^.]+$/, ""),
      slides: [
        {
          id: newSlideId(),
          mime: "image/svg+xml",
          bytes: linesToSvgSlide(lines, { title: originalName }),
          fileName: "slide-001.svg",
        },
      ],
      truncated: false,
    };
  }
  const truncated = pages.length > PRESENTATION_IMPORT_MAX_SLIDES;
  const slice = truncated
    ? pages.slice(0, PRESENTATION_IMPORT_MAX_SLIDES)
    : pages;
  const slides: RenderedSlide[] = slice.map((chunk, index) => {
    const lines = collectTextFromXml(chunk);
    return {
      id: newSlideId(),
      mime: "image/svg+xml",
      bytes: linesToSvgSlide(lines, { title: originalName }),
      fileName: `slide-${String(index + 1).padStart(3, "0")}.svg`,
    };
  });
  return {
    ok: true,
    title: originalName.replace(/\.[^.]+$/, ""),
    slides,
    truncated,
  };
}

/**
 * Import a presentation file into renderable projection slides.
 * Rejects macro-enabled Office formats (.pptm / .ppsm).
 * Legacy binary .ppt is accepted as upload target but cannot be rendered yet.
 */
export async function importPresentationFile(opts: {
  bytes: Buffer;
  originalName: string;
  mime?: string;
}): Promise<PresentationImportResult> {
  const name = opts.originalName || "presentation";
  const ext = extOf(name);

  if (MACRO_EXTS.has(ext)) {
    return {
      ok: false,
      error: "macros_not_supported",
      message:
        "No se admiten presentaciones con macros (.pptm / .ppsm) por seguridad. Guardá una copia sin macros (.pptx / .ppsx) o exportá a PDF e importá de nuevo.",
    };
  }

  if (opts.bytes.byteLength > PRESENTATION_IMPORT_MAX_BYTES) {
    return {
      ok: false,
      error: "file_too_large",
      message: "El archivo es demasiado grande (máx. 80 MB).",
    };
  }

  if (ext === ".pdf" || opts.mime === "application/pdf") {
    try {
      return await importPdf(opts.bytes, name);
    } catch (err) {
      return {
        ok: false,
        error: "parse_failed",
        message: `No se pudo leer el PDF: ${String(err)}`,
      };
    }
  }

  if (ext === ".ppt") {
    return {
      ok: false,
      error: "legacy_ppt",
      message:
        "El formato binario .ppt no se puede proyectar aún. Abrí el archivo en PowerPoint o LibreOffice, guardalo como .pptx o .pdf, e importá de nuevo.",
    };
  }

  if (OFFICE_ZIP_EXTS.has(ext)) {
    try {
      return await importPptxLike(opts.bytes, name);
    } catch (err) {
      return {
        ok: false,
        error: "parse_failed",
        message: `No se pudo leer la presentación: ${String(err)}`,
      };
    }
  }

  return {
    ok: false,
    error: "unsupported_type",
    message:
      "Formato no soportado. Usá .pptx, .ppsx, .potx, .odp o .pdf (sin macros .pptm/.ppsm).",
  };
}

export function isPresentationFileName(name: string): boolean {
  const ext = extOf(name);
  return (
    ext === ".pdf" ||
    ext === ".ppt" ||
    OFFICE_ZIP_EXTS.has(ext) ||
    MACRO_EXTS.has(ext)
  );
}

export function isMacroPresentationName(name: string): boolean {
  return MACRO_EXTS.has(extOf(name));
}
