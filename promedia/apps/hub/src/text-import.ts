import { PDFParse } from "pdf-parse";

export const TEXT_IMPORT_MAX_BYTES = 8 * 1024 * 1024;
export const TEXT_IMPORT_MAX_PAGES = 100;

export type ParseDocumentResult = {
  pages: string[];
  title?: string;
  truncated: boolean;
};

/**
 * Extract text from a PDF buffer (one string per page).
 * Lightweight hub-side parse via pdf-parse — avoids shipping pdf.js to Control.
 */
export async function parsePdfBuffer(
  bytes: Buffer,
  opts?: { maxPages?: number },
): Promise<ParseDocumentResult> {
  const maxPages = opts?.maxPages ?? TEXT_IMPORT_MAX_PAGES;
  const parser = new PDFParse({ data: bytes });
  try {
    const info = await parser.getInfo().catch(() => null);
    const textResult = await parser.getText();
    const pagesRaw = (textResult.pages ?? []).map((p) =>
      typeof p?.text === "string" ? p.text : "",
    );

    const truncated = pagesRaw.length > maxPages;
    const pages = (truncated ? pagesRaw.slice(0, maxPages) : pagesRaw)
      .map((p) => p.replace(/\r\n/g, "\n").trim())
      .filter(Boolean);

    const metaTitle = (info as { info?: { Title?: unknown } } | null)?.info
      ?.Title;
    const title =
      typeof metaTitle === "string" && metaTitle.trim()
        ? metaTitle.trim().slice(0, 120)
        : undefined;

    return { pages, title, truncated };
  } finally {
    await parser.destroy().catch(() => undefined);
  }
}
