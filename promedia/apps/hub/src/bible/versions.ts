/**
 * Bible versions ProMedia ships in the Guion «Biblia» picker.
 *
 * RVR1960 is intentionally omitted: it remains under copyright
 * (United Bible Societies / Sociedades Bíblicas) and is not redistributed
 * by the free APIs we use (Midvash / bible-api.com). Prefer RV1909 (public domain).
 */

export type BibleVersionProvider = "midvash" | "bible-api" | "local";

export type BibleVersion = {
  id: string;
  label: string;
  shortLabel: string;
  language: "es" | "en" | string;
  provider: BibleVersionProvider;
  /** Midvash slug or bible-api.com translation id (unused for local). */
  apiId: string;
  license: string;
  default?: boolean;
  /** Present when provider === "local". */
  source?: "local" | "remote";
  libraryItemId?: string;
  note?: string;
};

export const BIBLE_VERSIONS: readonly BibleVersion[] = [
  {
    id: "rvr1909",
    label: "Reina-Valera 1909",
    shortLabel: "RV1909",
    language: "es",
    provider: "midvash",
    apiId: "rvr1909",
    license: "Dominio público",
    default: true,
  },
  {
    id: "pdt",
    label: "Palabra de Dios para ti",
    shortLabel: "PDT",
    language: "es",
    provider: "midvash",
    apiId: "pdt",
    license: "CC BY 4.0 (mostrar atribución)",
  },
  {
    id: "kjv",
    label: "King James Version",
    shortLabel: "KJV",
    language: "en",
    provider: "bible-api",
    apiId: "kjv",
    license: "Public domain",
  },
  {
    id: "asv",
    label: "American Standard Version",
    shortLabel: "ASV",
    language: "en",
    provider: "bible-api",
    apiId: "asv",
    license: "Public domain",
  },
];

export function getVersion(id: string): BibleVersion | undefined {
  const key = id.trim().toLowerCase();
  return BIBLE_VERSIONS.find((v) => v.id === key);
}

export function defaultVersionId(): string {
  return BIBLE_VERSIONS.find((v) => v.default)?.id ?? BIBLE_VERSIONS[0]!.id;
}
