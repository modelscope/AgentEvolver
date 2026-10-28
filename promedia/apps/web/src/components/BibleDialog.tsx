import { useEffect, useMemo, useRef, useState } from "react";
import { useLocale } from "../locale/LocaleContext";
import type { BiblePayload } from "@promedia/shared";
import {
  fetchBibleBooks,
  fetchBiblePassage,
  fetchBibleVersions,
  type BibleBookInfo,
  type BibleVersionInfo,
} from "../lib/bible";
import { DialogBrand } from "./DialogBrand";

export type BibleDialogProps = {
  open: boolean;
  onClose: () => void;
  onAdd: (payload: BiblePayload) => void;
};

export function BibleDialog({ open, onClose, onAdd }: BibleDialogProps) {
  const { t } = useLocale();

  const searchRef = useRef<HTMLInputElement>(null);
  const [versions, setVersions] = useState<BibleVersionInfo[]>([]);
  const [books, setBooks] = useState<BibleBookInfo[]>([]);
  const [versionId, setVersionId] = useState("rvr1909");
  const [bookId, setBookId] = useState("john");
  const [chapter, setChapter] = useState(3);
  const [verseStart, setVerseStart] = useState(16);
  const [verseEnd, setVerseEnd] = useState(16);
  const [searchQ, setSearchQ] = useState("");
  const [loadingMeta, setLoadingMeta] = useState(false);
  const [loadingPassage, setLoadingPassage] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<BiblePayload | null>(null);
  const [licenseNote, setLicenseNote] = useState<string | null>(null);

  const selectedBook = useMemo(
    () => books.find((b) => b.id === bookId) ?? null,
    [books, bookId],
  );
  const maxChapter = selectedBook?.chapters ?? 150;

  useEffect(() => {
    if (!open) {
      return;
    }
    setError(null);
    setPreview(null);
    setLoadingMeta(true);
    void (async () => {
      try {
        const [vRes, bRes] = await Promise.all([
          fetchBibleVersions(),
          fetchBibleBooks(),
        ]);
        setVersions(vRes.versions);
        setBooks(bRes.books);
        setVersionId(vRes.defaultVersionId || vRes.versions[0]?.id || "rvr1909");
        setLicenseNote(vRes.note ?? null);
        if (!bRes.books.some((b) => b.id === bookId) && bRes.books[0]) {
          setBookId(bRes.books[0].id);
        }
        window.setTimeout(() => searchRef.current?.focus(), 0);
      } catch (err) {
        setError(String(err));
      } finally {
        setLoadingMeta(false);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps -- open only
  }, [open]);

  useEffect(() => {
    if (chapter > maxChapter) {
      setChapter(maxChapter);
    }
  }, [chapter, maxChapter]);

  if (!open) {
    return null;
  }

  const toPayload = (
    data: Awaited<ReturnType<typeof fetchBiblePassage>>,
  ): BiblePayload => ({
    versionId: data.versionId,
    versionLabel: data.versionLabel,
    bookId: data.bookId,
    bookName: data.bookName,
    chapter: data.chapter,
    verseStart: data.verseStart,
    verseEnd: data.verseEnd,
    reference: data.reference,
    copyright: data.copyright,
    fromCache: data.fromCache,
    align: "center",
    slides: data.slides,
  });

  const runLookup = async (mode: "fields" | "search") => {
    setError(null);
    setPreview(null);
    setLoadingPassage(true);
    try {
      const data =
        mode === "search"
          ? await fetchBiblePassage({
              version: versionId,
              q: searchQ.trim(),
            })
          : await fetchBiblePassage({
              version: versionId,
              book: bookId,
              chapter,
              verseStart,
              verseEnd: Math.max(verseEnd, verseStart),
            });
      const payload = toPayload(data);
      setPreview(payload);
      if (data.fromCache) {
        setError(null);
      }
    } catch (err) {
      const e = err as Error & { code?: string };
      if (e.code === "api_failed" || /api_failed|bible_http_502/.test(e.message)) {
        setError(t("bible.apiFailed"));
      } else if (e.code === "invalid_reference") {
        setError(t("bible.invalidReference"));
      } else {
        setError(e.message || t("bible.apiFailed"));
      }
    } finally {
      setLoadingPassage(false);
    }
  };

  const addToGuion = () => {
    if (!preview) {
      return;
    }
    onAdd(preview);
  };

  return (
    <div
      className="promedia-dialog-backdrop"
      role="presentation"
      onClick={(e) => {
        if (e.target === e.currentTarget) {
          onClose();
        }
      }}
    >
      <div
        className="promedia-dialog bible-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="bible-dialog-title"
      >
        <DialogBrand id="bible-dialog-title">
          {t("bible.dialogTitle")}
        </DialogBrand>

        {loadingMeta ? (
          <p className="promedia-dialog-hint">{t("bible.loading")}</p>
        ) : (
          <>
            <label className="promedia-dialog-field">
              {t("bible.version")}
              <select
                value={versionId}
                onChange={(e) => {
                  setVersionId(e.target.value);
                  setPreview(null);
                }}
              >
                {versions.map((v) => (
                  <option key={v.id} value={v.id}>
                    {v.shortLabel} — {v.label}
                    {v.source === "local"
                      ? ` [${t("bible.localBadge")}]`
                      : ""}
                  </option>
                ))}
              </select>
            </label>

            <label className="promedia-dialog-field">
              {t("bible.searchRef")}
              <input
                ref={searchRef}
                value={searchQ}
                onChange={(e) => setSearchQ(e.target.value)}
                placeholder={t("bible.searchPlaceholder")}
                autoComplete="off"
                onKeyDown={(e) => {
                  if (e.key === "Enter" && searchQ.trim()) {
                    e.preventDefault();
                    void runLookup("search");
                  } else if (e.key === "Escape") {
                    e.preventDefault();
                    onClose();
                  }
                }}
              />
            </label>
            <div className="bible-dialog-row">
              <button
                type="button"
                className="btn"
                disabled={!searchQ.trim() || loadingPassage}
                onClick={() => void runLookup("search")}
              >
                {t("bible.search")}
              </button>
            </div>

            <p className="bible-dialog-or">{t("bible.orPick")}</p>

            <div className="bible-dialog-grid">
              <label className="promedia-dialog-field bible-dialog-book">
                {t("bible.book")}
                <select
                  value={bookId}
                  onChange={(e) => {
                    setBookId(e.target.value);
                    setChapter(1);
                    setVerseStart(1);
                    setVerseEnd(1);
                    setPreview(null);
                  }}
                >
                  {books.map((b) => (
                    <option key={b.id} value={b.id}>
                      {b.nameEs}
                    </option>
                  ))}
                </select>
              </label>
              <label className="promedia-dialog-field bible-dialog-num">
                {t("bible.chapter")}
                <input
                  type="number"
                  inputMode="numeric"
                  maxLength={3}
                  min={1}
                  max={maxChapter}
                  value={chapter}
                  onChange={(e) => {
                    const digits = e.target.value.replace(/\D/g, "").slice(0, 3);
                    setChapter(
                      Math.min(
                        maxChapter,
                        Math.max(1, Number(digits) || 1),
                      ),
                    );
                    setPreview(null);
                  }}
                />
              </label>
              <label className="promedia-dialog-field bible-dialog-num">
                {t("bible.verseFrom")}
                <input
                  type="number"
                  inputMode="numeric"
                  maxLength={3}
                  min={1}
                  max={999}
                  value={verseStart}
                  onChange={(e) => {
                    const digits = e.target.value.replace(/\D/g, "").slice(0, 3);
                    const n = Math.min(999, Math.max(1, Number(digits) || 1));
                    setVerseStart(n);
                    if (verseEnd < n) {
                      setVerseEnd(n);
                    }
                    setPreview(null);
                  }}
                />
              </label>
              <label className="promedia-dialog-field bible-dialog-num">
                {t("bible.verseTo")}
                <input
                  type="number"
                  inputMode="numeric"
                  maxLength={3}
                  min={verseStart}
                  max={999}
                  value={verseEnd}
                  onChange={(e) => {
                    const digits = e.target.value.replace(/\D/g, "").slice(0, 3);
                    setVerseEnd(
                      Math.min(
                        999,
                        Math.max(
                          verseStart,
                          Number(digits) || verseStart,
                        ),
                      ),
                    );
                    setPreview(null);
                  }}
                />
              </label>
            </div>

            <div className="bible-dialog-row">
              <button
                type="button"
                className="btn btn-primary"
                disabled={loadingPassage}
                onClick={() => void runLookup("fields")}
              >
                {loadingPassage
                  ? t("bible.loading")
                  : t("bible.preview")}
              </button>
            </div>
          </>
        )}

        {error ? (
          <p className="promedia-dialog-error" role="alert">
            {error}
          </p>
        ) : null}

        {preview ? (
          <div className="bible-preview-box">
            <p className="bible-preview-ref">
              {preview.reference} · {preview.versionLabel}
              {preview.fromCache ? ` · ${t("bible.fromCache")}` : ""}
            </p>
            <ul className="bible-preview-verses">
              {preview.slides.map((slide) => (
                <li key={slide.id}>
                  {slide.lines.map((line, i) => (
                    <span key={i} className={i === 0 ? "bible-v-ref" : "bible-v-text"}>
                      {line}
                      {i < slide.lines.length - 1 ? " " : ""}
                    </span>
                  ))}
                </li>
              ))}
            </ul>
            {preview.copyright ? (
              <p className="bible-preview-copy">{preview.copyright}</p>
            ) : null}
          </div>
        ) : null}

        {licenseNote && !preview ? (
          <p className="promedia-dialog-hint">{licenseNote}</p>
        ) : (
          <p className="promedia-dialog-hint">{t("bible.hint")}</p>
        )}

        <div className="promedia-dialog-actions">
          <button type="button" className="btn" onClick={onClose}>
            {t("text.cancel")}
          </button>
          <button
            type="button"
            className="btn btn-primary"
            disabled={!preview || loadingPassage}
            onClick={addToGuion}
          >
            {t("bible.add")}
          </button>
        </div>
      </div>
    </div>
  );
}
