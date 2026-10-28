import { useState } from "react";
import type { LibrarySongPayload, LibrarySongStanza } from "@promedia/shared";
import { cryptoRandomId } from "@promedia/item-types";
import { useLocale } from "../locale/LocaleContext";
import { DialogBrand } from "./DialogBrand";

export type SongEditorProps = {
  open: boolean;
  initial?: LibrarySongPayload | null;
  busy?: boolean;
  error?: string | null;
  onClose: () => void;
  onSave: (song: LibrarySongPayload) => void;
};

function stanzaToText(stanza: LibrarySongStanza): string {
  return stanza.lines.join("\n");
}

function textToLines(text: string): string[] {
  // Keep blank editor rows while typing; projection normalizes blanks later.
  return text.replace(/\r\n/g, "\n").replace(/\r/g, "\n").split("\n");
}

/** Song editor: title, optional artist, lyrics by stanzas → 5-line projection. */
export function SongEditor({
  open,
  initial,
  busy,
  error,
  onClose,
  onSave,
}: SongEditorProps) {
  const { t } = useLocale();
  const [title, setTitle] = useState(initial?.title ?? "");
  const [artist, setArtist] = useState(initial?.artist ?? "");
  const [stanzas, setStanzas] = useState<LibrarySongStanza[]>(
    initial?.stanzas?.length
      ? initial.stanzas.map((s) => ({ ...s, lines: [...s.lines] }))
      : [{ id: cryptoRandomId(), lines: [""] }],
  );

  if (!open) {
    return null;
  }

  const updateStanza = (index: number, text: string) => {
    setStanzas((prev) =>
      prev.map((stanza, i) =>
        i === index ? { ...stanza, lines: textToLines(text) } : stanza,
      ),
    );
  };

  const addStanza = () => {
    setStanzas((prev) => [...prev, { id: cryptoRandomId(), lines: [""] }]);
  };

  const removeStanza = (index: number) => {
    setStanzas((prev) => {
      if (prev.length <= 1) {
        return prev;
      }
      return prev.filter((_, i) => i !== index);
    });
  };

  const save = () => {
    onSave({
      title: title.trim(),
      artist: artist.trim() || undefined,
      stanzas: stanzas.map((s) => ({
        id: s.id,
        lines: s.lines,
      })),
    });
  };

  return (
    <div
      className="promedia-dialog-backdrop"
      role="presentation"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget && !busy) {
          onClose();
        }
      }}
    >
      <div
        className="promedia-dialog song-editor-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="song-editor-title"
      >
        <DialogBrand id="song-editor-title">
          {t("library.songEditorTitle")}
        </DialogBrand>
        <p className="promedia-dialog-hint">{t("library.songEditorHint")}</p>

        <label className="promedia-dialog-field">
          {t("library.songTitle")}
          <input
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder={t("library.songTitlePlaceholder")}
            autoFocus
          />
        </label>

        <label className="promedia-dialog-field">
          {t("library.songArtist")}
          <input
            value={artist}
            onChange={(e) => setArtist(e.target.value)}
            placeholder={t("library.songArtistPlaceholder")}
          />
        </label>

        <div className="song-stanzas">
          {stanzas.map((stanza, index) => (
            <div key={stanza.id} className="song-stanza">
              <div className="song-stanza-header">
                <span>
                  {t("library.stanza")} {index + 1}
                </span>
                <button
                  type="button"
                  className="btn btn-ghost"
                  disabled={stanzas.length <= 1 || busy}
                  onClick={() => removeStanza(index)}
                >
                  {t("library.removeStanza")}
                </button>
              </div>
              <textarea
                rows={4}
                value={stanzaToText(stanza)}
                onChange={(e) => updateStanza(index, e.target.value)}
                placeholder={t("library.stanzaPlaceholder")}
              />
            </div>
          ))}
        </div>

        <button
          type="button"
          className="btn btn-ghost"
          disabled={busy}
          onClick={addStanza}
        >
          {t("library.addStanza")}
        </button>

        {error ? <p className="promedia-dialog-error">{error}</p> : null}

        <div className="promedia-dialog-actions">
          <button
            type="button"
            className="btn btn-ghost"
            disabled={busy}
            onClick={onClose}
          >
            {t("text.cancel")}
          </button>
          <button
            type="button"
            className="btn"
            disabled={busy || !title.trim()}
            onClick={save}
          >
            {busy ? t("library.saving") : t("text.save")}
          </button>
        </div>
      </div>
    </div>
  );
}
