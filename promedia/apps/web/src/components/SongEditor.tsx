import { useState } from "react";
import type { LibrarySongPayload } from "@promedia/shared";
import { flattenSongLyrics } from "@promedia/shared";
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

function textToLines(text: string): string[] {
  // Keep blank editor rows while typing; agent skips empties when syncing.
  return text.replace(/\r\n/g, "\n").replace(/\r/g, "\n").split("\n");
}

function initialLyricsText(initial?: LibrarySongPayload | null): string {
  if (!initial?.stanzas?.length) {
    return "";
  }
  return flattenSongLyrics(initial).stanzas[0]?.lines.join("\n") ?? "";
}

/** Song editor: title, optional artist, one lyric block for karaoke sync. */
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
  const [lyricsId] = useState(
    () => initial?.stanzas?.[0]?.id || cryptoRandomId(),
  );
  const [lyrics, setLyrics] = useState(() => initialLyricsText(initial));

  if (!open) {
    return null;
  }

  const save = () => {
    onSave({
      title: title.trim(),
      artist: artist.trim() || undefined,
      stanzas: [
        {
          id: lyricsId,
          lines: textToLines(lyrics),
        },
      ],
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

        <label className="promedia-dialog-field song-lyrics-field">
          {t("library.stanza")}
          <textarea
            rows={14}
            value={lyrics}
            onChange={(e) => setLyrics(e.target.value)}
            placeholder={t("library.stanzaPlaceholder")}
            disabled={busy}
          />
        </label>

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
