import { useCallback, useEffect, useState } from "react";
import { useLocale } from "../locale/LocaleContext";
import {
  deleteSavedPlaylist,
  listSavedPlaylists,
  loadSavedPlaylist,
  saveSessionPlaylist,
  type SavedPlaylistSummary,
} from "../lib/playlists";
import { DialogBrand } from "./DialogBrand";

export type SavedPlaylistPanelProps = {
  sessionId: string;
  wsTicket: string;
  disabled?: boolean;
  onHint?: (msg: string | null) => void;
  /** Compact layout for playlist column (legacy inline). */
  compact?: boolean;
  /** Dialog mode opened from Guion toolbar icons. */
  dialogMode?: boolean;
  saveOpen?: boolean;
  loadOpen?: boolean;
  onSaveOpenChange?: (open: boolean) => void;
  onLoadOpenChange?: (open: boolean) => void;
};

export function SavedPlaylistPanel({
  sessionId,
  wsTicket,
  disabled,
  onHint,
  compact,
  dialogMode,
  saveOpen = false,
  loadOpen = false,
  onSaveOpenChange,
  onLoadOpenChange,
}: SavedPlaylistPanelProps) {
  const { t } = useLocale();

  const [name, setName] = useState("");
  const [list, setList] = useState<SavedPlaylistSummary[]>([]);
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    try {
      setList(await listSavedPlaylists());
    } catch (err) {
      onHint?.(String(err));
    }
  }, [onHint]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if ((saveOpen || loadOpen) && dialogMode) {
      void refresh();
    }
  }, [saveOpen, loadOpen, dialogMode, refresh]);

  const onSave = async () => {
    if (!name.trim() || disabled) {
      return;
    }
    setBusy(true);
    try {
      await saveSessionPlaylist(sessionId, wsTicket, name.trim());
      setName("");
      onHint?.(t("playlist.savedOk"));
      await refresh();
      onSaveOpenChange?.(false);
    } catch (err) {
      onHint?.(String(err));
    } finally {
      setBusy(false);
    }
  };

  const onLoad = async (id: string) => {
    setBusy(true);
    try {
      const { skipped } = await loadSavedPlaylist(sessionId, wsTicket, id);
      onHint?.(
        skipped > 0
          ? t("playlist.loadedSkipped").replace(
              "{n}",
              String(skipped),
            )
          : t("playlist.loadedOk"),
      );
      onLoadOpenChange?.(false);
    } catch (err) {
      onHint?.(String(err));
    } finally {
      setBusy(false);
    }
  };

  const onDelete = async (id: string) => {
    setBusy(true);
    try {
      await deleteSavedPlaylist(id);
      await refresh();
    } catch (err) {
      onHint?.(String(err));
    } finally {
      setBusy(false);
    }
  };

  if (dialogMode) {
    return (
      <>
        {saveOpen ? (
          <div
            className="promedia-dialog-backdrop"
            role="presentation"
            onClick={(e) => {
              if (e.target === e.currentTarget) {
                onSaveOpenChange?.(false);
              }
            }}
          >
            <div
              className="promedia-dialog"
              role="dialog"
              aria-modal="true"
              aria-labelledby="playlist-save-title"
            >
              <DialogBrand id="playlist-save-title">
                {t("playlist.save")}
              </DialogBrand>
              <label className="promedia-dialog-field">
                {t("playlist.saveName")}
                <input
                  value={name}
                  disabled={disabled || busy}
                  autoFocus
                  placeholder={t("playlist.saveName")}
                  onChange={(e) => setName(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      void onSave();
                    } else if (e.key === "Escape") {
                      onSaveOpenChange?.(false);
                    }
                  }}
                />
              </label>
              <div className="promedia-dialog-actions">
                <button
                  type="button"
                  className="btn"
                  onClick={() => onSaveOpenChange?.(false)}
                >
                  {t("text.cancel")}
                </button>
                <button
                  type="button"
                  className="btn btn-primary"
                  disabled={disabled || busy || !name.trim()}
                  onClick={() => void onSave()}
                >
                  {t("playlist.save")}
                </button>
              </div>
            </div>
          </div>
        ) : null}

        {loadOpen ? (
          <div
            className="promedia-dialog-backdrop"
            role="presentation"
            onClick={(e) => {
              if (e.target === e.currentTarget) {
                onLoadOpenChange?.(false);
              }
            }}
          >
            <div
              className="promedia-dialog"
              role="dialog"
              aria-modal="true"
              aria-labelledby="playlist-load-title"
            >
              <DialogBrand id="playlist-load-title">
                {t("playlist.load")}
              </DialogBrand>
              {list.length === 0 ? (
                <p className="promedia-dialog-hint">
                  {t("playlist.savedEmpty")}
                </p>
              ) : (
                <ul className="saved-playlist-list saved-playlist-list-dialog">
                  {list.map((p) => (
                    <li key={p.id}>
                      <span className="saved-playlist-name">
                        {p.name}{" "}
                        <span className="footer-muted">({p.itemCount})</span>
                      </span>
                      <div className="playlist-row-actions">
                        <button
                          type="button"
                          className="btn btn-primary"
                          disabled={disabled || busy}
                          onClick={() => void onLoad(p.id)}
                        >
                          {t("playlist.load")}
                        </button>
                        <button
                          type="button"
                          className="btn btn-ghost"
                          disabled={busy}
                          onClick={() => void onDelete(p.id)}
                        >
                          {t("playlist.remove")}
                        </button>
                      </div>
                    </li>
                  ))}
                </ul>
              )}
              <div className="promedia-dialog-actions">
                <button
                  type="button"
                  className="btn"
                  onClick={() => onLoadOpenChange?.(false)}
                >
                  {t("text.cancel")}
                </button>
              </div>
            </div>
          </div>
        ) : null}
      </>
    );
  }

  return (
    <div
      className={
        compact
          ? "saved-playlist-panel saved-playlist-panel-compact"
          : "saved-playlist-panel"
      }
    >
      <h3 className="theme-controls-title">
        {t(compact ? "playlist.saveLoadTitle" : "playlist.savedTitle")}
      </h3>
      <div className="saved-playlist-save">
        <input
          value={name}
          disabled={disabled || busy}
          placeholder={t("playlist.saveName")}
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              void onSave();
            }
          }}
        />
        <button
          type="button"
          className="btn btn-primary"
          disabled={disabled || busy || !name.trim()}
          onClick={() => void onSave()}
        >
          {t("playlist.save")}
        </button>
      </div>
      {list.length === 0 ? (
        <p className="zone-empty">{t("playlist.savedEmpty")}</p>
      ) : (
        <ul className="saved-playlist-list">
          {list.map((p) => (
            <li key={p.id}>
              <span className="saved-playlist-name">
                {p.name}{" "}
                <span className="footer-muted">({p.itemCount})</span>
              </span>
              <div className="playlist-row-actions">
                <button
                  type="button"
                  className="btn"
                  disabled={disabled || busy}
                  onClick={() => void onLoad(p.id)}
                >
                  {t("playlist.load")}
                </button>
                <button
                  type="button"
                  className="btn btn-ghost"
                  disabled={busy}
                  onClick={() => void onDelete(p.id)}
                >
                  {t("playlist.remove")}
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
