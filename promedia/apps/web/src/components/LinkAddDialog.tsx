import { useEffect, useRef, useState } from "react";
import { useLocale } from "../locale/LocaleContext";
import { parseYouTubeUrl } from "@promedia/item-types";
import { DialogBrand } from "./DialogBrand";

export type LinkAddDialogProps = {
  open: boolean;
  onClose: () => void;
  /** YouTube → video item; other http(s) URL → text item with the link. */
  onAddYoutube: (url: string) => void;
  onAddGenericLink: (url: string) => void;
};

/**
 * Unified dialog for YouTube or generic http(s) links (primary «+» chooser).
 */
export function LinkAddDialog({
  open,
  onClose,
  onAddYoutube,
  onAddGenericLink,
}: LinkAddDialogProps) {
  const { t } = useLocale();

  const [url, setUrl] = useState("");
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!open) {
      return;
    }
    setUrl("");
    setError(null);
    window.setTimeout(() => inputRef.current?.focus(), 0);
  }, [open]);

  if (!open) {
    return null;
  }

  const submit = () => {
    const raw = url.trim();
    if (!raw) {
      return;
    }
    const yt = parseYouTubeUrl(raw);
    if (yt) {
      onAddYoutube(raw);
      return;
    }
    let parsed: URL;
    try {
      parsed = new URL(raw);
    } catch {
      setError(t("link.invalid"));
      return;
    }
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      setError(t("link.invalid"));
      return;
    }
    onAddGenericLink(parsed.toString());
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
        className="promedia-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="link-dialog-title"
      >
        <DialogBrand id="link-dialog-title">
          {t("link.dialogTitle")}
        </DialogBrand>
        <label className="promedia-dialog-field">
          {t("link.urlLabel")}
          <input
            ref={inputRef}
            value={url}
            onChange={(e) => {
              setUrl(e.target.value);
              setError(null);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && url.trim()) {
                e.preventDefault();
                submit();
              } else if (e.key === "Escape") {
                e.preventDefault();
                onClose();
              }
            }}
            placeholder="https://…"
            autoComplete="off"
          />
        </label>
        {error ? (
          <p className="promedia-dialog-error" role="alert">
            {error}
          </p>
        ) : (
          <p className="promedia-dialog-hint">{t("link.hint")}</p>
        )}
        <div className="promedia-dialog-actions">
          <button type="button" className="btn" onClick={onClose}>
            {t("text.cancel")}
          </button>
          <button
            type="button"
            className="btn btn-primary"
            disabled={!url.trim()}
            onClick={submit}
          >
            {t("link.add")}
          </button>
        </div>
      </div>
    </div>
  );
}
