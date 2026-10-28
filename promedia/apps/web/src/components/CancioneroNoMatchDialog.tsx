import { useLocale } from "../locale/LocaleContext";

export type CancioneroNoMatchDialogProps = {
  open: boolean;
  onClose: () => void;
  /** Optional media title shown in the body. */
  mediaTitle?: string | null;
  /** Hub hint: cancionero_no_match | cancionero_no_songs | cancionero_empty */
  hint?: string | null;
};

/**
 * Warning when Karaoke agent cannot find the Guion Audio / YouTube item
 * in Biblioteca → Canciones (option 2 — cancionero is source of truth).
 */
export function CancioneroNoMatchDialog({
  open,
  onClose,
  mediaTitle,
  hint,
}: CancioneroNoMatchDialogProps) {
  const { t } = useLocale();
  if (!open) {
    return null;
  }

  const bodyKey =
    hint === "cancionero_no_songs"
      ? "control.cancioneroNoSongsBody"
      : hint === "cancionero_empty"
        ? "control.cancioneroEmptyBody"
        : "control.cancioneroNoMatchBody";

  const titleLabel = mediaTitle?.trim();

  return (
    <div
      className="promedia-dialog-backdrop"
      role="presentation"
      onClick={onClose}
    >
      <div
        className="promedia-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="cancionero-no-match-title"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 id="cancionero-no-match-title">
          {t("control.cancioneroNoMatchTitle")}
        </h2>
        {titleLabel ? (
          <p className="promedia-dialog-hint" role="status">
            <strong>{titleLabel}</strong>
          </p>
        ) : null}
        <p className="promedia-dialog-error" role="alert">
          {t(bodyKey)}
        </p>
        <div className="promedia-dialog-actions">
          <button type="button" className="btn btn-primary" onClick={onClose}>
            {t("control.close")}
          </button>
        </div>
      </div>
    </div>
  );
}
