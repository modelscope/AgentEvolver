import { useLocale } from "../locale/LocaleContext";

export type AboutDialogProps = {
  open: boolean;
  onClose: () => void;
};

const CREDIT = "@2026 JulioAlvarez75";

export function AboutDialog({ open, onClose }: AboutDialogProps) {
  const { t } = useLocale();
  if (!open) {
    return null;
  }

  return (
    <div
      className="promedia-dialog-backdrop"
      role="presentation"
      onClick={onClose}
    >
      <div
        className="promedia-dialog about-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="about-dialog-title"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 id="about-dialog-title">{t("control.about")}</h2>
        <p className="about-credit">{CREDIT}</p>
        <p className="promedia-dialog-hint">{t("control.aboutBody")}</p>
        <div className="promedia-dialog-actions">
          <button type="button" className="btn btn-primary" onClick={onClose}>
            {t("control.close")}
          </button>
        </div>
      </div>
    </div>
  );
}
