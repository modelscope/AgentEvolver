import { useState } from "react";
import { useLocale } from "../locale/LocaleContext";
import type { SceneMode } from "@promedia/shared";

const DEFAULT_LOGO_SRC = "/logo-default.svg";

export type SceneOverlayProps = {
  mode: SceneMode;
};

/**
 * Full-stage overlay for logo / black / clear.
 * Item renderer stays mounted underneath so mediaClock (and audio) continue (Q1).
 */
export function SceneOverlay({ mode }: SceneOverlayProps) {
  const { t } = useLocale();

  const [logoFailed, setLogoFailed] = useState(false);

  if (mode === "normal") {
    return null;
  }

  if (mode === "black") {
    return (
      <div
        className="scene-overlay scene-overlay-black"
        aria-label={t("scene.black")}
      />
    );
  }

  if (mode === "clear") {
    return (
      <div
        className="scene-overlay scene-overlay-clear"
        aria-label={t("scene.clear")}
      >
        <p className="scene-clear-label">{t("output.waiting")}</p>
      </div>
    );
  }

  return (
    <div
      className="scene-overlay scene-overlay-logo"
      aria-label={t("scene.logo")}
    >
      {logoFailed ? (
        <p className="scene-logo-fallback">{t("scene.logoFallback")}</p>
      ) : (
        <img
          className="scene-logo-img"
          src={DEFAULT_LOGO_SRC}
          alt="ProMedia"
          onError={() => setLogoFailed(true)}
        />
      )}
    </div>
  );
}
