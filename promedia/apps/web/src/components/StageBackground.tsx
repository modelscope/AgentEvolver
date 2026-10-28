import {
  DEFAULT_THEME_BACKGROUND,
  normalizeThemeBackground,
  themeBackgroundPackSrc,
  type ThemeBackground,
} from "@promedia/shared";

export type StageBackgroundProps = {
  background?: ThemeBackground | null;
  /** Authenticated URL for uploaded session asset. */
  assetSrc?: string;
  /**
   * When false, skip the dots tile here (caller paints {@link StageDotsOverlay}
   * on the video plane instead so captions stay above the pattern).
   * Default true — dots sit inside `.pm-stage-bg` (under playlist content).
   */
  showDots?: boolean;
};

/**
 * CEOSA-style dots tile. Render on the video plane (inside the item/video
 * stacking context) so the pattern darkens media but cannot paint over
 * `.subtitle-overlay` siblings.
 */
export function StageDotsOverlay() {
  return <div className="pm-stage-bg-dots" aria-hidden />;
}

/**
 * Full-bleed stage backdrop for Output / En vivo / Vista previa.
 * Solid mode relies on parent `.pm-theme` `--pm-bg`; media uses pack video or upload.
 *
 * Dot pattern for non-video content stays inside `.pm-stage-bg` (z-index −1).
 * For playlist video/YouTube, callers mount {@link StageDotsOverlay} on the
 * video plane so the tile sits above the media and below captions.
 */
export function StageBackground({
  background,
  assetSrc,
  showDots = true,
}: StageBackgroundProps) {
  const bg = normalizeThemeBackground(background ?? DEFAULT_THEME_BACKGROUND);
  const showMedia = bg.mode === "media";
  const showPack = showMedia && Boolean(bg.packId);
  const showAsset = showMedia && Boolean(bg.assetId && assetSrc);
  const dotsInBg = showDots && bg.dotsOverlay === true;

  if (!showPack && !showAsset && !dotsInBg) {
    return null;
  }

  return (
    <div className="pm-stage-bg" aria-hidden>
      {showPack && bg.packId ? (
        <div
          key={`pack:${bg.packId}`}
          className={`pm-stage-bg-pack pm-stage-bg-${bg.packId} pm-stage-bg-enter`}
        >
          {/* CSS layers remain as fallback under the looping pack video. */}
          <div className="pm-stage-bg-layer pm-stage-bg-layer-a" />
          <div className="pm-stage-bg-layer pm-stage-bg-layer-b" />
          <video
            className="pm-stage-bg-media-el"
            autoPlay
            loop
            muted
            playsInline
          >
            <source
              src={themeBackgroundPackSrc(bg.packId, "webm")}
              type="video/webm"
            />
            <source
              src={themeBackgroundPackSrc(bg.packId, "mp4")}
              type="video/mp4"
            />
          </video>
        </div>
      ) : null}
      {showAsset && assetSrc ? (
        <div
          key={`asset:${bg.assetId ?? assetSrc}`}
          className="pm-stage-bg-media pm-stage-bg-enter"
        >
          {bg.mediaKind === "image" ? (
            <img className="pm-stage-bg-media-el" src={assetSrc} alt="" />
          ) : (
            <video
              className="pm-stage-bg-media-el"
              src={assetSrc}
              autoPlay
              loop
              muted
              playsInline
            />
          )}
        </div>
      ) : null}
      {dotsInBg ? <StageDotsOverlay /> : null}
    </div>
  );
}
