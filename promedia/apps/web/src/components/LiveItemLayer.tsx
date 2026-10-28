import type { CSSProperties, ReactNode } from "react";
import {
  TRANSITION_DURATION_MS,
  transitionCssId,
  type TransitionEffect,
} from "@promedia/shared";

export type LiveItemLayerProps = {
  effect: TransitionEffect;
  children: ReactNode;
  /** Hide from AT when scene overlays cover the stage. */
  ariaHidden?: boolean;
  /** Optional `data-content` for stage CSS (e.g. video plane). */
  dataContent?: string;
  /** Extra classes (e.g. `pm-stage-video-plane` on Control En vivo). */
  className?: string;
  /**
   * Sibling of the transition inner (not animated as content) —
   * e.g. StageDotsOverlay on the video plane under captions.
   */
  planeOverlay?: ReactNode;
};

/**
 * Single-layer live item wrapper so CSS enter transitions can run on remount.
 * Parent must set `key={liveItemId}` so the layer remounts on Go Live / next item.
 * Keeps YouTube to one iframe (no dual-embed crossfade).
 */
export function LiveItemLayer({
  effect,
  children,
  ariaHidden,
  dataContent,
  className,
  planeOverlay,
}: LiveItemLayerProps) {
  const txClass =
    effect !== "none" ? ` pm-tx pm-tx-${transitionCssId(effect)}` : "";
  const extra = className ? ` ${className}` : "";

  return (
    <div
      className={`output-item-layer${extra}${txClass}`}
      data-content={dataContent}
      style={
        {
          "--pm-tx-ms": `${TRANSITION_DURATION_MS}ms`,
        } as CSSProperties
      }
      aria-hidden={ariaHidden}
    >
      <div className="output-item-transition-inner">{children}</div>
      {planeOverlay}
    </div>
  );
}
