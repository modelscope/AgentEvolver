import { formatMediaMs } from "./VideoTransportBar.js";

export type VideoStageTimeOverlayProps = {
  positionMs: number;
  durationMs: number;
  remainingLabel?: string;
  totalLabel?: string;
};

/**
 * Restante + total as on-media overlays (Vista previa / En vivo stages).
 */
export function VideoStageTimeOverlay({
  positionMs,
  durationMs,
  remainingLabel = "Restante",
  totalLabel = "Total",
}: VideoStageTimeOverlayProps) {
  const remainingMs =
    durationMs > 0 ? Math.max(0, durationMs - Math.max(0, positionMs)) : -1;

  return (
    <div className="video-stage-time-overlay" aria-hidden="true">
      <div className="video-stage-time-block">
        <span className="video-stage-time-label">{remainingLabel}</span>
        <span className="video-stage-time-value">
          {remainingMs < 0 ? "−−:−−" : formatMediaMs(remainingMs)}
        </span>
      </div>
      <div className="video-stage-time-block video-stage-time-total">
        <span className="video-stage-time-label">{totalLabel}</span>
        <span className="video-stage-time-value">
          {durationMs > 0 ? formatMediaMs(durationMs) : "−−:−−"}
        </span>
      </div>
    </div>
  );
}
