import type { ReactNode } from "react";

export const SKIP_MS = 10_000;

export function formatMediaMs(ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${m}:${r.toString().padStart(2, "0")}`;
}

function IconPlay() {
  return (
    <svg className="preview-tb-icon" viewBox="0 0 24 24" width="16" height="16" aria-hidden>
      <path d="M8 5.5v13l11-6.5L8 5.5Z" fill="currentColor" />
    </svg>
  );
}

function IconPause() {
  return (
    <svg className="preview-tb-icon" viewBox="0 0 24 24" width="16" height="16" aria-hidden>
      <rect x="6" y="5" width="4" height="14" rx="1" fill="currentColor" />
      <rect x="14" y="5" width="4" height="14" rx="1" fill="currentColor" />
    </svg>
  );
}

function IconSkipBack() {
  return (
    <svg className="preview-tb-icon" viewBox="0 0 24 24" width="16" height="16" aria-hidden>
      <path d="M11 12 18 6.5v11L11 12Z" fill="currentColor" />
      <path d="M6 12 13 6.5v11L6 12Z" fill="currentColor" />
      <rect x="4" y="6" width="2.2" height="12" rx="0.6" fill="currentColor" />
    </svg>
  );
}

function IconSkipForward() {
  return (
    <svg className="preview-tb-icon" viewBox="0 0 24 24" width="16" height="16" aria-hidden>
      <path d="M13 12 6 6.5v11L13 12Z" fill="currentColor" />
      <path d="M18 12 11 6.5v11L18 12Z" fill="currentColor" />
      <rect x="17.8" y="6" width="2.2" height="12" rx="0.6" fill="currentColor" />
    </svg>
  );
}

function IconToStart() {
  return (
    <svg className="preview-tb-icon" viewBox="0 0 24 24" width="16" height="16" aria-hidden>
      <rect x="4" y="6" width="2.2" height="12" rx="0.6" fill="currentColor" />
      <path d="M20 6.5v11L10 12l10-5.5Z" fill="currentColor" />
    </svg>
  );
}

function IconToEnd() {
  return (
    <svg className="preview-tb-icon" viewBox="0 0 24 24" width="16" height="16" aria-hidden>
      <path d="M4 6.5v11L14 12 4 6.5Z" fill="currentColor" />
      <rect x="17.8" y="6" width="2.2" height="12" rx="0.6" fill="currentColor" />
    </svg>
  );
}

function IconVolumeOn() {
  return (
    <svg className="preview-tb-icon" viewBox="0 0 24 24" width="16" height="16" aria-hidden>
      <path d="M4 9.5h3.2L12 6v12l-4.8-3.5H4V9.5Z" fill="currentColor" />
      <path
        d="M15.2 9.2a3.2 3.2 0 0 1 0 5.6M17.5 7a6 6 0 0 1 0 10"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        fill="none"
      />
    </svg>
  );
}

function IconVolumeOff() {
  return (
    <svg className="preview-tb-icon" viewBox="0 0 24 24" width="16" height="16" aria-hidden>
      <path d="M4 9.5h3.2L12 6v12l-4.8-3.5H4V9.5Z" fill="currentColor" />
      <path
        d="m15 9.5 5 5M20 9.5l-5 5"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
        fill="none"
      />
    </svg>
  );
}

export type VideoTransportBarProps = {
  playing: boolean;
  positionMs: number;
  durationMs: number;
  scrubbing: boolean;
  scrubMs: number;
  disabled?: boolean;
  /** When false, hide mute (live clock transport has no local mute). */
  showMute?: boolean;
  muted?: boolean;
  labels?: {
    play?: string;
    pause?: string;
    skipBack?: string;
    skipForward?: string;
    toStart?: string;
    toEnd?: string;
    timeline?: string;
    mute?: string;
    unmute?: string;
    toolbar?: string;
  };
  remainingLabel?: ReactNode;
  onTogglePlay: () => void;
  onSkip: (deltaMs: number) => void;
  onSeekTo: (ms: number) => void;
  onToggleMute?: () => void;
  onScrubChange: (ms: number) => void;
  onScrubCommit: () => void;
  onScrubStart: () => void;
};

/**
 * Shared Preview / En vivo transport: play/pause, ±10s, start/end, scrubber.
 */
export function VideoTransportBar({
  playing,
  positionMs,
  durationMs,
  scrubbing,
  scrubMs,
  disabled = false,
  showMute = false,
  muted = false,
  labels = {},
  remainingLabel,
  onTogglePlay,
  onSkip,
  onSeekTo,
  onToggleMute,
  onScrubChange,
  onScrubCommit,
  onScrubStart,
}: VideoTransportBarProps) {
  const displayMs = scrubbing ? scrubMs : positionMs;
  const maxMs = Math.max(durationMs > 0 ? durationMs : Math.max(displayMs, 1), 1);
  const remainingMs =
    durationMs > 0 ? Math.max(0, durationMs - displayMs) : -1;
  const canScrub = !disabled && (durationMs > 0 || scrubbing);

  return (
    <div
      className="preview-toolbar preview-toolbar-video"
      role="toolbar"
      aria-label={labels.toolbar ?? "Video transport"}
    >
      <div className="preview-toolbar-row">
        <button
          type="button"
          className="btn preview-tb-btn"
          disabled={disabled}
          title={playing ? labels.pause ?? "Pause" : labels.play ?? "Play"}
          aria-label={playing ? labels.pause ?? "Pause" : labels.play ?? "Play"}
          onClick={onTogglePlay}
        >
          {playing ? <IconPause /> : <IconPlay />}
        </button>
        <button
          type="button"
          className="btn preview-tb-btn"
          disabled={disabled}
          title={labels.toStart ?? "Start"}
          aria-label={labels.toStart ?? "Start"}
          onClick={() => onSeekTo(0)}
        >
          <IconToStart />
        </button>
        <button
          type="button"
          className="btn preview-tb-btn"
          disabled={disabled}
          title={labels.skipBack ?? "−10s"}
          aria-label={labels.skipBack ?? "−10s"}
          onClick={() => onSkip(-SKIP_MS)}
        >
          <IconSkipBack />
        </button>
        <button
          type="button"
          className="btn preview-tb-btn"
          disabled={disabled}
          title={labels.skipForward ?? "+10s"}
          aria-label={labels.skipForward ?? "+10s"}
          onClick={() => onSkip(SKIP_MS)}
        >
          <IconSkipForward />
        </button>
        <button
          type="button"
          className="btn preview-tb-btn"
          disabled={disabled || durationMs <= 0}
          title={labels.toEnd ?? "End"}
          aria-label={labels.toEnd ?? "End"}
          onClick={() => onSeekTo(Math.max(0, durationMs - 250))}
        >
          <IconToEnd />
        </button>
        <span className="video-time" title="Elapsed">
          {formatMediaMs(displayMs)}
        </span>
        <input
          type="range"
          className="preview-scrubber"
          min={0}
          max={maxMs}
          step={100}
          value={Math.min(Math.max(0, displayMs), maxMs)}
          disabled={!canScrub}
          aria-label={labels.timeline ?? "Timeline"}
          onPointerDown={onScrubStart}
          onChange={(e) => {
            onScrubStart();
            onScrubChange(Number(e.target.value));
          }}
          onPointerUp={onScrubCommit}
          onKeyUp={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              onScrubCommit();
            }
          }}
        />
        <span
          className="video-time video-time-remaining"
          title="Remaining"
        >
          {remainingLabel ??
            (remainingMs >= 0
              ? `−${formatMediaMs(remainingMs)}`
              : "−−:−−")}
        </span>
        {showMute && onToggleMute ? (
          <button
            type="button"
            className="btn preview-tb-btn"
            disabled={disabled}
            title={
              muted
                ? labels.unmute ?? "Unmute"
                : labels.mute ?? "Mute"
            }
            aria-label={
              muted
                ? labels.unmute ?? "Unmute"
                : labels.mute ?? "Mute"
            }
            onClick={onToggleMute}
          >
            {muted ? <IconVolumeOff /> : <IconVolumeOn />}
          </button>
        ) : null}
      </div>
    </div>
  );
}
