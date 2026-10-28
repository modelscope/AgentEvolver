/** Shared stage art for Preview / En vivo / Output when projecting audio. */
export function AudioStage({ title }: { title?: string }) {
  return (
    <div className="audio-stage" aria-hidden={!title}>
      <svg
        className="audio-stage-icon"
        viewBox="0 0 64 64"
        width="72"
        height="72"
        aria-hidden
      >
        <circle
          cx="32"
          cy="32"
          r="28"
          fill="none"
          stroke="currentColor"
          strokeWidth="2.5"
          opacity="0.45"
        />
        <path
          d="M28 18v22.5a7 7 0 1 1-4-6.3V24.5l18-4.2V38a7 7 0 1 1-4-6.3V18.8L28 18Z"
          fill="currentColor"
        />
      </svg>
      {title ? <p className="audio-stage-title">{title}</p> : null}
    </div>
  );
}
