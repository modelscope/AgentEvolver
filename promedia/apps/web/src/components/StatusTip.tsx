import { useEffect, useId, useRef, useState } from "react";
import { IconInfo, IconWarn } from "./UiIcons";

export type StatusTipTone = "info" | "warn" | "error";

type StatusTipProps = {
  /** Full message shown only in the tooltip. */
  message: string;
  /** Short accessible name for the icon button. */
  label: string;
  tone?: StatusTipTone;
  className?: string;
};

/**
 * Compact status/warning glyph: message appears on hover, focus, or click
 * (touch) so tip text does not consume Control layout space.
 */
export function StatusTip({
  message,
  label,
  tone = "warn",
  className,
}: StatusTipProps) {
  const [pinned, setPinned] = useState(false);
  const tipId = useId();
  const rootRef = useRef<HTMLSpanElement>(null);
  const Icon = tone === "info" ? IconInfo : IconWarn;

  useEffect(() => {
    if (!pinned) {
      return;
    }
    const onPointerDown = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) {
        setPinned(false);
      }
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setPinned(false);
      }
    };
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [pinned]);

  return (
    <span
      ref={rootRef}
      className={[
        "status-tip",
        `status-tip-${tone}`,
        pinned ? "is-open" : "",
        className ?? "",
      ]
        .filter(Boolean)
        .join(" ")}
    >
      <button
        type="button"
        className="status-tip-trigger"
        aria-label={label}
        aria-expanded={pinned}
        aria-describedby={tipId}
        title={message}
        onClick={() => setPinned((open) => !open)}
      >
        <Icon />
      </button>
      <span id={tipId} role="tooltip" className="status-tip-bubble">
        {message}
      </span>
    </span>
  );
}
