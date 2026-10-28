import { useEffect, useMemo, useState, type CSSProperties } from "react";
import { FittedTextStage } from "@promedia/item-types";
import type {
  MediaClock,
  SubtitleAgentState,
  SubtitleFontSize,
  TransitionEffect,
} from "@promedia/shared";
import {
  cueAtPositionWithLeadMs,
  DEFAULT_TRANSITION_EFFECT,
  effectiveMediaClockPositionMs,
  emptySubtitleAgentState,
  isSubtitleFontSize,
  LYRIC_OVERLAY_MAX_LINES,
  LYRIC_OVERLAY_PREFERRED_LINES,
  LYRIC_PHRASE_SOFT_WRAP_CHARS,
  normalizeSubtitleLeadMs,
  softWrapProjectionLine,
  TRANSITION_DURATION_MS,
  transitionCssId,
} from "@promedia/shared";

export type SubtitleOverlayProps = {
  agent: SubtitleAgentState | null | undefined;
  /** Hub mediaClock when the shown item is live; drives sync Preview↔Output. */
  mediaClock: MediaClock | null | undefined;
  /**
   * When Preview is not projecting, pass the local <video> currentTime in ms
   * so captions still track scrubbing before Go Live.
   */
  localPositionMs?: number | null;
  /** Only render cues when this matches agent.itemId (or agent has cues for it). */
  itemId?: string | null;
  /**
   * TEMA transition (FreeShow-style tip De22bDYtucC) — applied on each cue
   * remount so lyric / subtitle changes fade instead of hard-cutting.
   */
  transitionEffect?: TransitionEffect;
};

/**
 * SM–XL biases fit toward IG Reel Dd5Bi1IzCuQ / LED karaoke size.
 * 1–2 short phrases only: heightFrac + widthFrac + hardCap lean large so
 * type owns center stage.
 */
const SUBTITLE_FIT_BY_SIZE: Record<
  SubtitleFontSize,
  { heightFrac: number; widthFrac: number; hardCapPx: number }
> = {
  sm: { heightFrac: 0.55, widthFrac: 0.3, hardCapPx: 300 },
  md: { heightFrac: 0.66, widthFrac: 0.33, hardCapPx: 380 },
  lg: { heightFrac: 0.76, widthFrac: 0.35, hardCapPx: 460 },
  xl: { heightFrac: 0.84, widthFrac: 0.38, hardCapPx: 540 },
};

/**
 * Prefer 1–2 visual rows; hard max 2 (karaoke Reel). Cancionero exact
 * newlines win when present (sliced if over max). Soft-wrap widens the
 * char budget before spilling so FittedTextStage stays large.
 */
function lyricOverlayDisplayLines(raw: string): string[] {
  const exact = String(raw ?? "")
    .replace(/\r\n/g, "\n")
    .replace(/\r/g, "\n")
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);
  if (exact.length > 1) {
    return exact.slice(0, LYRIC_OVERLAY_MAX_LINES);
  }
  const single = (exact[0] ?? String(raw ?? "").trim()).trim();
  if (!single) {
    return ["\u00a0"];
  }

  const tryBudget = (budget: number) =>
    softWrapProjectionLine(single, Math.max(12, Math.trunc(budget)));

  let rows = tryBudget(LYRIC_PHRASE_SOFT_WRAP_CHARS);
  if (rows.length <= LYRIC_OVERLAY_PREFERRED_LINES) {
    return rows;
  }

  // Widen wrap so a long cue lands in preferred (2) rows when possible.
  rows = tryBudget(
    Math.max(
      LYRIC_PHRASE_SOFT_WRAP_CHARS,
      Math.ceil(single.length / LYRIC_OVERLAY_PREFERRED_LINES),
    ),
  );
  if (rows.length <= LYRIC_OVERLAY_PREFERRED_LINES) {
    return rows;
  }

  // Hard max 2 — force-fit before FittedTextStage hard-caps.
  rows = tryBudget(
    Math.max(12, Math.ceil(single.length / LYRIC_OVERLAY_MAX_LINES)),
  );
  return rows.slice(0, LYRIC_OVERLAY_MAX_LINES);
}

/**
 * Full-stage karaoke projection — IG Reel Dd5Bi1IzCuQ / LED-wall look:
 * huge centered type, 1–2 rows, fluid cue enter, soft black shadow only.
 * Timing prefers hub mediaClock; negative `leadMs` shows the cue early.
 * Assisted timingMode holds the first lyric through musical lead-in;
 * exact (cancionero / native) uses cue windows only. Manual ←/→ song
 * slides do not use this path.
 */
export function SubtitleOverlay({
  agent,
  mediaClock,
  localPositionMs,
  itemId,
  transitionEffect = DEFAULT_TRANSITION_EFFECT,
}: SubtitleOverlayProps) {
  const state = agent ?? emptySubtitleAgentState();
  const fontSize = isSubtitleFontSize(state.fontSize) ? state.fontSize : "md";
  const leadMs = normalizeSubtitleLeadMs(state.leadMs);
  const holdIntro = state.timingMode !== "exact";
  const fit = SUBTITLE_FIT_BY_SIZE[fontSize];
  const effect = transitionEffect || DEFAULT_TRANSITION_EFFECT;
  const [nowMs, setNowMs] = useState(() => Date.now());

  useEffect(() => {
    if (!state.enabled || state.status !== "ready" || !state.cues.length) {
      return;
    }
    if (mediaClock?.playing) {
      const id = window.setInterval(() => setNowMs(Date.now()), 200);
      return () => window.clearInterval(id);
    }
    setNowMs(Date.now());
    return undefined;
  }, [state.enabled, state.status, state.cues.length, mediaClock?.playing, mediaClock?.updatedAt]);

  const cue = useMemo(() => {
    if (!state.enabled) {
      return null;
    }
    if (itemId && state.itemId && state.itemId !== itemId) {
      return null;
    }
    if (state.status !== "ready" || !state.cues.length) {
      return null;
    }

    let positionMs: number;
    if (mediaClock && (!itemId || mediaClock.itemId === itemId)) {
      positionMs = effectiveMediaClockPositionMs(mediaClock, nowMs);
    } else if (
      typeof localPositionMs === "number" &&
      Number.isFinite(localPositionMs)
    ) {
      positionMs = localPositionMs;
    } else if (mediaClock) {
      positionMs = effectiveMediaClockPositionMs(mediaClock, nowMs);
    } else {
      return null;
    }

    return cueAtPositionWithLeadMs(state.cues, positionMs, leadMs, {
      holdIntro,
    });
  }, [
    state.enabled,
    state.itemId,
    state.status,
    state.cues,
    itemId,
    mediaClock,
    localPositionMs,
    nowMs,
    leadMs,
    holdIntro,
  ]);

  const displayLines = useMemo(() => {
    if (!cue?.text) {
      return null;
    }
    // Prefer exact cue newlines (cancionero). Soft-wrap only when a single
    // source line has no breaks — prefer 1–2 rows, hard max 2.
    return lyricOverlayDisplayLines(cue.text);
  }, [cue?.text]);

  if (!cue?.text || !displayLines) {
    return null;
  }

  const cueMotionKey = `${cue.id}-${cue.startMs}`;
  const txClass =
    effect !== "none"
      ? ` pm-tx pm-tx-${transitionCssId(effect)}`
      : " subtitle-overlay-cue--reel";
  const cueStyle =
    effect !== "none"
      ? ({ "--pm-tx-ms": `${TRANSITION_DURATION_MS}ms` } as CSSProperties)
      : undefined;

  return (
    <div
      className="subtitle-overlay subtitle-overlay--reel"
      data-size={fontSize}
      data-motion="reel"
      data-tx={effect}
      data-cue={cueMotionKey}
      aria-live="polite"
    >
      {/* Key remounts so TEMA / Reel enter runs on each lyric cue change. */}
      <div
        key={cueMotionKey}
        className={`subtitle-overlay-cue${txClass}`}
        style={cueStyle}
      >
        <FittedTextStage
          lines={displayLines}
          lineKey={`${cueMotionKey}-${fontSize}`}
          className="subtitle-overlay-fit text-output"
          lineClassName="text-output-line"
          heightFrac={fit.heightFrac}
          widthFrac={fit.widthFrac}
          hardCapPx={fit.hardCapPx}
          maxLines={LYRIC_OVERLAY_MAX_LINES}
        />
      </div>
    </div>
  );
}
