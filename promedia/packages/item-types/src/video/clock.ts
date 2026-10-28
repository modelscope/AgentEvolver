import type { MediaClock } from "@promedia/shared";
import { effectiveMediaClockPositionMs } from "@promedia/shared";

export const MEDIA_CLOCK_DRIFT_MS = 400;

/**
 * Hub mediaClock extrapolates from Go Live even while Output is still mounting
 * the YouTube iframe or autoplay is blocked. If the player is still at the
 * head and the last *explicit* seek was 0 (no operator scrub), snap back to 0
 * so the audible theme start lines up with cue windows — not mid-track.
 */
export const MEDIA_CLOCK_COLD_START_MS = 1500;

/** How often Output reports real player time into hub mediaClock. */
export const MEDIA_CLOCK_REPORT_INTERVAL_MS = 250;

/**
 * Minimum |local − hub| before a position report is worth sending.
 * Heartbeats still refresh `updatedAt` when the player is stalled near the
 * last reported position (see shouldReportMediaClockPosition).
 */
export const MEDIA_CLOCK_REPORT_EPSILON_MS = 80;

/**
 * Fresh hub writes (Go Live, scrub, pause/play with position) stay authoritative
 * for this long so Output seeks to them. After that, the player is the source
 * of truth and reports real time — we must not chase wall-clock extrapolation.
 */
export const MEDIA_CLOCK_SEEK_AUTHORITY_MS = 450;

/**
 * Heartbeat: re-report even when position is unchanged so `updatedAt` stays
 * fresh. Without this, effectiveMediaClockPositionMs races ahead while YouTube
 * / audio is buffering and cancionero cues desync from the sung words.
 */
export const MEDIA_CLOCK_REPORT_HEARTBEAT_MS = 400;

/**
 * Seek only to the hub's *explicit* position while that write is fresh
 * (operator scrub / Go Live). Do **not** chase `effectiveMediaClockPositionMs`
 * — that wall-clock extrapolation was seeking YouTube/audio past buffers and
 * driving SubtitleOverlay off the audible timeline.
 */
export function shouldCorrectSeek(
  localPositionMs: number,
  clock: MediaClock,
  nowMs: number = Date.now(),
): { seek: boolean; targetMs: number } {
  const targetMs = Math.max(0, clock.positionMs);
  const drift = Math.abs(localPositionMs - targetMs);
  if (drift <= MEDIA_CLOCK_DRIFT_MS) {
    return { seek: false, targetMs };
  }
  const updated = Date.parse(clock.updatedAt);
  const age = Number.isFinite(updated)
    ? nowMs - updated
    : Number.POSITIVE_INFINITY;
  if (age <= MEDIA_CLOCK_SEEK_AUTHORITY_MS) {
    return { seek: true, targetMs };
  }
  // Stale hub clock: player owns the timeline; reporting will refresh the hub.
  return { seek: false, targetMs: localPositionMs };
}

/**
 * True when Output should push real player time into hub mediaClock so
 * SubtitleOverlay (cancionero −1s lead) tracks the sung words, not Go Live
 * wall-clock.
 */
export function shouldReportMediaClockPosition(
  localPositionMs: number,
  clock: MediaClock,
  nowMs: number = Date.now(),
): boolean {
  if (!clock.playing) {
    return false;
  }
  const local = Math.max(0, localPositionMs);
  const updated = Date.parse(clock.updatedAt);
  const age = Number.isFinite(updated)
    ? nowMs - updated
    : Number.POSITIVE_INFINITY;

  // Fresh operator scrub / Go Live — let shouldCorrectSeek apply first.
  if (
    age <= MEDIA_CLOCK_SEEK_AUTHORITY_MS &&
    Math.abs(local - clock.positionMs) > MEDIA_CLOCK_DRIFT_MS
  ) {
    return false;
  }

  if (Math.abs(local - clock.positionMs) > MEDIA_CLOCK_REPORT_EPSILON_MS) {
    return true;
  }
  // Stall / buffer heartbeat: keep updatedAt pinned to real media time.
  if (age >= MEDIA_CLOCK_REPORT_HEARTBEAT_MS) {
    return true;
  }
  return false;
}

/**
 * True when Output audio is about to start at t≈0 but the hub clock already
 * ran ahead during handoff / autoplay block (explicit position still 0).
 */
export function shouldSnapMediaClockToStart(
  localPositionMs: number,
  clock: MediaClock,
  nowMs: number = Date.now(),
): boolean {
  if (!clock.playing) {
    return false;
  }
  // Operator scrubbed / seeked away from the head — honor that position.
  if (clock.positionMs > MEDIA_CLOCK_COLD_START_MS) {
    return false;
  }
  if (localPositionMs > MEDIA_CLOCK_COLD_START_MS) {
    return false;
  }
  return (
    effectiveMediaClockPositionMs(clock, nowMs) > MEDIA_CLOCK_COLD_START_MS
  );
}
