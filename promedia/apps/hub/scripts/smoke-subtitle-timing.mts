/**
 * Smoke: exact (native) vs assisted intro-hold + Adelante + cold-start snap
 * + player→hub mediaClock authority (YouTube / Guion Audio cue sync).
 * Usage: pnpm --filter @promedia/hub exec tsx scripts/smoke-subtitle-timing.mts
 */
import {
  cueAtPositionWithLeadMs,
  DEFAULT_SUBTITLE_LEAD_MS,
  CANCIONERO_DEFAULT_LEAD_MS,
  effectiveMediaClockPositionMs,
  type MediaClock,
  type SubtitleCue,
} from "@promedia/shared";
import {
  shouldCorrectSeek,
  shouldReportMediaClockPosition,
  shouldSnapMediaClockToStart,
} from "@promedia/item-types";

function assert(cond: unknown, msg: string): asserts cond {
  if (!cond) throw new Error(msg);
}

const cues: SubtitleCue[] = [
  { id: "c0", startMs: 12_000, endMs: 15_000, text: "Primera letra" },
  { id: "c1", startMs: 15_000, endMs: 18_000, text: "Segunda letra" },
];

function main(): void {
  assert(DEFAULT_SUBTITLE_LEAD_MS === 0, "native default lead is 0");
  assert(CANCIONERO_DEFAULT_LEAD_MS === -1000, "cancionero default lead −1s");

  // Native / exact: blank until official startMs (matches YouTube lyric video).
  assert(
    cueAtPositionWithLeadMs(cues, 0, 0, { holdIntro: false }) === null,
    "exact: no intro-hold at t=0",
  );
  assert(
    cueAtPositionWithLeadMs(cues, 12_500, 0, { holdIntro: false })?.text ===
      "Primera letra",
    "exact: first lyric at official window",
  );

  // Cancionero exact + −1s: line appears 1s before voice onset startMs.
  assert(
    cueAtPositionWithLeadMs(cues, 11_000, -1000, { holdIntro: false })?.text ===
      "Primera letra",
    "cancionero −1s shows first line at onset−1s",
  );
  assert(
    cueAtPositionWithLeadMs(cues, 10_500, -1000, { holdIntro: false }) === null,
    "cancionero −1s still blank before onset−1s",
  );

  // Assisted ASR: at theme start, first lyric visible (cancionero hold).
  const atStart = cueAtPositionWithLeadMs(cues, 0, 0, { holdIntro: true });
  assert(atStart?.text === "Primera letra", `intro hold @0: ${atStart?.text}`);

  // Adelante −2s: still hold first line at t=0; natural window opens earlier.
  const withLead = cueAtPositionWithLeadMs(cues, 0, -2000, { holdIntro: true });
  assert(withLead?.text === "Primera letra", "intro hold with Adelante −2");

  // Inside first cue window (no lead).
  const midFirst = cueAtPositionWithLeadMs(cues, 13_000, 0);
  assert(midFirst?.text === "Primera letra", "mid first cue");

  // After first cue ends, before second — gap stays blank (no fill-forward).
  const gap = cueAtPositionWithLeadMs(cues, 15_500, 0);
  // 15500 is inside c1 (15000–18000)
  assert(gap?.text === "Segunda letra", `second cue: ${gap?.text}`);
  const trueGap = cueAtPositionWithLeadMs(
    [
      cues[0]!,
      { id: "c1", startMs: 16_000, endMs: 18_000, text: "Segunda letra" },
    ],
    15_500,
    0,
  );
  assert(trueGap === null, "gap after first stays blank");

  // Adelante −2 advances into second cue at position 13s → lookup 15s.
  const earlySecond = cueAtPositionWithLeadMs(cues, 13_000, -2000);
  assert(
    earlySecond?.text === "Segunda letra",
    `Adelante into second: ${earlySecond?.text}`,
  );

  // Cold-start snap: hub extrapolated from explicit 0 while player still at head.
  const now = Date.now();
  const clock: MediaClock = {
    itemId: "item1",
    playing: true,
    positionMs: 0,
    rate: 1,
    updatedAt: new Date(now - 8_000).toISOString(),
  };
  assert(
    effectiveMediaClockPositionMs(clock, now) >= 7_500,
    "clock extrapolated",
  );
  assert(
    shouldSnapMediaClockToStart(0, clock, now) === true,
    "snap when player at 0 and clock ran ahead",
  );
  assert(
    shouldSnapMediaClockToStart(0, { ...clock, positionMs: 60_000 }, now) ===
      false,
    "no snap after operator scrub",
  );
  assert(
    shouldSnapMediaClockToStart(20_000, clock, now) === false,
    "no snap when player already past head",
  );

  // Player authority: do NOT chase wall-clock extrapolation (stale hub).
  const stalePlaying: MediaClock = {
    itemId: "item1",
    playing: true,
    positionMs: 5_000,
    rate: 1,
    updatedAt: new Date(now - 3_000).toISOString(),
  };
  assert(
    effectiveMediaClockPositionMs(stalePlaying, now) >= 7_500,
    "stale hub extrapolates past player",
  );
  const noChase = shouldCorrectSeek(5_000, stalePlaying, now);
  assert(noChase.seek === false, "do not seek player to extrapolated clock");

  // Fresh operator scrub: seek to explicit hub position.
  const scrub: MediaClock = {
    itemId: "item1",
    playing: true,
    positionMs: 30_000,
    rate: 1,
    updatedAt: new Date(now).toISOString(),
  };
  const scrubSeek = shouldCorrectSeek(5_000, scrub, now);
  assert(
    scrubSeek.seek === true && scrubSeek.targetMs === 30_000,
    "seek on fresh operator scrub",
  );

  // Player→hub report: while playing near last position, heartbeat when aged.
  const agedReport: MediaClock = {
    itemId: "item1",
    playing: true,
    positionMs: 5_000,
    rate: 1,
    updatedAt: new Date(now - 500).toISOString(),
  };
  assert(
    shouldReportMediaClockPosition(5_020, agedReport, now) === true,
    "heartbeat report refreshes updatedAt while stalled/buffering",
  );
  assert(
    shouldReportMediaClockPosition(5_500, agedReport, now) === true,
    "report when player advanced past epsilon",
  );
  const freshMatch: MediaClock = {
    ...agedReport,
    updatedAt: new Date(now - 50).toISOString(),
  };
  assert(
    shouldReportMediaClockPosition(5_020, freshMatch, now) === false,
    "skip redundant report right after hub write",
  );
  assert(
    shouldReportMediaClockPosition(5_000, scrub, now) === false,
    "do not report over a fresh scrub — seek first",
  );

  console.log("SMOKE_OK smoke-subtitle-timing");
}

main();
