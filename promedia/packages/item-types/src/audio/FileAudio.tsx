import { useEffect, useRef, useState } from "react";
import type { MediaClock } from "@promedia/shared";
import {
  MEDIA_CLOCK_COLD_START_MS,
  MEDIA_CLOCK_REPORT_INTERVAL_MS,
  shouldCorrectSeek,
  shouldReportMediaClockPosition,
  shouldSnapMediaClockToStart,
} from "../video/clock.js";

export type FileAudioProps = {
  src: string;
  mediaClock?: MediaClock | null;
  /** Local preview — ignore mediaClock sync. */
  preview?: boolean;
  loop?: boolean;
  onBlocked?: () => void;
  /** Output: reset hub clock to 0 when audible playback finally starts after a block. */
  onMediaClockSet?: (patch: { playing: boolean; positionMs?: number }) => void;
  onDurationMs?: (ms: number) => void;
};

/**
 * Hidden &lt;audio&gt; synced to mediaClock (same drift rules as FileVideo).
 * Visual chrome lives in Preview / stage wrappers.
 */
export function FileAudio({
  src,
  mediaClock,
  preview = false,
  loop = false,
  onBlocked,
  onMediaClockSet,
  onDurationMs,
}: FileAudioProps) {
  const ref = useRef<HTMLAudioElement>(null);
  const [blocked, setBlocked] = useState(false);
  const mediaClockRef = useRef(mediaClock);
  mediaClockRef.current = mediaClock;
  const onMediaClockSetRef = useRef(onMediaClockSet);
  onMediaClockSetRef.current = onMediaClockSet;
  const coldSnapDoneRef = useRef(false);

  useEffect(() => {
    if (!mediaClock) {
      coldSnapDoneRef.current = false;
      return;
    }
    if (mediaClock.positionMs > MEDIA_CLOCK_COLD_START_MS) {
      coldSnapDoneRef.current = true;
    }
    if (!mediaClock.playing) {
      coldSnapDoneRef.current = false;
    }
  }, [mediaClock?.itemId, mediaClock?.positionMs, mediaClock?.playing]);

  useEffect(() => {
    const el = ref.current;
    if (!el) {
      return;
    }
    const report = () => {
      const ms = (el.duration || 0) * 1000;
      if (ms > 0) {
        onDurationMs?.(ms);
      }
    };
    el.addEventListener("loadedmetadata", report);
    el.addEventListener("durationchange", report);
    report();
    return () => {
      el.removeEventListener("loadedmetadata", report);
      el.removeEventListener("durationchange", report);
    };
  }, [src, onDurationMs]);

  useEffect(() => {
    const el = ref.current;
    if (!el || preview || !mediaClock) {
      return;
    }
    el.loop = loop;
    if (mediaClock.playing) {
      const p = el.play();
      if (p && typeof p.then === "function") {
        p.catch(() => {
          setBlocked(true);
          onBlocked?.();
        });
      }
    } else {
      el.pause();
    }
  }, [preview, loop, onBlocked, mediaClock?.playing, mediaClock?.itemId]);

  /** Operator scrub / Go Live / cold-snap seek (not player→hub heartbeats). */
  useEffect(() => {
    const el = ref.current;
    if (!el || preview || !mediaClock) {
      return;
    }
    const localMs = el.currentTime * 1000;
    if (
      !coldSnapDoneRef.current &&
      shouldSnapMediaClockToStart(localMs, mediaClock) &&
      onMediaClockSetRef.current
    ) {
      coldSnapDoneRef.current = true;
      onMediaClockSetRef.current({ playing: true, positionMs: 0 });
      el.currentTime = 0;
      return;
    }
    const { seek, targetMs } = shouldCorrectSeek(localMs, mediaClock);
    if (seek) {
      el.currentTime = targetMs / 1000;
    }
  }, [
    preview,
    mediaClock?.playing,
    mediaClock?.itemId,
    mediaClock?.positionMs,
    mediaClock?.updatedAt,
  ]);
  // Player → hub: Guion Audio currentTime → cancionero cue sync (−1s lead).
  useEffect(() => {
    if (preview || !mediaClock?.playing || !onMediaClockSet) {
      return;
    }
    const tick = () => {
      const el = ref.current;
      const clock = mediaClockRef.current;
      const report = onMediaClockSetRef.current;
      if (!el || !clock?.playing || !report) {
        return;
      }
      const localMs = el.currentTime * 1000;
      if (
        !coldSnapDoneRef.current &&
        shouldSnapMediaClockToStart(localMs, clock)
      ) {
        coldSnapDoneRef.current = true;
        report({ playing: true, positionMs: 0 });
        el.currentTime = 0;
        return;
      }
      if (shouldReportMediaClockPosition(localMs, clock)) {
        report({ playing: true, positionMs: Math.round(localMs) });
      }
    };
    tick();
    const id = window.setInterval(tick, MEDIA_CLOCK_REPORT_INTERVAL_MS);
    return () => window.clearInterval(id);
  }, [
    preview,
    mediaClock?.playing,
    mediaClock?.itemId,
    onMediaClockSet,
  ]);

  return (
    <>
      <audio
        ref={ref}
        src={src}
        className="audio-el"
        preload="metadata"
        loop={loop}
        autoPlay={false}
      />
      {blocked ? (
        <button
          type="button"
          className="btn btn-primary audio-unblock"
          onClick={() => {
            const el = ref.current;
            if (!el) {
              return;
            }
            const clock = mediaClockRef.current;
            if (
              clock &&
              !coldSnapDoneRef.current &&
              shouldSnapMediaClockToStart(el.currentTime * 1000, clock)
            ) {
              coldSnapDoneRef.current = true;
              onMediaClockSetRef.current?.({ playing: true, positionMs: 0 });
              el.currentTime = 0;
            } else if (clock) {
              const { seek, targetMs } = shouldCorrectSeek(
                el.currentTime * 1000,
                clock,
              );
              if (seek) {
                el.currentTime = targetMs / 1000;
              }
            }
            void el
              .play()
              .then(() => setBlocked(false))
              .catch(() => undefined);
          }}
        >
          Reproducir audio
        </button>
      ) : null}
    </>
  );
}