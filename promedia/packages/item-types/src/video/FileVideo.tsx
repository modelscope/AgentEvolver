import { useEffect, useRef, useState } from "react";
import type { MediaClock, VideoFit } from "@promedia/shared";
import {
  MEDIA_CLOCK_COLD_START_MS,
  MEDIA_CLOCK_REPORT_INTERVAL_MS,
  shouldCorrectSeek,
  shouldReportMediaClockPosition,
  shouldSnapMediaClockToStart,
} from "./clock.js";

export type FileVideoProps = {
  src: string;
  mediaClock?: MediaClock | null;
  preview?: boolean;
  loop?: boolean;
  muted?: boolean;
  fit?: VideoFit;
  onBlocked?: () => void;
  /** Output: reset hub clock to 0 when audible playback finally starts after a block. */
  onMediaClockSet?: (patch: { playing: boolean; positionMs?: number }) => void;
  onDurationMs?: (ms: number) => void;
};

export function FileVideo({
  src,
  mediaClock,
  preview = false,
  loop = false,
  muted = false,
  fit = "contain",
  onBlocked,
  onMediaClockSet,
  onDurationMs,
}: FileVideoProps) {
  const ref = useRef<HTMLVideoElement>(null);
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
    el.muted = muted;
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
  }, [
    preview,
    loop,
    muted,
    onBlocked,
    mediaClock?.playing,
    mediaClock?.itemId,
  ]);

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

  // Player → hub: real currentTime drives SubtitleOverlay (not wall-clock).
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
    <div className="video-file-wrap">
      <video
        ref={ref}
        src={src}
        className="video-el"
        style={{ objectFit: fit }}
        playsInline
        controls={preview}
        loop={loop}
        muted={muted}
        // Unmuted by default for live (Q2); browsers may still block autoplay
        autoPlay={false}
      />
      {blocked ? (
        <button
          type="button"
          className="btn btn-primary video-unblock"
          onClick={() => {
            const el = ref.current;
            if (!el) {
              return;
            }
            const clock = mediaClockRef.current;
            // Gesture = audible start of the theme: align clock + element to 0
            // when hub only extrapolated while autoplay was blocked.
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
            el.muted = false;
            void el
              .play()
              .then(() => setBlocked(false))
              .catch(() => undefined);
          }}
        >
          Clic para audio / play
        </button>
      ) : null}
    </div>
  );
}
