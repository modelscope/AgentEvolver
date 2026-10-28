import { useEffect, useRef, useState } from "react";
import type {
  AudioPlaylistItem,
  MediaClock,
  PlaylistItem,
} from "@promedia/shared";
import { effectiveMediaClockPositionMs } from "@promedia/shared";
import { AudioStage } from "./AudioStage.js";
import {
  VideoTransportBar,
  formatMediaMs,
} from "../video/VideoTransportBar.js";
import { VideoStageTimeOverlay } from "../video/VideoStageTimeOverlay.js";

export type AudioPreviewProps = {
  item: PlaylistItem;
  slideIndex: number;
  assetSrc?: string;
  projecting?: boolean;
  mediaClock?: MediaClock | null;
  onMediaClockSet?: (patch: { playing: boolean; positionMs?: number }) => void;
  onDurationMs?: (ms: number) => void;
  durationMsHint?: number;
};

/**
 * Local audio preview with transport (same pattern as file video).
 * While projecting, follows mediaClock — Output / En vivo share the clock.
 */
export function AudioPreview({
  item,
  assetSrc,
  projecting = false,
  mediaClock = null,
  onMediaClockSet,
  onDurationMs,
}: AudioPreviewProps) {
  if (item.type !== "audio") {
    return null;
  }
  const audio = item as AudioPlaylistItem;
  const loop = audio.payload.loop === true;
  const title = audio.payload.title;

  return (
    <div className="audio-preview">
      {assetSrc ? (
        projecting && mediaClock && onMediaClockSet ? (
          <LiveSyncedAudioPreview
            src={assetSrc}
            title={title}
            mediaClock={mediaClock}
            onMediaClockSet={onMediaClockSet}
            loop={loop}
            onDurationMs={onDurationMs}
          />
        ) : (
          <LocalAudioPreview
            src={assetSrc}
            title={title}
            loop={loop}
            onDurationMs={onDurationMs}
          />
        )
      ) : (
        <>
          <AudioStage title={title} />
          <VideoTransportBar
            playing={false}
            positionMs={0}
            durationMs={0}
            scrubbing={false}
            scrubMs={0}
            disabled
            showMute={false}
            onTogglePlay={() => undefined}
            onSkip={() => undefined}
            onSeekTo={() => undefined}
            onScrubChange={() => undefined}
            onScrubCommit={() => undefined}
            onScrubStart={() => undefined}
          />
        </>
      )}
      <p className="audio-caveat">
        {projecting
          ? "Preview local — reproducción sincronizada con En vivo / Output."
          : "Preview local — no sincroniza Output hasta Go Live."}
      </p>
    </div>
  );
}

function useMediaClockTransport(
  mediaClock: MediaClock | null | undefined,
  onMediaClockSet:
    | ((patch: { playing: boolean; positionMs?: number }) => void)
    | undefined,
  durationMs: number,
) {
  const [scrubbing, setScrubbing] = useState(false);
  const [scrubMs, setScrubMs] = useState(0);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    if (!mediaClock?.playing || scrubbing) {
      return;
    }
    const id = window.setInterval(() => setTick((n) => n + 1), 250);
    return () => window.clearInterval(id);
  }, [mediaClock?.playing, mediaClock?.updatedAt, scrubbing]);

  const pos = mediaClock ? effectiveMediaClockPositionMs(mediaClock) : 0;
  void tick;
  const playing = mediaClock?.playing ?? false;
  const canControl = Boolean(onMediaClockSet && mediaClock);

  const seekTo = (ms: number) => {
    if (!onMediaClockSet) {
      return;
    }
    const clamped =
      durationMs > 0
        ? Math.max(0, Math.min(ms, durationMs))
        : Math.max(0, ms);
    onMediaClockSet({ playing, positionMs: clamped });
    setScrubMs(clamped);
  };

  return {
    scrubbing,
    scrubMs,
    pos,
    playing,
    canControl,
    seekTo,
    setScrubbing,
    setScrubMs,
  };
}

function LiveSyncedAudioPreview({
  src,
  title,
  mediaClock,
  onMediaClockSet,
  loop,
  onDurationMs,
}: {
  src: string;
  title?: string;
  mediaClock: MediaClock;
  onMediaClockSet: (patch: { playing: boolean; positionMs?: number }) => void;
  loop: boolean;
  onDurationMs?: (ms: number) => void;
}) {
  const ref = useRef<HTMLAudioElement>(null);
  const [durationMs, setDurationMs] = useState(0);
  const transport = useMediaClockTransport(
    mediaClock,
    onMediaClockSet,
    durationMs,
  );

  useEffect(() => {
    const el = ref.current;
    if (!el) {
      return;
    }
    const report = () => {
      const ms = (el.duration || 0) * 1000;
      if (ms > 0) {
        setDurationMs(ms);
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
    if (!el) {
      return;
    }
    const target = effectiveMediaClockPositionMs(mediaClock) / 1000;
    if (Math.abs(el.currentTime - target) > 0.4) {
      el.currentTime = target;
    }
    el.loop = loop;
    if (mediaClock.playing) {
      void el.play().catch(() => undefined);
    } else {
      el.pause();
    }
  }, [mediaClock, loop]);

  return (
    <div className="audio-preview-scrub">
      <div className="audio-file-wrap">
        <AudioStage title={title} />
        <audio ref={ref} src={src} preload="metadata" loop={loop} />
        <VideoStageTimeOverlay
          positionMs={transport.scrubbing ? transport.scrubMs : transport.pos}
          durationMs={durationMs}
        />
      </div>
      <VideoTransportBar
        playing={transport.playing}
        positionMs={transport.pos}
        durationMs={durationMs}
        scrubbing={transport.scrubbing}
        scrubMs={transport.scrubMs}
        disabled={!transport.canControl}
        showMute={false}
        remainingLabel={
          durationMs > 0
            ? `−${formatMediaMs(Math.max(0, durationMs - (transport.scrubbing ? transport.scrubMs : transport.pos)))}`
            : "−−:−−"
        }
        onTogglePlay={() =>
          onMediaClockSet({
            playing: !transport.playing,
            positionMs: transport.scrubbing ? transport.scrubMs : transport.pos,
          })
        }
        onSkip={(delta) =>
          transport.seekTo(
            (transport.scrubbing ? transport.scrubMs : transport.pos) + delta,
          )
        }
        onSeekTo={transport.seekTo}
        onScrubStart={() => {
          transport.setScrubbing(true);
          transport.setScrubMs(transport.pos);
        }}
        onScrubChange={(ms) => {
          transport.setScrubbing(true);
          transport.setScrubMs(ms);
        }}
        onScrubCommit={() => {
          transport.seekTo(transport.scrubMs);
          transport.setScrubbing(false);
        }}
      />
    </div>
  );
}

function LocalAudioPreview({
  src,
  title,
  loop,
  onDurationMs,
}: {
  src: string;
  title?: string;
  loop: boolean;
  onDurationMs?: (ms: number) => void;
}) {
  const ref = useRef<HTMLAudioElement>(null);
  const [durationMs, setDurationMs] = useState(0);
  const [positionMs, setPositionMs] = useState(0);
  const [scrubbing, setScrubbing] = useState(false);
  const [scrubMs, setScrubMs] = useState(0);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) {
      return;
    }
    const onMeta = () => {
      const ms = (el.duration || 0) * 1000;
      setDurationMs(ms);
      if (ms > 0) {
        onDurationMs?.(ms);
      }
    };
    const onTime = () => {
      if (!scrubbing) {
        setPositionMs(el.currentTime * 1000);
      }
    };
    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    el.addEventListener("loadedmetadata", onMeta);
    el.addEventListener("durationchange", onMeta);
    el.addEventListener("timeupdate", onTime);
    el.addEventListener("play", onPlay);
    el.addEventListener("pause", onPause);
    return () => {
      el.removeEventListener("loadedmetadata", onMeta);
      el.removeEventListener("durationchange", onMeta);
      el.removeEventListener("timeupdate", onTime);
      el.removeEventListener("play", onPlay);
      el.removeEventListener("pause", onPause);
    };
  }, [scrubbing, src, onDurationMs]);

  const seekTo = (ms: number) => {
    const el = ref.current;
    if (!el) {
      return;
    }
    const clamped = Math.max(0, Math.min(ms, durationMs || ms));
    el.currentTime = clamped / 1000;
    setPositionMs(clamped);
  };

  const displayMs = scrubbing ? scrubMs : positionMs;

  return (
    <div className="audio-preview-scrub">
      <div className="audio-file-wrap">
        <AudioStage title={title} />
        <audio ref={ref} src={src} preload="metadata" loop={loop} />
        <VideoStageTimeOverlay
          positionMs={displayMs}
          durationMs={durationMs}
        />
      </div>
      <VideoTransportBar
        playing={playing}
        showMute={false}
        positionMs={positionMs}
        durationMs={durationMs}
        scrubbing={scrubbing}
        scrubMs={scrubMs}
        onTogglePlay={() => {
          const el = ref.current;
          if (!el) {
            return;
          }
          if (el.paused) {
            void el.play().catch(() => undefined);
          } else {
            el.pause();
          }
        }}
        onSkip={(delta) => {
          seekTo(displayMs + delta);
        }}
        onSeekTo={seekTo}
        onScrubStart={() => setScrubbing(true)}
        onScrubChange={(ms) => {
          setScrubbing(true);
          setScrubMs(ms);
        }}
        onScrubCommit={() => {
          seekTo(scrubMs);
          setScrubbing(false);
        }}
      />
    </div>
  );
}
