import { useEffect, useRef, useState } from "react";
import type {
  MediaClock,
  PlaylistItem,
  VideoPlaylistItem,
} from "@promedia/shared";
import { effectiveMediaClockPositionMs } from "@promedia/shared";
import {
  YouTubePlayer,
  type YouTubePlayerHandle,
} from "./YouTubePlayer.js";
import { YouTubeThumb } from "./YouTubeThumb.js";
import {
  VideoTransportBar,
  formatMediaMs,
} from "./VideoTransportBar.js";
import { VideoStageTimeOverlay } from "./VideoStageTimeOverlay.js";

export type VideoPreviewProps = {
  item: PlaylistItem;
  slideIndex: number;
  assetSrc?: string;
  /**
   * When this item is already live, do not mount a second YouTube iframe —
   * Output owns the embed. Show poster + mediaClock transport.
   */
  projecting?: boolean;
  mediaClock?: MediaClock | null;
  onMediaClockSet?: (patch: { playing: boolean; positionMs?: number }) => void;
  onDurationMs?: (ms: number) => void;
  /** Known duration while projecting YouTube (thumb-only; no local probe). */
  durationMsHint?: number;
};

/**
 * Local preview with dedicated transport toolbar so operators can seek /
 * mute / skip before Go Live (does not sync Output until live).
 * YouTube while projecting: thumb + mediaClock transport — Output owns iframe.
 */
export function VideoPreview({
  item,
  assetSrc,
  projecting = false,
  mediaClock = null,
  onMediaClockSet,
  onDurationMs,
  durationMsHint = 0,
}: VideoPreviewProps) {
  if (item.type !== "video") {
    return null;
  }
  const video = item as VideoPlaylistItem;
  const initialMuted = video.payload.muted === true;
  const loop = video.payload.loop === true;
  const fit = video.payload.fit ?? "contain";
  const isYoutube = video.payload.sourceKind === "youtube";

  return (
    <div className="video-preview">
      {video.payload.sourceKind === "file" ? (
        assetSrc ? (
          projecting && mediaClock && onMediaClockSet ? (
            <LiveSyncedFilePreview
              src={assetSrc}
              mediaClock={mediaClock}
              onMediaClockSet={onMediaClockSet}
              muted={initialMuted}
              loop={loop}
              fit={fit}
              onDurationMs={onDurationMs}
            />
          ) : (
            <FilePreviewWithToolbar
              src={assetSrc}
              initialMuted={initialMuted}
              loop={loop}
              fit={fit}
              onDurationMs={onDurationMs}
            />
          )
        ) : (
          <>
            <p className="video-missing">…</p>
            <VideoTransportBar
              playing={false}
              positionMs={0}
              durationMs={0}
              scrubbing={false}
              scrubMs={0}
              disabled
              showMute
              muted={initialMuted}
              onTogglePlay={() => undefined}
              onSkip={() => undefined}
              onSeekTo={() => undefined}
              onToggleMute={() => undefined}
              onScrubChange={() => undefined}
              onScrubCommit={() => undefined}
              onScrubStart={() => undefined}
            />
          </>
        )
      ) : projecting ? (
        <LiveSyncedYouTubePreview
          videoId={video.payload.videoId}
          title={video.payload.title}
          mediaClock={mediaClock}
          onMediaClockSet={onMediaClockSet}
          durationMsHint={durationMsHint}
        />
      ) : (
        <YouTubePreviewWithToolbar
          videoId={video.payload.videoId}
          initialMuted={initialMuted}
          loop={loop}
          onDurationMs={onDurationMs}
        />
      )}
      <p className="video-caveat">
        {isYoutube && projecting
          ? mediaClock?.playing
            ? "Reproduciendo en Output — Preview no embebe un segundo YouTube."
            : "Preview local — reproducción en la ventana Output (un solo iframe)."
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

/** File preview while item is live — follows mediaClock; no second source of truth. */
function LiveSyncedFilePreview({
  src,
  mediaClock,
  onMediaClockSet,
  muted,
  loop,
  fit,
  onDurationMs,
}: {
  src: string;
  mediaClock: MediaClock;
  onMediaClockSet: (patch: { playing: boolean; positionMs?: number }) => void;
  muted: boolean;
  loop: boolean;
  fit: "contain" | "cover";
  onDurationMs?: (ms: number) => void;
}) {
  const ref = useRef<HTMLVideoElement>(null);
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
    el.muted = muted;
    el.loop = loop;
    if (mediaClock.playing) {
      void el.play().catch(() => undefined);
    } else {
      el.pause();
    }
  }, [mediaClock, muted, loop]);

  return (
    <div className="video-preview-scrub">
      <div className="video-file-wrap">
        <video
          ref={ref}
          src={src}
          className="video-el"
          style={{ objectFit: fit }}
          playsInline
          controls={false}
          loop={loop}
          muted={muted}
          autoPlay={false}
        />
        <VideoStageTimeOverlay
          positionMs={transport.scrubbing ? transport.scrubMs : transport.pos}
          durationMs={durationMs}
        />
      </div>
      <MediaClockTransportBar
        transport={transport}
        durationMs={durationMs}
        onMediaClockSet={onMediaClockSet}
      />
    </div>
  );
}

function LiveSyncedYouTubePreview({
  videoId,
  title,
  mediaClock,
  onMediaClockSet,
  durationMsHint,
}: {
  videoId: string;
  title?: string;
  mediaClock?: MediaClock | null;
  onMediaClockSet?: (patch: { playing: boolean; positionMs?: number }) => void;
  durationMsHint: number;
}) {
  const transport = useMediaClockTransport(
    mediaClock,
    onMediaClockSet,
    durationMsHint,
  );

  return (
    <div className="video-preview-scrub">
      <div className="video-file-wrap">
        <YouTubeThumb
          videoId={videoId}
          title={title}
          className="video-yt-thumb-fill"
          playing={transport.playing}
          playingLabel="Reproduciendo en Output"
        />
        <VideoStageTimeOverlay
          positionMs={
            transport.scrubbing ? transport.scrubMs : transport.pos
          }
          durationMs={durationMsHint}
        />
      </div>
      <p className="preview-toolbar-note preview-toolbar-note-inline">
        {transport.playing
          ? "Audio y vídeo van en la ventana Output (un solo iframe). Aquí: transporte + miniatura."
          : "YouTube en Output — aquí solo miniatura + transporte (un solo iframe)."}
      </p>
      <MediaClockTransportBar
        transport={transport}
        durationMs={durationMsHint}
        onMediaClockSet={onMediaClockSet}
      />
    </div>
  );
}

function MediaClockTransportBar({
  transport,
  durationMs,
  onMediaClockSet,
}: {
  transport: ReturnType<typeof useMediaClockTransport>;
  durationMs: number;
  onMediaClockSet?: (patch: { playing: boolean; positionMs?: number }) => void;
}) {
  const {
    scrubbing,
    scrubMs,
    pos,
    playing,
    canControl,
    seekTo,
    setScrubbing,
    setScrubMs,
  } = transport;

  return (
    <VideoTransportBar
      playing={playing}
      positionMs={pos}
      durationMs={durationMs}
      scrubbing={scrubbing}
      scrubMs={scrubMs}
      disabled={!canControl}
      showMute={false}
      remainingLabel={
        durationMs > 0
          ? `−${formatMediaMs(Math.max(0, durationMs - (scrubbing ? scrubMs : pos)))}`
          : "−−:−−"
      }
      onTogglePlay={() =>
        onMediaClockSet?.({
          playing: !playing,
          positionMs: scrubbing ? scrubMs : pos,
        })
      }
      onSkip={(delta) => seekTo((scrubbing ? scrubMs : pos) + delta)}
      onSeekTo={seekTo}
      onScrubStart={() => {
        setScrubbing(true);
        setScrubMs(pos);
      }}
      onScrubChange={(ms) => {
        setScrubbing(true);
        setScrubMs(ms);
      }}
      onScrubCommit={() => {
        seekTo(scrubMs);
        setScrubbing(false);
      }}
    />
  );
}

function FilePreviewWithToolbar({
  src,
  initialMuted,
  loop,
  fit,
  onDurationMs,
}: {
  src: string;
  initialMuted: boolean;
  loop: boolean;
  fit: "contain" | "cover";
  onDurationMs?: (ms: number) => void;
}) {
  const ref = useRef<HTMLVideoElement>(null);
  const [durationMs, setDurationMs] = useState(0);
  const [positionMs, setPositionMs] = useState(0);
  const [scrubbing, setScrubbing] = useState(false);
  const [scrubMs, setScrubMs] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [muted, setMuted] = useState(initialMuted);

  useEffect(() => {
    setMuted(initialMuted);
  }, [initialMuted, src]);

  useEffect(() => {
    const el = ref.current;
    if (!el) {
      return;
    }
    el.muted = muted;
  }, [muted]);

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
    <div className="video-preview-scrub">
      <div className="video-file-wrap">
        <video
          ref={ref}
          src={src}
          className="video-el"
          style={{ objectFit: fit }}
          playsInline
          controls={false}
          loop={loop}
          muted={muted}
          autoPlay={false}
        />
        <VideoStageTimeOverlay
          positionMs={displayMs}
          durationMs={durationMs}
        />
      </div>
      <VideoTransportBar
        playing={playing}
        muted={muted}
        showMute
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
        onToggleMute={() => setMuted((m) => !m)}
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

function YouTubePreviewWithToolbar({
  videoId,
  initialMuted,
  loop,
  onDurationMs,
}: {
  videoId: string;
  initialMuted: boolean;
  loop: boolean;
  onDurationMs?: (ms: number) => void;
}) {
  const playerRef = useRef<YouTubePlayerHandle>(null);
  const [ready, setReady] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [muted, setMuted] = useState(initialMuted);
  const [durationMs, setDurationMs] = useState(0);
  const [positionMs, setPositionMs] = useState(0);
  const [scrubbing, setScrubbing] = useState(false);
  const [scrubMs, setScrubMs] = useState(0);

  useEffect(() => {
    setMuted(initialMuted);
    setPlaying(false);
    setPositionMs(0);
    setDurationMs(0);
  }, [initialMuted, videoId]);

  useEffect(() => {
    if (!ready) {
      return;
    }
    const id = window.setInterval(() => {
      const p = playerRef.current;
      if (!p?.isReady()) {
        return;
      }
      const dur = p.getDurationMs();
      if (dur > 0) {
        setDurationMs(dur);
        onDurationMs?.(dur);
      }
      if (!scrubbing) {
        setPositionMs(p.getCurrentTimeMs());
      }
    }, 250);
    return () => window.clearInterval(id);
  }, [ready, scrubbing, onDurationMs]);

  useEffect(() => {
    const p = playerRef.current;
    if (!p?.isReady()) {
      return;
    }
    if (muted) {
      p.mute();
    } else {
      p.unMute();
    }
  }, [muted, ready]);

  const seekTo = (ms: number) => {
    const p = playerRef.current;
    if (!p) {
      return;
    }
    const clamped = Math.max(0, Math.min(ms, durationMs || ms));
    p.seekToMs(clamped);
    setPositionMs(clamped);
  };

  const displayMs = scrubbing ? scrubMs : positionMs;

  return (
    <div className="video-preview-scrub">
      <div className="video-yt-wrap">
        <YouTubePlayer
          ref={playerRef}
          videoId={videoId}
          preview
          hideNativeControls
          muted={muted}
          loop={loop}
          onReadyChange={setReady}
          onPlayingChange={setPlaying}
        />
        <VideoStageTimeOverlay
          positionMs={displayMs}
          durationMs={durationMs}
        />
      </div>
      <VideoTransportBar
        playing={playing}
        muted={muted}
        showMute
        positionMs={positionMs}
        durationMs={durationMs}
        scrubbing={scrubbing}
        scrubMs={scrubMs}
        disabled={!ready}
        onTogglePlay={() => {
          const p = playerRef.current;
          if (!p) {
            return;
          }
          if (playing) {
            p.pause();
          } else {
            p.play();
          }
        }}
        onSkip={(delta) => {
          seekTo(displayMs + delta);
        }}
        onSeekTo={seekTo}
        onToggleMute={() => setMuted((m) => !m)}
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
