import type {
  MediaClock,
  PlaylistItem,
  VideoPlaylistItem,
} from "@promedia/shared";
import { FileVideo } from "./FileVideo.js";
import { YouTubeThumb } from "./YouTubeThumb.js";

export type VideoLivePanelProps = {
  item: PlaylistItem;
  slideIndex: number;
  assetSrc?: string;
  mediaClock?: MediaClock | null;
  onMediaClockSet?: (patch: { playing: boolean; positionMs?: number }) => void;
  /** Optional duration hint (ms) when known from preview / metadata. */
  durationMsHint?: number;
  onDurationMs?: (ms: number) => void;
};

/**
 * Live monitor — content only (no transport / timers). Video transport stays
 * on Vista previa; Output owns the YouTube iframe.
 */
export function VideoLivePanel({
  item,
  assetSrc,
  mediaClock,
  onDurationMs,
}: VideoLivePanelProps) {
  if (item.type !== "video") {
    return null;
  }
  const video = item as VideoPlaylistItem;
  const muted = video.payload.muted === true;
  const loop = video.payload.loop === true;
  const fit = video.payload.fit ?? "contain";

  return (
    <div className="video-live-panel">
      <div className="video-stage">
        {video.payload.sourceKind === "file" ? (
          assetSrc ? (
            <FileVideo
              src={assetSrc}
              mediaClock={mediaClock}
              muted={muted}
              loop={loop}
              fit={fit}
              onDurationMs={onDurationMs}
            />
          ) : (
            <p className="video-missing">…</p>
          )
        ) : (
          <YouTubeThumb
            videoId={video.payload.videoId}
            title={video.payload.title}
          />
        )}
      </div>
    </div>
  );
}
