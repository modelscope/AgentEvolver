import type {
  AudioPlaylistItem,
  MediaClock,
  PlaylistItem,
} from "@promedia/shared";
import { AudioStage } from "./AudioStage.js";
import { FileAudio } from "./FileAudio.js";

export type AudioLivePanelProps = {
  item: PlaylistItem;
  slideIndex: number;
  assetSrc?: string;
  mediaClock?: MediaClock | null;
  onMediaClockSet?: (patch: { playing: boolean; positionMs?: number }) => void;
  durationMsHint?: number;
  onDurationMs?: (ms: number) => void;
};

/**
 * Live monitor — content only (title art + synced audio). Transport stays on
 * Vista previa when projecting.
 */
export function AudioLivePanel({
  item,
  assetSrc,
  mediaClock,
  onDurationMs,
}: AudioLivePanelProps) {
  if (item.type !== "audio") {
    return null;
  }
  const audio = item as AudioPlaylistItem;
  const loop = audio.payload.loop === true;

  return (
    <div className="audio-live-panel">
      <div className="audio-file-wrap">
        <AudioStage title={audio.payload.title} />
        {assetSrc ? (
          <FileAudio
            src={assetSrc}
            mediaClock={mediaClock}
            loop={loop}
            onDurationMs={onDurationMs}
          />
        ) : (
          <p className="audio-missing">…</p>
        )}
      </div>
    </div>
  );
}
