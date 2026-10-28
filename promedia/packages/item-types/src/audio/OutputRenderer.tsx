import type {
  AudioPlaylistItem,
  MediaClock,
  PlaylistItem,
} from "@promedia/shared";
import { AudioStage } from "./AudioStage.js";
import { FileAudio } from "./FileAudio.js";

export type AudioOutputRendererProps = {
  item: PlaylistItem;
  slideIndex: number;
  assetSrc?: string;
  mediaClock?: MediaClock | null;
  onMediaClockSet?: (patch: { playing: boolean; positionMs?: number }) => void;
};

/** Output window — title art + audio synced to mediaClock. */
export function AudioOutputRenderer({
  item,
  assetSrc,
  mediaClock,
  onMediaClockSet,
}: AudioOutputRendererProps) {
  if (item.type !== "audio") {
    return null;
  }
  const audio = item as AudioPlaylistItem;
  const loop = audio.payload.loop === true;

  return (
    <div className="audio-output">
      <div className="audio-file-wrap">
        <AudioStage title={audio.payload.title} />
        {assetSrc ? (
          <FileAudio
            src={assetSrc}
            mediaClock={mediaClock}
            loop={loop}
            onMediaClockSet={onMediaClockSet}
          />
        ) : (
          <p className="output-waiting">…</p>
        )}
      </div>
    </div>
  );
}
