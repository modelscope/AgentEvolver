import type { ComponentType } from "react";
import type { ImageFit, MediaClock, PlaylistItem } from "@promedia/shared";
import { BibleLivePanel } from "./bible/LivePanel.js";
import { BibleOutputRenderer } from "./bible/OutputRenderer.js";
import { BiblePreview } from "./bible/Preview.js";
import { isBiblePlaylistItem } from "./bible/schema.js";
import { ImageLivePanel } from "./image/LivePanel.js";
import { ImageOutputRenderer } from "./image/OutputRenderer.js";
import { ImagePreview } from "./image/Preview.js";
import { isImagePlaylistItem } from "./image/schema.js";
import { TextLivePanel } from "./text/LivePanel.js";
import { TextOutputRenderer } from "./text/OutputRenderer.js";
import { TextPreview } from "./text/Preview.js";
import { isTextPlaylistItem } from "./text/schema.js";
import { PresentationLivePanel } from "./presentation/LivePanel.js";
import { PresentationOutputRenderer } from "./presentation/OutputRenderer.js";
import { PresentationPreview } from "./presentation/Preview.js";
import { isPresentationPlaylistItem } from "./presentation/schema.js";
import { AudioLivePanel } from "./audio/LivePanel.js";
import { AudioOutputRenderer } from "./audio/OutputRenderer.js";
import { AudioPreview } from "./audio/Preview.js";
import { isAudioPlaylistItem } from "./audio/schema.js";
import { VideoLivePanel } from "./video/LivePanel.js";
import { VideoOutputRenderer } from "./video/OutputRenderer.js";
import { VideoPreview } from "./video/Preview.js";
import { isVideoPlaylistItem } from "./video/schema.js";

export type MediaTypeModule = {
  type: string;
  labelKey: string;
  validate: (item: unknown) => item is PlaylistItem;
  Preview: ComponentType<{
    item: PlaylistItem;
    slideIndex: number;
    onSlideIndexChange?: (index: number) => void;
    assetSrc?: string;
    onFitChange?: (fit: ImageFit) => void;
    /** Video/YouTube: true when this item is live — avoid a second YT iframe. */
    projecting?: boolean;
    mediaClock?: MediaClock | null;
    onMediaClockSet?: (patch: {
      playing: boolean;
      positionMs?: number;
    }) => void;
    onDurationMs?: (ms: number) => void;
    durationMsHint?: number;
  }>;
  LivePanel: ComponentType<{
    item: PlaylistItem;
    slideIndex: number;
    onPrev?: () => void;
    onNext?: () => void;
    assetSrc?: string;
    mediaClock?: MediaClock | null;
    onMediaClockSet?: (patch: {
      playing: boolean;
      positionMs?: number;
    }) => void;
    durationMsHint?: number;
    onDurationMs?: (ms: number) => void;
  }>;
  OutputRenderer: ComponentType<{
    item: PlaylistItem;
    slideIndex: number;
    assetSrc?: string;
    mediaClock?: MediaClock | null;
    /** Video/audio Output: snap hub clock when audible start follows autoplay block. */
    onMediaClockSet?: (patch: {
      playing: boolean;
      positionMs?: number;
    }) => void;
  }>;
};

const textModule: MediaTypeModule = {
  type: "text",
  labelKey: "itemTypes.text",
  validate: isTextPlaylistItem,
  Preview: TextPreview,
  LivePanel: TextLivePanel,
  OutputRenderer: TextOutputRenderer,
};

const imageModule: MediaTypeModule = {
  type: "image",
  labelKey: "itemTypes.image",
  validate: isImagePlaylistItem,
  Preview: ImagePreview,
  LivePanel: ImageLivePanel,
  OutputRenderer: ImageOutputRenderer,
};

const videoModule: MediaTypeModule = {
  type: "video",
  labelKey: "itemTypes.video",
  validate: isVideoPlaylistItem,
  Preview: VideoPreview,
  LivePanel: VideoLivePanel,
  OutputRenderer: VideoOutputRenderer,
};

const audioModule: MediaTypeModule = {
  type: "audio",
  labelKey: "itemTypes.audio",
  validate: isAudioPlaylistItem,
  Preview: AudioPreview,
  LivePanel: AudioLivePanel,
  OutputRenderer: AudioOutputRenderer,
};

const bibleModule: MediaTypeModule = {
  type: "bible",
  labelKey: "itemTypes.bible",
  validate: isBiblePlaylistItem,
  Preview: BiblePreview,
  LivePanel: BibleLivePanel,
  OutputRenderer: BibleOutputRenderer,
};

const presentationModule: MediaTypeModule = {
  type: "presentation",
  labelKey: "itemTypes.presentation",
  validate: isPresentationPlaylistItem,
  Preview: PresentationPreview,
  LivePanel: PresentationLivePanel,
  OutputRenderer: PresentationOutputRenderer,
};

const modules: MediaTypeModule[] = [
  textModule,
  imageModule,
  videoModule,
  audioModule,
  bibleModule,
  presentationModule,
];

export const itemTypeRegistry: readonly MediaTypeModule[] = modules;

export function getItemType(type: string): MediaTypeModule | undefined {
  return modules.find((m) => m.type === type);
}

export function listItemTypes(): readonly MediaTypeModule[] {
  return itemTypeRegistry;
}
