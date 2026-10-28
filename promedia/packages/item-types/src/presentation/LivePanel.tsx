import type { PlaylistItem, PresentationPlaylistItem } from "@promedia/shared";

export type PresentationLivePanelProps = {
  item: PlaylistItem;
  slideIndex: number;
  assetSrc?: string;
  onPrev?: () => void;
  onNext?: () => void;
};

/** Live monitor — slide image only (nav stays on Vista previa / shortcuts). */
export function PresentationLivePanel({
  item,
  assetSrc,
}: PresentationLivePanelProps) {
  if (item.type !== "presentation") {
    return null;
  }
  const presentation = item as PresentationPlaylistItem;
  return (
    <div className="presentation-live-panel image-live-panel">
      <div className="image-stage">
        {assetSrc ? (
          <img
            src={assetSrc}
            alt={presentation.payload.title ?? "presentación"}
            style={{ objectFit: presentation.payload.fit }}
          />
        ) : (
          <p className="image-missing">…</p>
        )}
      </div>
    </div>
  );
}
