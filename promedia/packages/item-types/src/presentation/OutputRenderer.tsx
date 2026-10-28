import type { PlaylistItem, PresentationPlaylistItem } from "@promedia/shared";

export type PresentationOutputRendererProps = {
  item: PlaylistItem;
  slideIndex: number;
  assetSrc?: string;
};

export function PresentationOutputRenderer({
  item,
  assetSrc,
}: PresentationOutputRendererProps) {
  if (item.type !== "presentation") {
    return null;
  }
  const presentation = item as PresentationPlaylistItem;
  return (
    <div className="presentation-output image-output">
      {assetSrc ? (
        <img
          src={assetSrc}
          alt={presentation.payload.title ?? "presentación"}
          style={{ objectFit: presentation.payload.fit }}
        />
      ) : (
        <p className="output-waiting">…</p>
      )}
    </div>
  );
}
