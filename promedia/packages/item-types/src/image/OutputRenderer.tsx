import type { ImagePlaylistItem, PlaylistItem } from "@promedia/shared";

export type ImageOutputRendererProps = {
  item: PlaylistItem;
  slideIndex: number;
  assetSrc?: string;
};

export function ImageOutputRenderer({ item, assetSrc }: ImageOutputRendererProps) {
  if (item.type !== "image") {
    return null;
  }
  const image = item as ImagePlaylistItem;
  return (
    <div className="image-output">
      {assetSrc ? (
        <img
          src={assetSrc}
          alt={image.payload.title ?? "image"}
          style={{ objectFit: image.payload.fit }}
        />
      ) : (
        <p className="output-waiting">…</p>
      )}
    </div>
  );
}
