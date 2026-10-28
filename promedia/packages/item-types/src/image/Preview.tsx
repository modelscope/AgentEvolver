import type { ImageFit, ImagePlaylistItem, PlaylistItem } from "@promedia/shared";

export type ImagePreviewProps = {
  item: PlaylistItem;
  slideIndex: number;
  assetSrc?: string;
  onFitChange?: (fit: ImageFit) => void;
};

const FIT_OPTIONS: ImageFit[] = ["contain", "cover", "fill"];

export function ImagePreview({ item, assetSrc, onFitChange }: ImagePreviewProps) {
  if (item.type !== "image") {
    return null;
  }
  const image = item as ImagePlaylistItem;
  const fit = image.payload.fit;

  return (
    <div className="image-preview">
      <div className="image-stage">
        {assetSrc ? (
          <img
            src={assetSrc}
            alt={image.payload.title ?? "image"}
            style={{ objectFit: fit }}
          />
        ) : (
          <p className="image-missing">…</p>
        )}
      </div>
      {onFitChange ? (
        <label className="image-fit-field">
          fit
          <select
            value={fit}
            onChange={(e) => onFitChange(e.target.value as ImageFit)}
          >
            {FIT_OPTIONS.map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </select>
        </label>
      ) : null}
    </div>
  );
}
