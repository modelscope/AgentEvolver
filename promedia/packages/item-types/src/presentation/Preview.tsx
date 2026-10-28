import type {
  ImageFit,
  PlaylistItem,
  PresentationPlaylistItem,
} from "@promedia/shared";

export type PresentationPreviewProps = {
  item: PlaylistItem;
  slideIndex: number;
  assetSrc?: string;
  onFitChange?: (fit: ImageFit) => void;
};

const FIT_OPTIONS: ImageFit[] = ["contain", "cover", "fill"];

export function PresentationPreview({
  item,
  slideIndex,
  assetSrc,
  onFitChange,
}: PresentationPreviewProps) {
  if (item.type !== "presentation") {
    return null;
  }
  const presentation = item as PresentationPlaylistItem;
  const fit = presentation.payload.fit;
  const total = presentation.payload.slides.length;
  const label = presentation.payload.title ?? "presentación";

  return (
    <div className="presentation-preview image-preview">
      <div className="image-stage">
        {assetSrc ? (
          <img
            src={assetSrc}
            alt={label}
            style={{ objectFit: fit }}
          />
        ) : (
          <p className="image-missing">…</p>
        )}
      </div>
      <p className="presentation-slide-meta">
        {Math.min(slideIndex + 1, total)} / {total}
      </p>
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
