import type { PlaylistItem, TextPayload } from "@promedia/shared";
import { projectionTextSlides } from "@promedia/shared";
import { FittedTextStage } from "./FittedTextStage.js";

export type TextPreviewProps = {
  item: PlaylistItem;
  slideIndex: number;
  onSlideIndexChange?: (index: number) => void;
};

/** Preview uses the same theme CSS vars + fit path as Output. */
export function TextPreview({ item, slideIndex }: TextPreviewProps) {
  const payload = item.payload as TextPayload;
  const slides = projectionTextSlides(payload.slides);
  const index = Math.min(Math.max(slideIndex, 0), Math.max(slides.length - 1, 0));
  const slide = slides[index];
  const lines = slide?.lines ?? [""];

  return (
    <div className="text-preview">
      <FittedTextStage
        lines={lines}
        lineKey={slide?.id ?? `prev-${index}`}
        className="text-stage"
        lineClassName="text-line"
      />
    </div>
  );
}
