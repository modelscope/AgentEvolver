import type { PlaylistItem, TextPayload } from "@promedia/shared";
import { projectionTextSlides } from "@promedia/shared";
import { FittedTextStage } from "./FittedTextStage.js";

export type TextOutputRendererProps = {
  item: PlaylistItem;
  slideIndex: number;
};

/**
 * Text on Output. Align / colors / font family come from session theme CSS
 * vars (`--pm-*` on `.pm-theme`); item.payload.align does not win (subtarea 8).
 * Shows at most 5 lines per slide; font size auto-fits the projection stage.
 */
export function TextOutputRenderer({ item, slideIndex }: TextOutputRendererProps) {
  const payload = item.payload as TextPayload;
  const slides = projectionTextSlides(payload.slides);
  const index = Math.min(Math.max(slideIndex, 0), Math.max(slides.length - 1, 0));
  const slide = slides[index];
  const lines = slide?.lines ?? [""];

  return (
    <FittedTextStage
      lines={lines}
      lineKey={slide?.id ?? `out-${index}`}
      className="text-output"
      lineClassName="text-output-line"
    />
  );
}
