import type { PlaylistItem, TextPayload } from "@promedia/shared";
import { resolveProjectionTextSlides } from "@promedia/shared";
import { FittedTextStage } from "./FittedTextStage.js";

export type TextOutputRendererProps = {
  item: PlaylistItem;
  slideIndex: number;
};

/**
 * Text on Output. Align / colors / font family come from session theme CSS
 * vars (`--pm-*` on `.pm-theme`); item.payload.align does not win (subtarea 8).
 * Songs with chunkLines:false keep one lyric block; karaoke agent advances lines.
 */
export function TextOutputRenderer({ item, slideIndex }: TextOutputRendererProps) {
  const payload = item.payload as TextPayload;
  const slides = resolveProjectionTextSlides(payload);
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
