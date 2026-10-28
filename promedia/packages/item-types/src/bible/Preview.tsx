import type { BiblePayload, PlaylistItem } from "@promedia/shared";
import { TextPreview } from "../text/Preview.js";

export type BiblePreviewProps = {
  item: PlaylistItem;
  slideIndex: number;
  onSlideIndexChange?: (index: number) => void;
};

/** Reuses text preview; bible payload has the same slides shape. */
export function BiblePreview(props: BiblePreviewProps) {
  if (props.item.type !== "bible") {
    return null;
  }
  const payload = props.item.payload as BiblePayload;
  const textLike = {
    ...props.item,
    type: "text" as const,
    payload: {
      title: payload.reference,
      align: payload.align ?? "center",
      slides: payload.slides,
    },
  };
  return (
    <TextPreview
      item={textLike}
      slideIndex={props.slideIndex}
      onSlideIndexChange={props.onSlideIndexChange}
    />
  );
}
