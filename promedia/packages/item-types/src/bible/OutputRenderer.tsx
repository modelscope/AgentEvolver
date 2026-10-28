import type { BiblePayload, PlaylistItem } from "@promedia/shared";
import { TextOutputRenderer } from "../text/OutputRenderer.js";

export type BibleOutputRendererProps = {
  item: PlaylistItem;
  slideIndex: number;
};

export function BibleOutputRenderer(props: BibleOutputRendererProps) {
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
    <TextOutputRenderer item={textLike} slideIndex={props.slideIndex} />
  );
}
