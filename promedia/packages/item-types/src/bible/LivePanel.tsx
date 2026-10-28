import type { BiblePayload, PlaylistItem } from "@promedia/shared";
import { TextLivePanel } from "../text/LivePanel.js";

export type BibleLivePanelProps = {
  item: PlaylistItem;
  slideIndex: number;
  onPrev?: () => void;
  onNext?: () => void;
};

export function BibleLivePanel(props: BibleLivePanelProps) {
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
    <TextLivePanel
      item={textLike}
      slideIndex={props.slideIndex}
      onPrev={props.onPrev}
      onNext={props.onNext}
    />
  );
}
