import type { TextAlign } from "@promedia/shared";

export type TextAlignCss = {
  textAlign: "left" | "center" | "right";
  justifyContent: "flex-start" | "center" | "flex-end";
};

export function textAlignCss(align: TextAlign | undefined): TextAlignCss {
  const a = align ?? "center";
  return {
    textAlign: a,
    justifyContent: a === "left" ? "flex-start" : a === "right" ? "flex-end" : "center",
  };
}
