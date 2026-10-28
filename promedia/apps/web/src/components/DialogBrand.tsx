import type { ReactNode } from "react";

type Props = {
  id: string;
  children: ReactNode;
};

/** Dialog title row with ProMedia mark (Guion-opened dialogs). */
export function DialogBrand({ id, children }: Props) {
  return (
    <h3 id={id} className="promedia-dialog-title promedia-dialog-title-brand">
      <img
        className="promedia-dialog-logo"
        src="/logo-mark.svg"
        alt=""
        width={28}
        height={28}
        decoding="async"
      />
      <span>{children}</span>
    </h3>
  );
}
