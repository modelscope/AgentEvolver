/** Shared SVG glyphs for Guion toolbar + preview transport (no emoji). */

type IconProps = {
  className?: string;
  size?: number;
};

function svgProps({ className, size = 18 }: IconProps) {
  return {
    className: className ?? "ui-icon-svg",
    width: size,
    height: size,
    viewBox: "0 0 24 24",
    fill: "none",
    "aria-hidden": true as const,
  };
}

/** Primary «Añadir»: plus + files — opens multi-file picker. */
export function IconAddFiles(props: IconProps) {
  return (
    <svg {...svgProps(props)}>
      <path
        d="M4 7.5A2.5 2.5 0 0 1 6.5 5H10l1.5 1.5H17.5A2.5 2.5 0 0 1 20 9v8.5a2.5 2.5 0 0 1-2.5 2.5h-11A2.5 2.5 0 0 1 4 17.5v-10Z"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
      <path
        d="M12 10.5v6M9 13.5h6"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function IconChevronDown(props: IconProps) {
  return (
    <svg {...svgProps({ ...props, size: props.size ?? 14 })}>
      <path
        d="M6 9.5 12 15l6-5.5"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

/** Open / load playlist dialog. */
export function IconFolderOpen(props: IconProps) {
  return (
    <svg {...svgProps(props)}>
      <path
        fill="currentColor"
        d="M2 6a2 2 0 0 1 2-2h5l2 2h9a2 2 0 0 1 2 2v1H4a2 2 0 0 0-2 2v7a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-1V8a1 1 0 0 0-1-1H10.83l-2-2H4a1 1 0 0 0-1 1v1H2V6z"
      />
    </svg>
  );
}

/** Save playlist dialog. */
export function IconSave(props: IconProps) {
  return (
    <svg {...svgProps(props)}>
      <path
        fill="currentColor"
        d="M17 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V7l-4-4zm-5 16a3 3 0 1 1 0-6 3 3 0 0 1 0 6zm3-10H5V5h10v4z"
      />
    </svg>
  );
}

export function IconPlay(props: IconProps) {
  return (
    <svg {...svgProps(props)}>
      <path d="M8 5.5v13l11-6.5L8 5.5Z" fill="currentColor" />
    </svg>
  );
}

export function IconPause(props: IconProps) {
  return (
    <svg {...svgProps(props)}>
      <rect x="6" y="5" width="4" height="14" rx="1" fill="currentColor" />
      <rect x="14" y="5" width="4" height="14" rx="1" fill="currentColor" />
    </svg>
  );
}

export function IconSkipBack(props: IconProps) {
  return (
    <svg {...svgProps(props)}>
      <path d="M11 12 18 6.5v11L11 12Z" fill="currentColor" />
      <path d="M6 12 13 6.5v11L6 12Z" fill="currentColor" />
      <rect x="4" y="6" width="2.2" height="12" rx="0.6" fill="currentColor" />
    </svg>
  );
}

export function IconSkipForward(props: IconProps) {
  return (
    <svg {...svgProps(props)}>
      <path d="M13 12 6 6.5v11L13 12Z" fill="currentColor" />
      <path d="M18 12 11 6.5v11L18 12Z" fill="currentColor" />
      <rect x="17.8" y="6" width="2.2" height="12" rx="0.6" fill="currentColor" />
    </svg>
  );
}

export function IconVolumeOn(props: IconProps) {
  return (
    <svg {...svgProps(props)}>
      <path
        d="M4 9.5h3.2L12 6v12l-4.8-3.5H4V9.5Z"
        fill="currentColor"
      />
      <path
        d="M15.2 9.2a3.2 3.2 0 0 1 0 5.6M17.5 7a6 6 0 0 1 0 10"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function IconVolumeOff(props: IconProps) {
  return (
    <svg {...svgProps(props)}>
      <path
        d="M4 9.5h3.2L12 6v12l-4.8-3.5H4V9.5Z"
        fill="currentColor"
      />
      <path
        d="m15 9.5 5 5M20 9.5l-5 5"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function IconPrev(props: IconProps) {
  return (
    <svg {...svgProps({ ...props, size: props.size ?? 16 })}>
      <path
        d="M15 5 8 12l7 7"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function IconNext(props: IconProps) {
  return (
    <svg {...svgProps({ ...props, size: props.size ?? 16 })}>
      <path
        d="m9 5 7 7-7 7"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

/** Send selected item live (Go Live). */
export function IconGoLive(props: IconProps) {
  return (
    <svg {...svgProps({ ...props, size: props.size ?? 16 })}>
      <circle cx="12" cy="12" r="3.2" fill="currentColor" />
      <path
        d="M7.2 7.2a6.8 6.8 0 0 1 9.6 0M5 5a9.5 9.5 0 0 1 14 0"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
      />
    </svg>
  );
}

/** Soft tip / status — circle with “i”. */
export function IconInfo(props: IconProps) {
  return (
    <svg {...svgProps({ ...props, size: props.size ?? 16 })}>
      <circle
        cx="12"
        cy="12"
        r="9"
        stroke="currentColor"
        strokeWidth="1.7"
      />
      <path
        d="M12 10.5V17M12 7.2h.01"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
      />
    </svg>
  );
}

/** Warning tip — triangle with “!”. */
export function IconWarn(props: IconProps) {
  return (
    <svg {...svgProps({ ...props, size: props.size ?? 16 })}>
      <path
        d="M12 3.5 21 19.5H3L12 3.5Z"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinejoin="round"
      />
      <path
        d="M12 9.5v5M12 16.8h.01"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
      />
    </svg>
  );
}

/** Project / cast to output. */
export function IconProject(props: IconProps) {
  return (
    <svg {...svgProps({ ...props, size: props.size ?? 20 })}>
      <rect
        x="3"
        y="4"
        width="18"
        height="12"
        rx="1.5"
        stroke="currentColor"
        strokeWidth="1.7"
      />
      <path
        d="M8 20h8M12 16v4"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
      />
    </svg>
  );
}

/** Language / locale. */
export function IconLanguage(props: IconProps) {
  return (
    <svg {...svgProps({ ...props, size: props.size ?? 20 })}>
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.7" />
      <path
        d="M3.5 12h17M12 3c2.5 2.8 3.8 5.8 3.8 9s-1.3 6.2-3.8 9c-2.5-2.8-3.8-5.8-3.8-9s1.3-6.2 3.8-9Z"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function IconAbout(props: IconProps) {
  return (
    <svg {...svgProps({ ...props, size: props.size ?? 20 })}>
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.7" />
      <path
        d="M12 10.5V17M12 7.2h.01"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
      />
    </svg>
  );
}

/** Local media library (Biblioteca) — bookshelf. */
export function IconLibrary(props: IconProps) {
  return (
    <svg {...svgProps({ ...props, size: props.size ?? 20 })}>
      <path
        d="M4 5.5h3.2v13H4A1.5 1.5 0 0 1 2.5 17V7A1.5 1.5 0 0 1 4 5.5Z"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
      <path
        d="M7.2 5.5h3.5v13H7.2V5.5Z"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
      <path
        d="M10.7 6.2 19 4.5v14.2l-8.3 1.7V6.2Z"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
    </svg>
  );
}

/** Local AI subtitle agent — caption bars. */
export function IconSubtitles(props: IconProps) {
  return (
    <svg {...svgProps({ ...props, size: props.size ?? 20 })}>
      <rect
        x="3"
        y="5"
        width="18"
        height="14"
        rx="2"
        stroke="currentColor"
        strokeWidth="1.6"
        fill="none"
      />
      <path
        d="M6 14.5h7M6 11h12"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
    </svg>
  );
}
