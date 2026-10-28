/** Preset ids (subtarea 8). */
export type ThemeId = "oscuro" | "claro" | "acento";

type ThemeAlign = "left" | "center" | "right";

export const THEME_IDS: readonly ThemeId[] = [
  "oscuro",
  "claro",
  "acento",
] as const;

export const DEFAULT_THEME_ID: ThemeId = "oscuro";

/**
 * Live Output / En vivo / lyric-cue transitions (TEMA menu).
 * Primary set mirrors FreeShow (jimena_hidalgo__ Reel De22bDYtucC):
 * none | blur | fade | crossfade | fly | scale | slide | spin.
 * ProMedia extras: motionBlur, pixelate. softScale = legacy alias of scale.
 */
export type TransitionEffect =
  | "none"
  | "fade"
  | "crossfade"
  | "blur"
  | "scale"
  | "slide"
  | "fly"
  | "spin"
  | "motionBlur"
  | "pixelate"
  /** @deprecated use `scale` — kept for older session snapshots */
  | "softScale";

/** Menu order — FreeShow defaults first, then ProMedia extras. */
export const TRANSITION_EFFECTS: readonly TransitionEffect[] = [
  "none",
  "fade",
  "crossfade",
  "blur",
  "scale",
  "slide",
  "fly",
  "spin",
  "motionBlur",
  "pixelate",
] as const;

/**
 * FreeShow-style default (De22bDYtucC tip): soft fade — not a hard cut.
 */
export const DEFAULT_TRANSITION_EFFECT: TransitionEffect = "fade";

/** Fixed transition length on Output / En vivo / lyric cues (ms). */
export const TRANSITION_DURATION_MS = 550;

export function isTransitionEffect(value: unknown): value is TransitionEffect {
  if (typeof value !== "string") return false;
  if ((TRANSITION_EFFECTS as readonly string[]).includes(value)) return true;
  return value === "softScale";
}

/** CSS class suffix (`pm-tx-${id}`) — maps legacy softScale → scale. */
export function transitionCssId(effect: TransitionEffect): string {
  if (effect === "softScale") return "scale";
  return effect;
}

export type ThemeFontSize = "sm" | "md" | "lg" | "xl";

export type ThemeFontFamily = "sans" | "serif";

/** Deltas over a preset; editor writes here (preset base kept). */
export type ThemeOverrides = {
  bg?: string;
  fg?: string;
  align?: ThemeAlign;
  fontFamily?: ThemeFontFamily;
  fontSize?: ThemeFontSize;
};

/**
 * Stage background behind projected content (Output + En vivo + Vista previa).
 * Solid uses theme bg color; media uses a built-in looping video pack or a session asset.
 */
export type ThemeBackgroundMode = "solid" | "media";

export type ThemeBackgroundPackId =
  | "aurora"
  | "embers"
  | "ocean"
  | "softgrid"
  | "abstract-blue";

export type ThemeBackgroundMediaKind = "video" | "image";

export type ThemeBackground = {
  mode: ThemeBackgroundMode;
  /** Built-in looping video pack under public/theme-backgrounds/ (mode=media). */
  packId?: ThemeBackgroundPackId | null;
  /** Uploaded session asset id (mode=media). */
  assetId?: string | null;
  /** Hint for rendering uploaded asset. */
  mediaKind?: ThemeBackgroundMediaKind | null;
  /**
   * Black dots pattern overlay to darken busy/video backgrounds
   * so light text stays readable. Applied automatically with Karaoke agent.
   */
  dotsOverlay?: boolean;
};

export const THEME_BACKGROUND_PACK_IDS: readonly ThemeBackgroundPackId[] = [
  "aurora",
  "embers",
  "ocean",
  "softgrid",
  "abstract-blue",
] as const;

/** Vite public/ path for a built-in theme pack looping video. */
export function themeBackgroundPackSrc(
  packId: ThemeBackgroundPackId,
  format: "mp4" | "webm" = "mp4",
): string {
  return `/theme-backgrounds/${packId}.${format}`;
}

export const DEFAULT_THEME_BACKGROUND: ThemeBackground = {
  mode: "solid",
  dotsOverlay: false,
};

export function isThemeBackgroundPackId(
  value: unknown,
): value is ThemeBackgroundPackId {
  return (
    typeof value === "string" &&
    (THEME_BACKGROUND_PACK_IDS as readonly string[]).includes(value)
  );
}

export function isThemeBackgroundMediaKind(
  value: unknown,
): value is ThemeBackgroundMediaKind {
  return value === "video" || value === "image";
}

/** Normalize / sanitize theme background from WS or storage. */
export function normalizeThemeBackground(
  raw: unknown,
): ThemeBackground {
  if (!raw || typeof raw !== "object") {
    return { ...DEFAULT_THEME_BACKGROUND };
  }
  const o = raw as Record<string, unknown>;
  const dotsOverlay = o.dotsOverlay === true;
  const mode: ThemeBackgroundMode = o.mode === "media" ? "media" : "solid";
  if (mode === "solid") {
    return { mode: "solid", dotsOverlay };
  }
  const packId = isThemeBackgroundPackId(o.packId) ? o.packId : null;
  const assetId =
    typeof o.assetId === "string" && o.assetId.trim()
      ? o.assetId.trim()
      : null;
  const mediaKind = isThemeBackgroundMediaKind(o.mediaKind)
    ? o.mediaKind
    : null;
  if (!packId && !assetId) {
    return { mode: "solid", dotsOverlay };
  }
  return {
    mode: "media",
    packId: assetId ? null : packId,
    assetId,
    mediaKind: assetId ? mediaKind ?? "video" : null,
    dotsOverlay,
  };
}

export type ResolvedTheme = {
  themeId: ThemeId;
  bg: string;
  fg: string;
  align: ThemeAlign;
  fontFamily: ThemeFontFamily;
  fontSize: ThemeFontSize;
  /** Soft accent (borde / título); only meaningful for acento preset. */
  accent?: string;
};

export type ThemePreset = {
  id: ThemeId;
  labelKey: string;
  values: Omit<ResolvedTheme, "themeId">;
};

const FONT_STACKS: Record<ThemeFontFamily, string> = {
  sans: '"Segoe UI", system-ui, -apple-system, sans-serif',
  serif: 'Georgia, "Times New Roman", serif',
};

const FONT_SIZES: Record<ThemeFontSize, string> = {
  sm: "clamp(1.25rem, 3vw, 2rem)",
  md: "clamp(1.75rem, 4.5vw, 3.25rem)",
  lg: "clamp(2.25rem, 5.5vw, 4rem)",
  xl: "clamp(2.75rem, 7vw, 5rem)",
};

export const THEME_PRESETS: readonly ThemePreset[] = [
  {
    id: "oscuro",
    labelKey: "theme.oscuro",
    values: {
      bg: "#0a0e14",
      fg: "#f2f5f8",
      align: "center",
      fontFamily: "sans",
      fontSize: "md",
    },
  },
  {
    id: "claro",
    labelKey: "theme.claro",
    values: {
      bg: "#f4f6f8",
      fg: "#12151a",
      align: "center",
      fontFamily: "sans",
      fontSize: "md",
    },
  },
  {
    id: "acento",
    labelKey: "theme.acento",
    values: {
      bg: "#0c1219",
      fg: "#e8eef6",
      align: "center",
      fontFamily: "sans",
      fontSize: "md",
      accent: "#3d8bfd",
    },
  },
] as const;

export function isThemeId(value: unknown): value is ThemeId {
  return (
    typeof value === "string" &&
    (THEME_IDS as readonly string[]).includes(value)
  );
}

export function isThemeFontSize(value: unknown): value is ThemeFontSize {
  return value === "sm" || value === "md" || value === "lg" || value === "xl";
}

export function isThemeFontFamily(value: unknown): value is ThemeFontFamily {
  return value === "sans" || value === "serif";
}

export function getThemePreset(id: ThemeId): ThemePreset {
  return THEME_PRESETS.find((p) => p.id === id) ?? THEME_PRESETS[0]!;
}

export function normalizeThemeOverrides(
  raw: unknown,
): ThemeOverrides | null {
  if (raw === null || raw === undefined) {
    return null;
  }
  if (!raw || typeof raw !== "object") {
    return null;
  }
  const o = raw as Record<string, unknown>;
  const next: ThemeOverrides = {};
  if (typeof o.bg === "string" && o.bg.trim()) {
    next.bg = o.bg.trim();
  }
  if (typeof o.fg === "string" && o.fg.trim()) {
    next.fg = o.fg.trim();
  }
  if (o.align === "left" || o.align === "center" || o.align === "right") {
    next.align = o.align;
  }
  if (isThemeFontFamily(o.fontFamily)) {
    next.fontFamily = o.fontFamily;
  }
  if (isThemeFontSize(o.fontSize)) {
    next.fontSize = o.fontSize;
  }
  return Object.keys(next).length > 0 ? next : null;
}

export function resolveTheme(
  themeId: ThemeId | null | undefined,
  overrides?: ThemeOverrides | null,
): ResolvedTheme {
  const id = isThemeId(themeId) ? themeId : DEFAULT_THEME_ID;
  const base = getThemePreset(id).values;
  const o = overrides ?? {};
  return {
    themeId: id,
    bg: o.bg ?? base.bg,
    fg: o.fg ?? base.fg,
    align: o.align ?? base.align,
    fontFamily: o.fontFamily ?? base.fontFamily,
    fontSize: o.fontSize ?? base.fontSize,
    accent: base.accent,
  };
}

/** CSS custom properties for projection surfaces. */
export function themeToCssVars(
  theme: ResolvedTheme,
): Record<string, string> {
  const vars: Record<string, string> = {
    "--pm-bg": theme.bg,
    "--pm-fg": theme.fg,
    "--pm-align": theme.align,
    "--pm-font-family": FONT_STACKS[theme.fontFamily],
    "--pm-font-size": FONT_SIZES[theme.fontSize],
  };
  if (theme.accent) {
    vars["--pm-accent"] = theme.accent;
  }
  return vars;
}

export function fontFamilyCss(family: ThemeFontFamily): string {
  return FONT_STACKS[family];
}

export function fontSizeCss(size: ThemeFontSize): string {
  return FONT_SIZES[size];
}
