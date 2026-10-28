import type { CSSProperties } from "react";
import {
  DEFAULT_THEME_ID,
  resolveTheme,
  themeToCssVars,
  type ThemeId,
  type ThemeOverrides,
} from "@promedia/shared";

/** Build inline style with --pm-* vars for a projection surface. */
export function themeSurfaceStyle(
  themeId: ThemeId | null | undefined,
  themeOverrides: ThemeOverrides | null | undefined,
): CSSProperties {
  const resolved = resolveTheme(themeId ?? DEFAULT_THEME_ID, themeOverrides);
  return themeToCssVars(resolved) as CSSProperties;
}
