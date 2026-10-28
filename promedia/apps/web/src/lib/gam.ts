/**
 * Google Ad Manager (GPT) config from Vite env.
 * When network code + ad unit are unset, the Control header shows a placeholder.
 */

export type GamConfig = {
  /** Numeric GAM network code, e.g. "123456789". */
  networkCode: string;
  /** Ad unit path under the network, e.g. "promedia_control_header" or "site/header". */
  adUnit: string;
  /** Full GPT ad unit path: /{networkCode}/{adUnit}. */
  slotPath: string;
  /** Stable DOM id for googletag.display. */
  slotElementId: string;
};

/** Header leaderboard-style sizes (documented for Julio / GAM inventory). */
export const GAM_HEADER_SIZES = [
  [970, 90],
  [728, 90],
  [320, 50],
  [320, 100],
] as const;

export const GAM_HEADER_SIZE_NOTES = [
  "970×90 — large desktop leaderboard",
  "728×90 — standard leaderboard",
  "320×50 — mobile banner",
  "320×100 — large mobile banner",
  "fluid — GPT fluid size when the creative supports it",
] as const;

const SLOT_ELEMENT_ID = "promedia-control-gam-header";

function trimEnv(value: string | undefined): string {
  return (value ?? "").trim();
}

/**
 * Returns GAM config when both env vars are set; otherwise null (placeholder UI).
 * - VITE_GAM_NETWORK_CODE
 * - VITE_GAM_AD_UNIT
 */
export function getGamConfig(): GamConfig | null {
  const networkCode = trimEnv(import.meta.env.VITE_GAM_NETWORK_CODE);
  const adUnit = trimEnv(import.meta.env.VITE_GAM_AD_UNIT).replace(
    /^\/+/,
    "",
  );
  if (!networkCode || !adUnit) {
    return null;
  }
  // Allow either "unit" or "/123/unit" pasted by mistake for AD_UNIT.
  const slotPath = adUnit.startsWith(`${networkCode}/`)
    ? `/${adUnit}`
    : `/${networkCode}/${adUnit}`;
  return {
    networkCode,
    adUnit,
    slotPath,
    slotElementId: SLOT_ELEMENT_ID,
  };
}

export function isGamConfigured(): boolean {
  return getGamConfig() !== null;
}
