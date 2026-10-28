/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_HUB_URL?: string;
  /** Direct WebSocket base, e.g. ws://localhost:8787 (dev default). */
  readonly VITE_HUB_WS_URL?: string;
  /** Google Ad Manager network code (numeric). Both GAM vars required for live ads. */
  readonly VITE_GAM_NETWORK_CODE?: string;
  /** GAM ad unit path under the network, e.g. promedia_control_header */
  readonly VITE_GAM_AD_UNIT?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

/** Chrome Window Management API (optional). */
interface ScreenDetailed extends Screen {
  readonly availLeft: number;
  readonly availTop: number;
  readonly left: number;
  readonly top: number;
  readonly isPrimary: boolean;
  readonly isInternal: boolean;
  readonly devicePixelRatio: number;
  readonly label: string;
}

interface ScreenDetails {
  readonly screens: ReadonlyArray<ScreenDetailed>;
  readonly currentScreen: ScreenDetailed;
}

interface Window {
  getScreenDetails?: () => Promise<ScreenDetails>;
}
