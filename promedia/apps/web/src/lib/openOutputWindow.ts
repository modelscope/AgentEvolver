/**
 * Open / focus the Output window, prefer secondary display (Window Management API),
 * and ask the Output page to enter fullscreen when the browser allows it.
 *
 * Limits: browsers block popup + cross-monitor placement without a user gesture /
 * permission; fullscreen almost always needs a gesture in the Output window.
 */

export const OUTPUT_WINDOW_NAME = "promedia-output";

export type OutputPlacement = {
  left: number;
  top: number;
  width: number;
  height: number;
  /** True when coordinates came from getScreenDetails() on a non-current screen. */
  onSecondary: boolean;
  /** Window Management API available and used successfully. */
  usedScreenDetails: boolean;
};

export type OpenOutputResult = {
  url: string;
  opened: boolean;
  focusedExisting: boolean;
  placement: OutputPlacement | null;
  /** Popup blocked — caller should show the URL. */
  popupBlocked: boolean;
};

type ScreenDetailedLike = {
  availLeft: number;
  availTop: number;
  availWidth: number;
  availHeight: number;
  isPrimary?: boolean;
};

type ScreenDetailsLike = {
  screens: readonly ScreenDetailedLike[];
  currentScreen: ScreenDetailedLike;
};

function screensApi():
  | { getScreenDetails: () => Promise<ScreenDetailsLike> }
  | null {
  const w = window as Window & {
    getScreenDetails?: () => Promise<ScreenDetailsLike>;
  };
  return typeof w.getScreenDetails === "function"
    ? { getScreenDetails: () => w.getScreenDetails!() }
    : null;
}

/** Pick a screen that is not the one hosting Control, else any non-primary. */
export function pickSecondaryScreen(
  details: ScreenDetailsLike,
): ScreenDetailedLike | null {
  const screens = details.screens ?? [];
  if (screens.length < 2) {
    return null;
  }
  const current = details.currentScreen;
  return (
    screens.find((s) => s !== current) ??
    screens.find((s) => s.isPrimary === false) ??
    null
  );
}

/**
 * Resolve where to place Output. May prompt for Window Management permission (Chrome).
 * Call from a user-gesture handler when possible.
 */
export async function resolveOutputPlacement(): Promise<OutputPlacement | null> {
  const api = screensApi();
  if (!api) {
    return null;
  }
  try {
    const details = await api.getScreenDetails();
    const secondary = pickSecondaryScreen(details);
    const target = secondary ?? details.currentScreen;
    return {
      left: Math.round(target.availLeft),
      top: Math.round(target.availTop),
      width: Math.max(320, Math.round(target.availWidth)),
      height: Math.max(240, Math.round(target.availHeight)),
      onSecondary: Boolean(secondary),
      usedScreenDetails: true,
    };
  } catch {
    return null;
  }
}

function featuresFromPlacement(p: OutputPlacement): string {
  // Do NOT include noopener — we need the Window handle to navigate / resize.
  return [
    `left=${p.left}`,
    `top=${p.top}`,
    `width=${p.width}`,
    `height=${p.height}`,
  ].join(",");
}

export function codeOutputUrl(
  code: string,
  opts?: { autofullscreen?: boolean },
): string {
  const u = new URL(`${window.location.origin}/output`);
  u.searchParams.set("code", code);
  if (opts?.autofullscreen) {
    u.searchParams.set("autofullscreen", "1");
  }
  return u.toString();
}

export function opaqueOutputUrl(
  accessToken: string,
  opts?: { autofullscreen?: boolean },
): string {
  const u = new URL(`${window.location.origin}/output`);
  u.searchParams.set("token", accessToken);
  if (opts?.autofullscreen) {
    u.searchParams.set("autofullscreen", "1");
  }
  return u.toString();
}

function tryMoveResize(win: Window, p: OutputPlacement): void {
  try {
    win.moveTo(p.left, p.top);
  } catch {
    // Browser policy / not yet loaded
  }
  try {
    win.resizeTo(p.width, p.height);
  } catch {
    // ignore
  }
}

function postAutofullscreen(win: Window): void {
  try {
    win.postMessage(
      { type: "promedia.requestFullscreen" },
      window.location.origin,
    );
  } catch {
    // ignore
  }
}

function navigateAndFinish(
  win: Window,
  url: string,
  placement: OutputPlacement | null,
  focusedExisting: boolean,
): OpenOutputResult {
  try {
    win.location.replace(url);
  } catch {
    try {
      win.location.href = url;
    } catch {
      // ignore
    }
  }
  if (placement) {
    tryMoveResize(win, placement);
    window.setTimeout(() => tryMoveResize(win, placement), 300);
  }
  window.setTimeout(() => postAutofullscreen(win), 400);
  window.setTimeout(() => postAutofullscreen(win), 1200);
  try {
    win.focus();
  } catch {
    // ignore
  }
  return {
    url,
    opened: true,
    focusedExisting,
    placement,
    popupBlocked: false,
  };
}

/**
 * Open or focus Output for a known session code.
 * Prefer calling from a click handler so popup + screen permission work.
 */
export async function openOutputWindow(
  code: string,
  opts?: { autofullscreen?: boolean; preferSecondary?: boolean },
): Promise<OpenOutputResult> {
  const autofullscreen = opts?.autofullscreen !== false;
  const preferSecondary = opts?.preferSecondary !== false;
  const url = codeOutputUrl(code, { autofullscreen });

  // Open under the user gesture first — awaiting getScreenDetails() before
  // window.open often loses transient activation and gets the popup blocked.
  const win = window.open(url, OUTPUT_WINDOW_NAME, "width=1280,height=720");
  if (!win) {
    return {
      url,
      opened: false,
      focusedExisting: false,
      placement: null,
      popupBlocked: true,
    };
  }

  let placement: OutputPlacement | null = null;
  if (preferSecondary) {
    placement = await resolveOutputPlacement();
  }

  if (placement) {
    tryMoveResize(win, placement);
    window.setTimeout(() => tryMoveResize(win, placement), 300);
  }
  if (autofullscreen) {
    window.setTimeout(() => postAutofullscreen(win), 400);
    window.setTimeout(() => postAutofullscreen(win), 1200);
  }
  try {
    win.focus();
  } catch {
    // ignore
  }

  return {
    url,
    opened: true,
    focusedExisting: false,
    placement,
    popupBlocked: false,
  };
}

/**
 * Primary CTA: ensure session, open Output on secondary screen when possible,
 * request autofullscreen. Opens `about:blank` first so `await ensureSession()`
 * does not lose the popup under the user gesture.
 */
export async function startProjection(opts: {
  existingCode: string | null;
  ensureSession: () => Promise<string>;
}): Promise<OpenOutputResult & { code: string }> {
  // Session already known — open/focus Output directly (still under the click).
  if (opts.existingCode) {
    const result = await openOutputWindow(opts.existingCode, {
      autofullscreen: true,
      preferSecondary: true,
    });
    return { code: opts.existingCode, ...result };
  }

  // 1) Reserve the named window synchronously under the user gesture
  //    (before await createSession — otherwise the popup is blocked).
  const popup = window.open(
    "about:blank",
    OUTPUT_WINDOW_NAME,
    "width=1280,height=720",
  );

  // 2) Window Management API (may prompt). Still often allowed after open.
  let placement: OutputPlacement | null = null;
  try {
    placement = await resolveOutputPlacement();
  } catch {
    placement = null;
  }

  // 3) Create/resume session.
  const code = await opts.ensureSession();
  const url = codeOutputUrl(code, { autofullscreen: true });
  const features = placement ? featuresFromPlacement(placement) : undefined;

  if (!popup) {
    const retry = window.open(url, OUTPUT_WINDOW_NAME, features);
    if (!retry) {
      return {
        code,
        url,
        opened: false,
        focusedExisting: false,
        placement,
        popupBlocked: true,
      };
    }
    return { code, ...navigateAndFinish(retry, url, placement, false) };
  }

  return { code, ...navigateAndFinish(popup, url, placement, false) };
}
