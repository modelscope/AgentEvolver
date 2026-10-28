/** Persist last Control session code so a refresh can resume (same browser tab). */

const KEY = "promedia.control.session.v1";

export type PersistedControlSession = {
  code: string;
  accessToken?: string;
  savedAt: number;
};

export function loadPersistedControlSession(): PersistedControlSession | null {
  try {
    const raw = sessionStorage.getItem(KEY);
    if (!raw) {
      return null;
    }
    const parsed = JSON.parse(raw) as PersistedControlSession;
    if (!parsed?.code || typeof parsed.code !== "string") {
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}

export function savePersistedControlSession(opts: {
  code: string;
  accessToken?: string;
}): void {
  try {
    const payload: PersistedControlSession = {
      code: opts.code,
      accessToken: opts.accessToken,
      savedAt: Date.now(),
    };
    sessionStorage.setItem(KEY, JSON.stringify(payload));
  } catch {
    // private mode / quota
  }
}

export function clearPersistedControlSession(): void {
  try {
    sessionStorage.removeItem(KEY);
  } catch {
    // ignore
  }
}
