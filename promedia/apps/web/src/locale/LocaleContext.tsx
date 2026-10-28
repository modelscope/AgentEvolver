import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import {
  DEFAULT_LOCALE,
  LOCALES,
  hasLocale,
  t as translate,
  type Locale,
} from "@promedia/i18n";

const STORAGE_KEY = "promedia.control.locale";

function readStoredLocale(): Locale {
  if (typeof window === "undefined") {
    return DEFAULT_LOCALE;
  }
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (raw && hasLocale(raw)) {
      return raw;
    }
  } catch {
    // ignore storage errors
  }
  return DEFAULT_LOCALE;
}

export type LocaleContextValue = {
  locale: Locale;
  setLocale: (locale: Locale) => void;
  t: (key: string) => string;
  locales: readonly Locale[];
};

const LocaleContext = createContext<LocaleContextValue | null>(null);

export function LocaleProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>(readStoredLocale);

  const setLocale = useCallback((next: Locale) => {
    setLocaleState(next);
    try {
      window.localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // ignore
    }
  }, []);

  const value = useMemo<LocaleContextValue>(
    () => ({
      locale,
      setLocale,
      t: (key: string) => translate(locale, key),
      locales: LOCALES,
    }),
    [locale, setLocale],
  );

  return (
    <LocaleContext.Provider value={value}>{children}</LocaleContext.Provider>
  );
}

export function useLocale(): LocaleContextValue {
  const ctx = useContext(LocaleContext);
  if (!ctx) {
    // Fallback for Output / tests outside provider — Spanish default.
    return {
      locale: DEFAULT_LOCALE,
      setLocale: () => undefined,
      t: (key: string) => translate(DEFAULT_LOCALE, key),
      locales: LOCALES,
    };
  }
  return ctx;
}

/** Short labels for the language toolbar control. */
export const LOCALE_SHORT: Record<Locale, string> = {
  es: "ES",
  en: "EN",
  pt: "PT",
  fr: "FR",
  "zh-CN": "ZH",
};
