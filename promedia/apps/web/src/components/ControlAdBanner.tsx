import { useEffect, useRef, useState } from "react";
import { useLocale } from "../locale/LocaleContext";
import {
  GAM_HEADER_SIZES,
  getGamConfig,
  type GamConfig,
} from "../lib/gam";
import { StatusTip } from "./StatusTip";

const GPT_SCRIPT_SRC = "https://securepubads.g.doubleclick.net/tag/js/gpt.js";

type GptSize = [number, number] | "fluid";

type GoogletagSlot = {
  addService: (service: unknown) => GoogletagSlot;
  defineSizeMapping: (mapping: unknown) => GoogletagSlot;
};

type GoogletagApi = {
  cmd: Array<() => void>;
  apiReady?: boolean;
  defineSlot: (
    adUnitPath: string,
    size: GptSize | GptSize[],
    optDiv?: string,
  ) => GoogletagSlot | null;
  sizeMapping: () => {
    addSize: (
      viewport: [number, number],
      sizes: GptSize | GptSize[],
    ) => ReturnType<GoogletagApi["sizeMapping"]>;
    build: () => unknown;
  };
  pubads: () => {
    enableSingleRequest: () => void;
    collapseEmptyDivs: () => void;
    setForceSafeFrame: (force: boolean) => void;
  };
  enableServices: () => void;
  display: (divOrSlot: string) => void;
  destroySlots: (slots?: GoogletagSlot[]) => boolean;
};

declare global {
  interface Window {
    googletag?: GoogletagApi;
  }
}

let gptScriptPromise: Promise<void> | null = null;
let headerSlot: GoogletagSlot | null = null;
let pubadsConfigured = false;

function loadGptScript(): Promise<void> {
  if (typeof window === "undefined") {
    return Promise.resolve();
  }
  window.googletag =
    window.googletag ?? ({ cmd: [] } as unknown as GoogletagApi);
  if (window.googletag.apiReady) {
    return Promise.resolve();
  }
  if (gptScriptPromise) {
    return gptScriptPromise;
  }
  const existing = document.getElementById("promedia-gpt-js");
  if (existing) {
    gptScriptPromise = new Promise((resolve) => {
      window.googletag!.cmd.push(() => resolve());
    });
    return gptScriptPromise;
  }
  gptScriptPromise = new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.id = "promedia-gpt-js";
    script.async = true;
    script.src = GPT_SCRIPT_SRC;
    script.onload = () => {
      window.googletag!.cmd.push(() => resolve());
    };
    script.onerror = () => {
      gptScriptPromise = null;
      reject(new Error("gpt_load_failed"));
    };
    document.head.appendChild(script);
  });
  return gptScriptPromise;
}

function defineHeaderSlot(config: GamConfig): void {
  const googletag = window.googletag;
  if (!googletag) {
    return;
  }
  googletag.cmd.push(() => {
    const el = document.getElementById(config.slotElementId);
    if (!el) {
      return;
    }
    if (headerSlot) {
      googletag.destroySlots([headerSlot]);
      headerSlot = null;
    }

    const sizes: GptSize[] = [
      ...GAM_HEADER_SIZES.map(([w, h]) => [w, h] as [number, number]),
      "fluid",
    ];

    const mapping = googletag
      .sizeMapping()
      .addSize([1024, 0], [
        [970, 90],
        [728, 90],
        "fluid",
      ])
      .addSize([728, 0], [
        [728, 90],
        "fluid",
      ])
      .addSize([0, 0], [
        [320, 50],
        [320, 100],
        "fluid",
      ])
      .build();

    const slot = googletag.defineSlot(
      config.slotPath,
      sizes,
      config.slotElementId,
    );
    if (!slot) {
      return;
    }
    slot.defineSizeMapping(mapping);
    slot.addService(googletag.pubads());

    if (!pubadsConfigured) {
      // SafeFrame only — no out-of-page overlays that could cover Control UI.
      googletag.pubads().setForceSafeFrame(true);
      googletag.pubads().collapseEmptyDivs();
      googletag.pubads().enableSingleRequest();
      googletag.enableServices();
      pubadsConfigured = true;
    }

    headerSlot = slot;
    googletag.display(config.slotElementId);
  });
}

function destroyHeaderSlot(): void {
  const googletag = window.googletag;
  if (!googletag || !headerSlot) {
    return;
  }
  const slot = headerSlot;
  headerSlot = null;
  googletag.cmd.push(() => {
    googletag.destroySlots([slot]);
  });
}

/**
 * Header advertising slot: GPT when env is set, dashed placeholder otherwise.
 * Contained (overflow + low z-index) so creatives cannot cover Guion / Preview / Live.
 */
export function ControlAdBanner() {
  const { t } = useLocale();

  const config = getGamConfig();
  const [loadError, setLoadError] = useState(false);
  const mounted = useRef(true);
  const slotPath = config?.slotPath ?? null;
  const slotElementId = config?.slotElementId ?? null;

  useEffect(() => {
    mounted.current = true;
    if (!slotPath || !slotElementId) {
      return;
    }
    const active: GamConfig = {
      networkCode: "",
      adUnit: "",
      slotPath,
      slotElementId,
    };
    setLoadError(false);
    void loadGptScript()
      .then(() => {
        if (!mounted.current) {
          return;
        }
        defineHeaderSlot(active);
      })
      .catch(() => {
        if (mounted.current) {
          setLoadError(true);
        }
      });
    return () => {
      mounted.current = false;
      destroyHeaderSlot();
    };
  }, [slotPath, slotElementId]);

  if (!config) {
    return (
      <div
        className="control-ad-slot control-ad-slot-placeholder"
        role="complementary"
        aria-label={t("control.adSlotLabel")}
      >
        <span className="control-ad-placeholder-tip">
          <StatusTip
            tone="info"
            label={t("control.adSlotLabel")}
            message={`${t("control.adSlotPlaceholder")} — ${t("control.adSlotHint")}`}
          />
        </span>
      </div>
    );
  }

  return (
    <div
      className="control-ad-slot"
      role="complementary"
      aria-label={t("control.adSlotLabel")}
    >
      {loadError ? (
        <div className="control-ad-slot-placeholder control-ad-slot-error">
          <span className="control-ad-placeholder-tip">
            <StatusTip
              tone="warn"
              label={t("control.statusTipWarn")}
              message={t("control.adSlotLoadError")}
            />
          </span>
        </div>
      ) : null}
      <div
        id={config.slotElementId}
        className="control-ad-gpt"
        style={{ minWidth: "320px", minHeight: "50px" }}
      />
    </div>
  );
}
