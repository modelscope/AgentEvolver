import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { getItemType } from "@promedia/item-types";
import { DEFAULT_LOCALE, t } from "@promedia/i18n";
import type { SceneMode, ThemeId, TransitionEffect } from "@promedia/shared";
import {
  DEFAULT_THEME_BACKGROUND,
  DEFAULT_THEME_ID,
  DEFAULT_TRANSITION_EFFECT,
  emptySubtitleAgentState,
  normalizeThemeBackground,
} from "@promedia/shared";
import { LiveItemLayer } from "../components/LiveItemLayer";
import { SceneOverlay } from "../components/SceneOverlay";
import {
  StageBackground,
  StageDotsOverlay,
} from "../components/StageBackground";
import { SubtitleOverlay } from "../components/SubtitleOverlay";
import { useSessionConnection } from "../hooks/useSessionConnection";
import { buildAssetUrl } from "../lib/assets";
import { themeSurfaceStyle } from "../lib/theme";

const locale = DEFAULT_LOCALE;

/** Product Output — live + sceneMode overlays (subtarea 7). */
export function OutputPage() {
  const [params] = useSearchParams();
  const {
    hasSession,
    wsStatus,
    error,
    join,
    snapshot,
    sessionId,
    wsTicket,
    setMediaClock,
  } = useSessionConnection("output");
  const stageRef = useRef<HTMLDivElement>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [manualCode, setManualCode] = useState("");
  const autoJoinStarted = useRef(false);
  const autofullscreenTried = useRef(false);
  const joinRef = useRef(join);
  joinRef.current = join;

  const queryCode = (params.get("code") ?? "").trim().toUpperCase();
  const queryToken =
    (params.get("token") ?? params.get("accessToken") ?? "").trim();
  const wantAutofullscreen =
    params.get("autofullscreen") === "1" ||
    params.get("autofullscreen") === "true";

  const liveItem = useMemo(() => {
    const id = snapshot?.live.currentItemId;
    if (!id || !snapshot) {
      return null;
    }
    return snapshot.playlist.items.find((i) => i.id === id) ?? null;
  }, [snapshot]);

  const liveSlideIndex = snapshot?.live.currentSlideIndex ?? 0;
  const mediaClock = snapshot?.live.mediaClock ?? null;
  const sceneMode: SceneMode = snapshot?.live.sceneMode ?? "normal";
  const themeId: ThemeId = snapshot?.live.themeId ?? DEFAULT_THEME_ID;
  const themeOverrides = snapshot?.live.themeOverrides ?? null;
  const themeBackground = normalizeThemeBackground(
    snapshot?.live.themeBackground ?? DEFAULT_THEME_BACKGROUND,
  );
  const dotsOverlay = themeBackground.dotsOverlay === true;
  /** Dots on the video plane (inside item layer) so captions stay above. */
  const videoPlaneDots = dotsOverlay && liveItem?.type === "video";
  const transitionEffect: TransitionEffect =
    snapshot?.live.transitionEffect ?? DEFAULT_TRANSITION_EFFECT;
  const subtitleAgent =
    snapshot?.live.subtitleAgent ?? emptySubtitleAgentState();
  const themeStyle = themeSurfaceStyle(themeId, themeOverrides);
  const liveItemKey = liveItem?.id ?? "__empty__";
  const OutputRenderer = liveItem
    ? getItemType(liveItem.type)?.OutputRenderer
    : undefined;
  const themeBgAssetSrc =
    themeBackground.mode === "media" &&
    themeBackground.assetId &&
    sessionId &&
    wsTicket
      ? buildAssetUrl(sessionId, themeBackground.assetId, wsTicket)
      : undefined;

  const assetSrc = (() => {
    if (!liveItem || !sessionId || !wsTicket) {
      return undefined;
    }
    if (liveItem.type === "image") {
      return buildAssetUrl(sessionId, liveItem.payload.assetId, wsTicket);
    }
    if (liveItem.type === "presentation") {
      const slide =
        liveItem.payload.slides[liveSlideIndex] ?? liveItem.payload.slides[0];
      if (!slide) {
        return undefined;
      }
      return buildAssetUrl(sessionId, slide.assetId, wsTicket);
    }
    if (liveItem.type === "video" && liveItem.payload.sourceKind === "file") {
      return buildAssetUrl(sessionId, liveItem.payload.assetId, wsTicket);
    }
    if (liveItem.type === "audio") {
      return buildAssetUrl(sessionId, liveItem.payload.assetId, wsTicket);
    }
    return undefined;
  })();

  useEffect(() => {
    const onFsChange = () => {
      setIsFullscreen(Boolean(document.fullscreenElement));
    };
    document.addEventListener("fullscreenchange", onFsChange);
    return () => document.removeEventListener("fullscreenchange", onFsChange);
  }, []);

  useEffect(() => {
    if (autoJoinStarted.current || hasSession) {
      return;
    }
    if (queryCode) {
      autoJoinStarted.current = true;
      void joinRef.current({ code: queryCode });
      return;
    }
    if (queryToken) {
      autoJoinStarted.current = true;
      void joinRef.current({ accessToken: queryToken });
    }
  }, [queryCode, queryToken, hasSession]);

  // WS auto-reconnect lives in useSessionConnection (shared with Control).

  const enterFullscreen = async () => {
    const el = stageRef.current;
    if (!el) {
      return;
    }
    try {
      if (el.requestFullscreen) {
        await el.requestFullscreen();
      }
    } catch {
      // Browser denied — F11 / button hint remains visible
    }
  };

  const exitFullscreen = async () => {
    try {
      if (document.fullscreenElement) {
        await document.exitFullscreen();
      }
    } catch {
      // ignore
    }
  };

  // postMessage from Control — best-effort fullscreen (needs gesture in many browsers).
  useEffect(() => {
    const onMessage = (ev: MessageEvent) => {
      if (ev.origin !== window.location.origin) {
        return;
      }
      const data = ev.data as { type?: string } | null;
      if (data?.type !== "promedia.requestFullscreen") {
        return;
      }
      if (document.fullscreenElement) {
        return;
      }
      const el = stageRef.current;
      if (el?.requestFullscreen) {
        void el.requestFullscreen().catch(() => undefined);
      }
    };
    window.addEventListener("message", onMessage);
    return () => window.removeEventListener("message", onMessage);
  }, []);

  // ?autofullscreen=1 — try a few times after load / join.
  useEffect(() => {
    if (!wantAutofullscreen || autofullscreenTried.current) {
      return;
    }
    if (!hasSession && wsStatus === "connecting") {
      return;
    }
    autofullscreenTried.current = true;
    const timers = [200, 600, 1500].map((ms) =>
      window.setTimeout(() => {
        if (document.fullscreenElement) {
          return;
        }
        const el = stageRef.current;
        if (el?.requestFullscreen) {
          void el.requestFullscreen().catch(() => undefined);
        }
      }, ms),
    );
    return () => {
      for (const id of timers) {
        window.clearTimeout(id);
      }
    };
  }, [wantAutofullscreen, hasSession, wsStatus]);

  const showJoinForm = !hasSession && !queryCode && !queryToken;

  return (
    <div className="output-shell">
      <div
        ref={stageRef}
        className={
          isFullscreen
            ? "output-stage output-stage-fs pm-theme"
            : "output-stage pm-theme"
        }
        style={themeStyle}
        data-theme={themeId}
      >
        <StageBackground
          background={themeBackground}
          assetSrc={themeBgAssetSrc}
          showDots={!videoPlaneDots}
        />
        {wsStatus === "connecting" && !hasSession ? (
          <p className="output-waiting">{t(locale, "output.joining")}</p>
        ) : error && !hasSession ? (
          <p className="output-waiting output-error">
            {t(locale, "output.joinFailed")}
          </p>
        ) : (
          <>
            {/* Keep item mounted under overlays so video clock/audio continue (Q1). */}
            {liveItem && OutputRenderer ? (
              <LiveItemLayer
                key={liveItemKey}
                effect={transitionEffect}
                dataContent={liveItem.type}
                ariaHidden={sceneMode !== "normal"}
                planeOverlay={
                  videoPlaneDots ? <StageDotsOverlay /> : null
                }
              >
                <OutputRenderer
                  item={liveItem}
                  slideIndex={liveSlideIndex}
                  assetSrc={assetSrc}
                  mediaClock={mediaClock}
                  onMediaClockSet={
                    liveItem.type === "video" || liveItem.type === "audio"
                      ? (patch) => setMediaClock(patch)
                      : undefined
                  }
                />
              </LiveItemLayer>
            ) : sceneMode === "normal" ? (
              <LiveItemLayer key={liveItemKey} effect={transitionEffect}>
                <p className="output-waiting">
                  {t(locale, "output.waiting")}
                </p>
              </LiveItemLayer>
            ) : null}
            <SubtitleOverlay
              agent={subtitleAgent}
              mediaClock={mediaClock}
              itemId={liveItem?.id ?? null}
              transitionEffect={transitionEffect}
            />
            <SceneOverlay mode={sceneMode} />
          </>
        )}

        {!isFullscreen && !hasSession ? (
          <div className="output-chrome">
            {showJoinForm ? (
              <form
                className="output-join"
                onSubmit={(e) => {
                  e.preventDefault();
                  if (manualCode.trim()) {
                    void join({ code: manualCode.trim().toUpperCase() });
                  }
                }}
              >
                <label>
                  {t(locale, "output.codeLabel")}
                  <input
                    value={manualCode}
                    onChange={(e) =>
                      setManualCode(e.target.value.toUpperCase())
                    }
                    placeholder="AB12"
                    maxLength={8}
                    autoFocus
                  />
                </label>
                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={!manualCode.trim()}
                >
                  {t(locale, "output.join")}
                </button>
              </form>
            ) : null}

            <p className="output-hint">{t(locale, "output.fullscreenHint")}</p>
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => void enterFullscreen()}
            >
              {t(locale, "output.enterFullscreen")}
            </button>
          </div>
        ) : !isFullscreen && hasSession ? (
          /* Joined: live content only — no control bar / seq counters. */
          <button
            type="button"
            className="output-exit-fs"
            onClick={() => void enterFullscreen()}
            title={t(locale, "output.enterFullscreen")}
            aria-label={t(locale, "output.enterFullscreen")}
          />
        ) : (
          <button
            type="button"
            className="output-exit-fs"
            onClick={() => void exitFullscreen()}
            aria-label="Exit fullscreen"
          />
        )}
      </div>
    </div>
  );
}
