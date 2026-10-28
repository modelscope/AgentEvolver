import { useCallback, useEffect, useRef, useState } from "react";
import type {
  ClientRole,
  HubMessage,
  ImagePayload,
  PlaylistItem,
  PresentationPayload,
  SceneMode,
  SessionSnapshot,
  SubtitleFontSize,
  SubtitleLeadMs,
  SubtitleTargetLang,
  TextPayload,
  ThemeBackground,
  ThemeId,
  ThemeOverrides,
  TransitionEffect,
  VideoPayload,
} from "@promedia/shared";
import {
  buildWsUrl,
  createSession,
  isHubUnreachableError,
  joinSession,
  probeHubHealth,
} from "../lib/hub";
import { SessionSocket } from "../lib/session-socket";

export type WsStatus = "idle" | "connecting" | "open" | "closed" | "error";

const RECONNECT_MS = 800;
const HUB_HINT =
  "HUB_UNREACHABLE: El hub no está corriendo. Abrí Terminal en la carpeta promedia y ejecutá: pnpm dev";

function applyHubMessage(
  msg: HubMessage,
  prev: SessionSnapshot | null,
): SessionSnapshot | null {
  if (msg.type === "session.snapshot") {
    return msg.payload as SessionSnapshot;
  }
  if (msg.type === "session.updated") {
    const p = msg.payload as Partial<SessionSnapshot>;
    if (!prev) {
      return prev;
    }
    return {
      ...prev,
      seq: msg.seq,
      lastTouchAt: p.lastTouchAt ?? prev.lastTouchAt,
      touchNote: p.touchNote ?? prev.touchNote,
      live: p.live ?? prev.live,
      playlist: p.playlist ?? prev.playlist,
    };
  }
  return prev;
}

function formatSessionError(err: unknown): string {
  if (
    (err instanceof Error && err.message === "HUB_UNREACHABLE") ||
    isHubUnreachableError(err)
  ) {
    return HUB_HINT;
  }
  return String(err);
}

/**
 * Shared create/join + WS wiring for Control and Output shells.
 */
export function useSessionConnection(role: ClientRole) {
  const [code, setCode] = useState("");
  const [accessToken, setAccessToken] = useState("");
  const [sessionId, setSessionId] = useState("");
  const [wsTicket, setWsTicket] = useState("");
  const [snapshot, setSnapshot] = useState<SessionSnapshot | null>(null);
  const [lastSeq, setLastSeq] = useState(-1);
  const [wsStatus, setWsStatus] = useState<WsStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [hubUnreachable, setHubUnreachable] = useState(false);
  const [hubChecking, setHubChecking] = useState(false);
  const socketRef = useRef<SessionSocket | null>(null);
  const generationRef = useRef(0);
  const reconnectTimerRef = useRef<number | null>(null);

  const clearReconnectTimer = useCallback(() => {
    if (reconnectTimerRef.current != null) {
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
  }, []);

  const detach = useCallback(() => {
    clearReconnectTimer();
    generationRef.current += 1;
    socketRef.current?.disconnect({ notifyClose: false });
    socketRef.current = null;
    setWsStatus("idle");
  }, [clearReconnectTimer]);

  useEffect(() => () => detach(), [detach]);

  /** Probe /health — used on mount and when Julio taps “Reintentar”. */
  const checkHubHealth = useCallback(async (): Promise<boolean> => {
    setHubChecking(true);
    try {
      const ok = await probeHubHealth();
      setHubUnreachable(!ok);
      if (ok) {
        setError((prev) =>
          prev && prev.startsWith("HUB_UNREACHABLE") ? null : prev,
        );
      } else {
        setError(HUB_HINT);
      }
      return ok;
    } finally {
      setHubChecking(false);
    }
  }, []);

  // Before any session: surface hub-down loudly (not only footer "WS desconectado").
  useEffect(() => {
    if (role !== "control") {
      return;
    }
    void checkHubHealth();
  }, [role, checkHubHealth]);

  const attachSocket = useCallback((url: string) => {
    clearReconnectTimer();
    socketRef.current?.disconnect({ notifyClose: false });
    const generation = ++generationRef.current;
    setWsStatus("connecting");
    const sock = new SessionSocket(url, {
      onOpen: () => {
        if (generation !== generationRef.current) {
          return;
        }
        setWsStatus("open");
        setHubUnreachable(false);
        setError(null);
      },
      onClose: () => {
        if (generation !== generationRef.current) {
          return;
        }
        setWsStatus("closed");
      },
      onError: () => {
        if (generation !== generationRef.current) {
          return;
        }
        setWsStatus("error");
        void probeHubHealth().then((ok) => {
          if (generation !== generationRef.current) {
            return;
          }
          if (!ok) {
            setHubUnreachable(true);
            setError(HUB_HINT);
          }
        });
      },
      onMessage: (msg: HubMessage) => {
        if (generation !== generationRef.current) {
          return;
        }
        setLastSeq(msg.seq);
        setSnapshot((prev) => applyHubMessage(msg, prev));
      },
    });
    socketRef.current = sock;
    sock.connect();
  }, [clearReconnectTimer]);

  const send = useCallback((type: string, payload?: unknown): boolean => {
    return socketRef.current?.send(type, payload) ?? false;
  }, []);

  const create = useCallback(async () => {
    setError(null);
    setHubUnreachable(false);
    setWsStatus("connecting");
    try {
      const res = await createSession("lan");
      setSessionId(res.sessionId);
      setCode(res.code);
      setAccessToken(res.accessToken);
      setWsTicket(res.wsTicket);
      setSnapshot(res.snapshot);
      setLastSeq(res.snapshot.seq);
      attachSocket(buildWsUrl(res.sessionId, role, res.wsTicket));
      return res;
    } catch (err) {
      setWsStatus("error");
      const formatted = formatSessionError(err);
      setError(formatted);
      setHubUnreachable(formatted.startsWith("HUB_UNREACHABLE"));
      throw err;
    }
  }, [attachSocket, role]);

  const join = useCallback(
    async (opts: { code?: string; accessToken?: string }) => {
      setError(null);
      setHubUnreachable(false);
      setWsStatus("connecting");
      try {
        const res = await joinSession({ ...opts, role });
        setSessionId(res.sessionId);
        setCode(res.code);
        setWsTicket(res.wsTicket);
        setSnapshot(res.snapshot);
        setLastSeq(res.snapshot.seq);
        attachSocket(buildWsUrl(res.sessionId, role, res.wsTicket));
        return res;
      } catch (err) {
        setWsStatus("error");
        const formatted = formatSessionError(err);
        setError(formatted);
        setHubUnreachable(formatted.startsWith("HUB_UNREACHABLE"));
        throw err;
      }
    },
    [attachSocket, role],
  );

  const reconnect = useCallback(() => {
    if (!wsTicket || !sessionId) {
      return;
    }
    clearReconnectTimer();
    setError(null);
    if (!socketRef.current) {
      attachSocket(buildWsUrl(sessionId, role, wsTicket));
      return;
    }
    // Keep generation — same handlers; silent disconnect keeps status "connecting"
    // so the auto-reconnect effect does not re-fire mid-handshake.
    setWsStatus("connecting");
    socketRef.current.reconnectWithResync();
  }, [attachSocket, clearReconnectTimer, role, sessionId, wsTicket]);

  // Auto-reconnect on closed/error for Control and Output.
  useEffect(() => {
    if (!sessionId || !wsTicket) {
      return;
    }
    if (wsStatus !== "closed" && wsStatus !== "error") {
      return;
    }
    clearReconnectTimer();
    reconnectTimerRef.current = window.setTimeout(() => {
      reconnectTimerRef.current = null;
      reconnect();
    }, RECONNECT_MS);
    return () => clearReconnectTimer();
  }, [wsStatus, sessionId, wsTicket, reconnect, clearReconnectTimer]);

  const addItem = useCallback(
    (item: PlaylistItem): boolean => send("playlist.itemAdd", { item }),
    [send],
  );
  const updateItem = useCallback(
    (
      id: string,
      patch: {
        payload?:
          | TextPayload
          | ImagePayload
          | VideoPayload
          | PresentationPayload;
        order?: number;
      },
    ): boolean => send("playlist.itemUpdate", { id, patch }),
    [send],
  );
  const removeItem = useCallback(
    (id: string): boolean => send("playlist.itemRemove", { id }),
    [send],
  );
  const reorder = useCallback(
    (orderedIds: string[]): boolean =>
      send("playlist.reorder", { orderedIds }),
    [send],
  );
  const goLive = useCallback(
    (itemId: string): boolean => send("live.go", { itemId }),
    [send],
  );
  const clearLive = useCallback(
    (): boolean => send("live.clear", {}),
    [send],
  );
  const setSlide = useCallback(
    (index: number): boolean => send("live.setSlide", { index }),
    [send],
  );
  const setMediaClock = useCallback(
    (patch: { playing: boolean; positionMs?: number }): boolean =>
      send("live.mediaClock.set", patch),
    [send],
  );
  const setSceneMode = useCallback(
    (sceneMode: SceneMode): boolean =>
      send("live.setSceneMode", { sceneMode }),
    [send],
  );
  const setTheme = useCallback(
    (patch: {
      themeId?: ThemeId;
      themeOverrides?: ThemeOverrides | null;
      themeBackground?: ThemeBackground | null;
      transitionEffect?: TransitionEffect;
    }): boolean => send("live.setTheme", patch),
    [send],
  );

  const setSubtitleAgent = useCallback(
    (patch: {
      enabled?: boolean;
      targetLang?: SubtitleTargetLang;
      fontSize?: SubtitleFontSize;
      leadMs?: SubtitleLeadMs;
      itemId?: string;
    }): boolean => send("live.subtitleAgent.set", patch),
    [send],
  );

  const isWsOpen = useCallback(
    () => socketRef.current?.isOpen === true,
    [],
  );

  return {
    code,
    accessToken,
    sessionId,
    wsTicket,
    snapshot,
    lastSeq,
    wsStatus,
    error,
    hubUnreachable,
    hubChecking,
    checkHubHealth,
    create,
    join,
    reconnect,
    detach,
    send,
    addItem,
    updateItem,
    removeItem,
    reorder,
    goLive,
    clearLive,
    setSlide,
    setMediaClock,
    setSceneMode,
    setTheme,
    setSubtitleAgent,
    isWsOpen,
    hasSession: Boolean(sessionId && wsTicket),
  };
}
