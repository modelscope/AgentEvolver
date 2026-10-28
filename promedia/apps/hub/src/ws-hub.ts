import type { IncomingMessage, Server as HttpServer } from "node:http";
import { WebSocketServer, type WebSocket } from "ws";
import type {
  ClientRole,
  ClientToHubMessage,
  ErrorPayload,
  HubMessage,
  ImagePayload,
  PlaylistItem,
  TextPayload,
  VideoPayload,
} from "@promedia/shared";
import {
  isSceneMode,
  isSubtitleFontSize,
  isSubtitleTargetLang,
  isThemeId,
  isTransitionEffect,
  normalizeSubtitleLeadMs,
  normalizeThemeBackground,
  normalizeThemeOverrides,
} from "@promedia/shared";
import {
  maybeRefreshSubtitleAgent,
  setSubtitleAgent,
} from "./subtitle-agent.js";
import { sessionStore, type SessionRecord } from "./session-store.js";

type ClientConn = {
  ws: WebSocket;
  sessionId: string;
  role: ClientRole;
  lastSeq: number;
};

function send(ws: WebSocket, message: HubMessage): void {
  if (ws.readyState === ws.OPEN) {
    ws.send(JSON.stringify(message));
  }
}

function sendError(
  ws: WebSocket,
  sessionId: string,
  seq: number,
  code: string,
  message: string,
): void {
  const payload: ErrorPayload = { code, message };
  send(ws, {
    type: "error",
    seq,
    sessionId,
    ts: new Date().toISOString(),
    payload,
  });
}

function parseClientMessage(raw: string): ClientToHubMessage | null {
  try {
    const data = JSON.parse(raw) as { type?: string; payload?: unknown };
    if (!data || typeof data.type !== "string") {
      return null;
    }
    const type = data.type;
    const payload = data.payload;

    if (type === "client.hello") {
      return { type: "client.hello", payload: payload as Record<string, unknown> };
    }
    if (type === "client.resync") {
      const lastSeq = (payload as { lastSeq?: unknown } | undefined)?.lastSeq;
      if (typeof lastSeq !== "number" || !Number.isFinite(lastSeq)) {
        return null;
      }
      return { type: "client.resync", payload: { lastSeq } };
    }
    if (type === "session.touch") {
      const note = (payload as { note?: unknown } | undefined)?.note;
      return {
        type: "session.touch",
        payload: typeof note === "string" ? { note } : {},
      };
    }
    if (type === "playlist.itemAdd") {
      const item = (payload as { item?: unknown } | undefined)?.item;
      if (!item) {
        return null;
      }
      return { type: "playlist.itemAdd", payload: { item: item as PlaylistItem } };
    }
    if (type === "playlist.itemUpdate") {
      const p = payload as { id?: unknown; patch?: unknown } | undefined;
      if (!p || typeof p.id !== "string" || !p.patch || typeof p.patch !== "object") {
        return null;
      }
      return {
        type: "playlist.itemUpdate",
        payload: {
          id: p.id,
          patch: p.patch as {
            payload?: TextPayload | ImagePayload | VideoPayload;
            order?: number;
          },
        },
      };
    }
    if (type === "playlist.itemRemove") {
      const id = (payload as { id?: unknown } | undefined)?.id;
      if (typeof id !== "string") {
        return null;
      }
      return { type: "playlist.itemRemove", payload: { id } };
    }
    if (type === "playlist.reorder") {
      const orderedIds = (payload as { orderedIds?: unknown } | undefined)?.orderedIds;
      if (
        !Array.isArray(orderedIds) ||
        !orderedIds.every((id) => typeof id === "string")
      ) {
        return null;
      }
      return { type: "playlist.reorder", payload: { orderedIds } };
    }
    if (type === "live.go") {
      const itemId = (payload as { itemId?: unknown } | undefined)?.itemId;
      if (typeof itemId !== "string") {
        return null;
      }
      return { type: "live.go", payload: { itemId } };
    }
    if (type === "live.clear") {
      return { type: "live.clear", payload: {} };
    }
    if (type === "live.setSlide") {
      const index = (payload as { index?: unknown } | undefined)?.index;
      if (typeof index !== "number" || !Number.isFinite(index)) {
        return null;
      }
      return { type: "live.setSlide", payload: { index } };
    }
    if (type === "live.mediaClock.set") {
      const p = payload as { playing?: unknown; positionMs?: unknown } | undefined;
      if (!p || typeof p.playing !== "boolean") {
        return null;
      }
      const positionMs =
        typeof p.positionMs === "number" && Number.isFinite(p.positionMs)
          ? p.positionMs
          : undefined;
      return {
        type: "live.mediaClock.set",
        payload: { playing: p.playing, positionMs },
      };
    }
    if (type === "live.subtitleAgent.set") {
      const p = payload as {
        enabled?: unknown;
        targetLang?: unknown;
        fontSize?: unknown;
        leadMs?: unknown;
        itemId?: unknown;
      } | undefined;
      if (!p) {
        return null;
      }
      const out: {
        enabled?: boolean;
        targetLang?: import("@promedia/shared").SubtitleTargetLang;
        fontSize?: import("@promedia/shared").SubtitleFontSize;
        leadMs?: import("@promedia/shared").SubtitleLeadMs;
        itemId?: string;
      } = {};
      if (p.enabled !== undefined) {
        if (typeof p.enabled !== "boolean") {
          return null;
        }
        out.enabled = p.enabled;
      }
      if (p.targetLang !== undefined) {
        if (!isSubtitleTargetLang(p.targetLang)) {
          return null;
        }
        out.targetLang = p.targetLang;
      }
      if (p.fontSize !== undefined) {
        if (!isSubtitleFontSize(p.fontSize)) {
          return null;
        }
        out.fontSize = p.fontSize;
      }
      if (p.leadMs !== undefined) {
        if (typeof p.leadMs !== "number" || !Number.isFinite(p.leadMs)) {
          return null;
        }
        // Accept signed presets and legacy positive “how early” ms.
        out.leadMs = normalizeSubtitleLeadMs(p.leadMs);
      }
      if (p.itemId !== undefined) {
        if (typeof p.itemId !== "string") {
          return null;
        }
        out.itemId = p.itemId;
      }
      if (
        out.enabled === undefined &&
        out.targetLang === undefined &&
        out.fontSize === undefined &&
        out.leadMs === undefined &&
        out.itemId === undefined
      ) {
        return null;
      }
      return { type: "live.subtitleAgent.set", payload: out };
    }
    if (type === "live.setSceneMode") {
      const sceneMode = (payload as { sceneMode?: unknown } | undefined)?.sceneMode;
      if (!isSceneMode(sceneMode)) {
        return null;
      }
      return { type: "live.setSceneMode", payload: { sceneMode } };
    }
    if (type === "live.setTheme") {
      const p = payload as {
        themeId?: unknown;
        themeOverrides?: unknown;
        themeBackground?: unknown;
        transitionEffect?: unknown;
      } | undefined;
      if (!p) {
        return null;
      }
      const out: {
        themeId?: import("@promedia/shared").ThemeId;
        themeOverrides?: import("@promedia/shared").ThemeOverrides | null;
        themeBackground?: import("@promedia/shared").ThemeBackground | null;
        transitionEffect?: import("@promedia/shared").TransitionEffect;
      } = {};
      if (p.themeId !== undefined) {
        if (!isThemeId(p.themeId)) {
          return null;
        }
        out.themeId = p.themeId;
      }
      if (p.themeOverrides !== undefined) {
        if (p.themeOverrides === null) {
          out.themeOverrides = null;
        } else {
          const n = normalizeThemeOverrides(p.themeOverrides);
          if (n === null) {
            // allow empty object as clear
            out.themeOverrides = null;
          } else {
            out.themeOverrides = n;
          }
        }
      }
      if (p.themeBackground !== undefined) {
        if (p.themeBackground === null) {
          out.themeBackground = { mode: "solid" };
        } else {
          out.themeBackground = normalizeThemeBackground(p.themeBackground);
        }
      }
      if (p.transitionEffect !== undefined) {
        if (!isTransitionEffect(p.transitionEffect)) {
          return null;
        }
        out.transitionEffect = p.transitionEffect;
      }
      if (
        out.themeId === undefined &&
        out.themeOverrides === undefined &&
        out.themeBackground === undefined &&
        out.transitionEffect === undefined
      ) {
        return null;
      }
      return { type: "live.setTheme", payload: out };
    }
    return null;
  } catch {
    return null;
  }
}

export class WsHub {
  private readonly wss: WebSocketServer;
  private readonly clients = new Set<ClientConn>();

  constructor(server: HttpServer) {
    this.wss = new WebSocketServer({ noServer: true });

    server.on("upgrade", (req, socket, head) => {
      const url = new URL(req.url ?? "/", "http://localhost");
      if (!url.pathname.endsWith("/api/v1/ws") && url.pathname !== "/api/v1/ws") {
        socket.destroy();
        return;
      }

      const sessionId = url.searchParams.get("sessionId") ?? "";
      const role = url.searchParams.get("role") as ClientRole | null;
      const credential = url.searchParams.get("credential") ?? "";

      const rejected = this.validateUpgrade(sessionId, role, credential);
      if (rejected) {
        socket.write(
          `HTTP/1.1 ${rejected.status} ${rejected.statusText}\r\nConnection: close\r\n\r\n`,
        );
        socket.destroy();
        return;
      }

      this.wss.handleUpgrade(req, socket, head, (ws) => {
        this.onConnection(ws, req, sessionId, role as ClientRole);
      });
    });
  }

  private validateUpgrade(
    sessionId: string,
    role: ClientRole | null,
    credential: string,
  ): { status: number; statusText: string } | null {
    if (!sessionId || !credential || (role !== "control" && role !== "output")) {
      return { status: 401, statusText: "Unauthorized" };
    }
    const session = sessionStore.get(sessionId);
    if (!session) {
      return { status: 401, statusText: "Unauthorized" };
    }
    const ticket = sessionStore.getTicket(credential);
    if (!ticket || ticket.sessionId !== sessionId || ticket.role !== role) {
      return { status: 403, statusText: "Forbidden" };
    }
    return null;
  }

  private onConnection(
    ws: WebSocket,
    _req: IncomingMessage,
    sessionId: string,
    role: ClientRole,
  ): void {
    const session = sessionStore.get(sessionId);
    if (!session) {
      ws.close(4401, "session not found");
      return;
    }

    const conn: ClientConn = {
      ws,
      sessionId,
      role,
      lastSeq: session.seq,
    };
    this.clients.add(conn);

    const snapshotMsg = sessionStore.makeMessage(
      session,
      "session.snapshot",
      sessionStore.toSnapshot(session),
    );
    send(ws, snapshotMsg);

    ws.on("message", (data) => {
      const text = typeof data === "string" ? data : data.toString("utf8");
      this.handleClientMessage(conn, text);
    });

    ws.on("close", () => {
      this.clients.delete(conn);
    });

    ws.on("error", () => {
      this.clients.delete(conn);
    });
  }

  private requireControl(conn: ClientConn, session: SessionRecord): boolean {
    if (conn.role !== "control") {
      sendError(
        conn.ws,
        session.id,
        session.seq,
        "forbidden",
        "output cannot mutate session",
      );
      return false;
    }
    return true;
  }

  private applyResult(
    conn: ClientConn,
    session: SessionRecord,
    result: HubMessage | { error: string },
  ): void {
    if ("error" in result) {
      sendError(conn.ws, session.id, session.seq, result.error, result.error);
      return;
    }
    this.broadcast(session.id, result);
  }

  private handleClientMessage(conn: ClientConn, raw: string): void {
    const session = sessionStore.get(conn.sessionId);
    if (!session) {
      sendError(conn.ws, conn.sessionId, 0, "not_found", "Session gone");
      return;
    }

    const msg = parseClientMessage(raw);
    if (!msg) {
      sendError(conn.ws, session.id, session.seq, "bad_request", "Invalid message");
      return;
    }

    if (msg.type === "client.hello") {
      const snapshotMsg = sessionStore.makeMessage(
        session,
        "session.snapshot",
        sessionStore.toSnapshot(session),
      );
      send(conn.ws, snapshotMsg);
      conn.lastSeq = session.seq;
      return;
    }

    if (msg.type === "client.resync") {
      this.handleResync(conn, session, msg.payload.lastSeq);
      return;
    }

    if (!this.requireControl(conn, session)) {
      return;
    }

    switch (msg.type) {
      case "session.touch":
        this.applyResult(conn, session, sessionStore.touch(session, msg.payload?.note));
        return;
      case "playlist.itemAdd":
        this.applyResult(conn, session, sessionStore.addItem(session, msg.payload.item));
        return;
      case "playlist.itemUpdate":
        this.applyResult(
          conn,
          session,
          sessionStore.updateItem(session, msg.payload.id, msg.payload.patch),
        );
        return;
      case "playlist.itemRemove":
        this.applyResult(
          conn,
          session,
          sessionStore.removeItem(session, msg.payload.id),
        );
        return;
      case "playlist.reorder":
        this.applyResult(
          conn,
          session,
          sessionStore.reorder(session, msg.payload.orderedIds),
        );
        return;
      case "live.go": {
        const goResult = sessionStore.goLive(session, msg.payload.itemId);
        this.applyResult(conn, session, goResult);
        if (!("error" in goResult)) {
          maybeRefreshSubtitleAgent(session);
        }
        return;
      }
      case "live.clear":
        this.applyResult(conn, session, sessionStore.clearLive(session));
        return;
      case "live.setSlide":
        this.applyResult(
          conn,
          session,
          sessionStore.setSlide(session, msg.payload.index),
        );
        return;
      case "live.mediaClock.set":
        this.applyResult(
          conn,
          session,
          sessionStore.setMediaClock(session, msg.payload),
        );
        return;
      case "live.subtitleAgent.set":
        this.applyResult(
          conn,
          session,
          setSubtitleAgent(session, msg.payload),
        );
        return;
      case "live.setSceneMode":
        this.applyResult(
          conn,
          session,
          sessionStore.setSceneMode(session, msg.payload.sceneMode),
        );
        return;
      case "live.setTheme":
        this.applyResult(
          conn,
          session,
          sessionStore.setTheme(session, msg.payload),
        );
        return;
      default:
        sendError(conn.ws, session.id, session.seq, "bad_request", "Unknown type");
    }
  }

  private handleResync(
    conn: ClientConn,
    session: SessionRecord,
    lastSeq: number,
  ): void {
    const live = sessionStore.get(session.id);
    if (!live) {
      sendError(conn.ws, session.id, 0, "not_found", "Session gone");
      return;
    }

    if (lastSeq >= live.seq) {
      conn.lastSeq = live.seq;
      return;
    }

    const events = sessionStore.eventsAfter(live, lastSeq);
    if (events === null) {
      const snapshotMsg = sessionStore.makeMessage(
        live,
        "session.snapshot",
        sessionStore.toSnapshot(live),
      );
      send(conn.ws, snapshotMsg);
      conn.lastSeq = live.seq;
      return;
    }

    for (const event of events) {
      send(conn.ws, event);
      conn.lastSeq = event.seq;
    }
  }

  broadcast(sessionId: string, message: HubMessage): void {
    for (const conn of this.clients) {
      if (conn.sessionId !== sessionId) {
        continue;
      }
      if (message.seq <= conn.lastSeq) {
        continue;
      }
      send(conn.ws, message);
      conn.lastSeq = message.seq;
    }
  }
}
