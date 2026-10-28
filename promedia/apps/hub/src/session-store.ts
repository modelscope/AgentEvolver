import type {
  ClientRole,
  HubMessage,
  ImagePayload,
  MediaClock,
  PlaylistItem,
  PresentationPayload,
  SceneMode,
  SessionLiveState,
  SessionMode,
  SessionPlaylist,
  SessionSnapshot,
  SessionUpdatedPayload,
  SubtitleAgentState,
  TextPayload,
  ThemeBackground,
  ThemeId,
  ThemeOverrides,
  TransitionEffect,
  VideoPayload,
} from "@promedia/shared";
import {
  emptyLiveState,
  emptyPlaylist,
  emptySubtitleAgentState,
  isSubtitleFontSize,
  isSubtitleTimingMode,
  normalizeSubtitleLeadMs,
  isSceneMode,
  isThemeId,
  isTransitionEffect,
  itemProjectionSlideCount,
  normalizeThemeBackground,
  normalizeThemeOverrides,
} from "@promedia/shared";
import { validatePlaylistItem } from "@promedia/item-types/validate";
import { assetStore } from "./asset-store.js";
import {
  generateAccessToken,
  generateSessionId,
  generateShortCode,
  generateWsTicket,
} from "./ids.js";

const EVENT_BUFFER_SIZE = 100;

export type WsTicketRecord = {
  ticket: string;
  sessionId: string;
  role: ClientRole;
  createdAt: string;
};

export type SessionRecord = {
  id: string;
  code: string;
  accessToken: string;
  mode: SessionMode;
  createdAt: string;
  seq: number;
  live: SessionLiveState;
  playlist: SessionPlaylist;
  lastTouchAt?: string;
  touchNote?: string;
  eventBuffer: HubMessage[];
};

function nowIso(): string {
  return new Date().toISOString();
}

function sortItems(items: PlaylistItem[]): PlaylistItem[] {
  return [...items]
    .sort((a, b) => a.order - b.order)
    .map((item, index) => ({ ...item, order: index }));
}

export class SessionStore {
  private readonly byId = new Map<string, SessionRecord>();
  private readonly byCode = new Map<string, string>();
  private readonly byAccessToken = new Map<string, string>();
  private readonly tickets = new Map<string, WsTicketRecord>();

  create(mode: SessionMode = "lan"): {
    session: SessionRecord;
    ticket: WsTicketRecord;
  } {
    let code = generateShortCode(4);
    while (this.byCode.has(code)) {
      code = generateShortCode(4);
    }

    const id = generateSessionId();
    const accessToken = generateAccessToken();
    const createdAt = nowIso();

    const session: SessionRecord = {
      id,
      code,
      accessToken,
      mode,
      createdAt,
      seq: 0,
      live: emptyLiveState(),
      playlist: emptyPlaylist(),
      eventBuffer: [],
    };

    this.byId.set(id, session);
    this.byCode.set(code, id);
    this.byAccessToken.set(accessToken, id);

    const ticket = this.issueTicket(id, "control");
    return { session, ticket };
  }

  get(sessionId: string): SessionRecord | undefined {
    return this.byId.get(sessionId);
  }

  findByCode(code: string): SessionRecord | undefined {
    const id = this.byCode.get(code.trim().toUpperCase());
    return id ? this.byId.get(id) : undefined;
  }

  findByAccessToken(accessToken: string): SessionRecord | undefined {
    const id = this.byAccessToken.get(accessToken);
    return id ? this.byId.get(id) : undefined;
  }

  issueTicket(sessionId: string, role: ClientRole): WsTicketRecord {
    const record: WsTicketRecord = {
      ticket: generateWsTicket(),
      sessionId,
      role,
      createdAt: nowIso(),
    };
    this.tickets.set(record.ticket, record);
    return record;
  }

  getTicket(ticket: string): WsTicketRecord | undefined {
    return this.tickets.get(ticket);
  }

  toSnapshot(session: SessionRecord): SessionSnapshot {
    return {
      id: session.id,
      code: session.code,
      mode: session.mode,
      createdAt: session.createdAt,
      seq: session.seq,
      live: { ...session.live },
      playlist: { items: session.playlist.items.map((item) => ({ ...item })) },
      lastTouchAt: session.lastTouchAt,
      touchNote: session.touchNote,
    };
  }

  makeMessage(
    session: SessionRecord,
    type: string,
    payload: unknown,
  ): HubMessage {
    return {
      type,
      seq: session.seq,
      sessionId: session.id,
      ts: nowIso(),
      payload,
    };
  }

  pushEvent(session: SessionRecord, message: HubMessage): void {
    session.eventBuffer.push(message);
    if (session.eventBuffer.length > EVENT_BUFFER_SIZE) {
      session.eventBuffer.splice(0, session.eventBuffer.length - EVENT_BUFFER_SIZE);
    }
  }

  private ensureSubtitleAgent(session: SessionRecord): SubtitleAgentState {
    if (!session.live.subtitleAgent) {
      session.live.subtitleAgent = emptySubtitleAgentState();
    } else {
      const agent = session.live.subtitleAgent;
      const fontOk = isSubtitleFontSize(agent.fontSize);
      const leadOk =
        typeof agent.leadMs === "number" &&
        Number.isFinite(agent.leadMs) &&
        agent.leadMs === normalizeSubtitleLeadMs(agent.leadMs);
      const timingOk = isSubtitleTimingMode(
        (agent as { timingMode?: unknown }).timingMode,
      );
      if (!fontOk || !leadOk || !timingOk) {
        session.live.subtitleAgent = {
          ...agent,
          fontSize: fontOk ? agent.fontSize : "md",
          leadMs: normalizeSubtitleLeadMs(agent.leadMs),
          timingMode: timingOk ? agent.timingMode : "exact",
        };
      }
    }
    return session.live.subtitleAgent;
  }

  private bumpUpdated(session: SessionRecord): HubMessage {
    this.ensureSubtitleAgent(session);
    session.seq += 1;
    const payload: SessionUpdatedPayload = {
      seq: session.seq,
      live: {
        ...session.live,
        subtitleAgent: { ...session.live.subtitleAgent },
      },
      playlist: { items: session.playlist.items.map((item) => ({ ...item })) },
      lastTouchAt: session.lastTouchAt,
      touchNote: session.touchNote,
    };
    const message = this.makeMessage(session, "session.updated", payload);
    this.pushEvent(session, message);
    return message;
  }

  /** Persist subtitleAgent field already mutated on session.live. */
  touchSubtitleAgent(session: SessionRecord): HubMessage {
    this.ensureSubtitleAgent(session);
    return this.bumpUpdated(session);
  }

  touch(session: SessionRecord, note?: string): HubMessage {
    session.lastTouchAt = nowIso();
    if (note !== undefined) {
      session.touchNote = note;
    }
    return this.bumpUpdated(session);
  }

  /**
   * Replace session playlist with a validated snapshot (SavedPlaylist load).
   * Skips file items whose MediaAsset is missing in this session (text/YouTube OK).
   */
  replacePlaylist(
    session: SessionRecord,
    rawItems: unknown[],
  ):
    | { message: HubMessage; skipped: number }
    | { error: string } {
    const next: PlaylistItem[] = [];
    let skipped = 0;
    for (const raw of rawItems) {
      const item = validatePlaylistItem(raw);
      if (!item) {
        return { error: "invalid_item" };
      }
      const assetErr = this.requireAsset(session, item);
      if (assetErr) {
        if (assetErr.error === "unknown_asset") {
          skipped += 1;
          continue;
        }
        return assetErr;
      }
      next.push(item);
    }
    session.playlist.items = sortItems(next);
    if (
      session.live.currentItemId &&
      !session.playlist.items.some((i) => i.id === session.live.currentItemId)
    ) {
      session.live.currentItemId = null;
      session.live.currentSlideIndex = 0;
      session.live.mediaClock = null;
      session.live.sceneMode = "normal";
    }
    return { message: this.bumpUpdated(session), skipped };
  }

  addItem(session: SessionRecord, rawItem: unknown): HubMessage | { error: string } {
    const item = validatePlaylistItem(rawItem);
    if (!item) {
      return { error: "invalid_item" };
    }
    if (session.playlist.items.some((existing) => existing.id === item.id)) {
      return { error: "duplicate_id" };
    }
    const assetErr = this.requireAsset(session, item);
    if (assetErr) {
      return assetErr;
    }
    const order =
      session.playlist.items.length === 0
        ? 0
        : Math.max(...session.playlist.items.map((i) => i.order)) + 1;
    session.playlist.items = sortItems([
      ...session.playlist.items,
      { ...item, order },
    ]);
    return this.bumpUpdated(session);
  }

  updateItem(
    session: SessionRecord,
    id: string,
    patch: {
      payload?: TextPayload | ImagePayload | VideoPayload | PresentationPayload;
      order?: number;
    },
  ): HubMessage | { error: string } {
    const idx = session.playlist.items.findIndex((i) => i.id === id);
    if (idx < 0) {
      return { error: "not_found" };
    }
    const current = session.playlist.items[idx]!;
    const nextRaw = {
      id: current.id,
      type: current.type,
      order: patch.order ?? current.order,
      payload: patch.payload ?? current.payload,
    };
    const validated = validatePlaylistItem(nextRaw);
    if (!validated) {
      return { error: "invalid_item" };
    }
    const assetErr = this.requireAsset(session, validated);
    if (assetErr) {
      return assetErr;
    }
    const items = [...session.playlist.items];
    items[idx] = validated;
    session.playlist.items = sortItems(items);
    return this.bumpUpdated(session);
  }

  removeItem(session: SessionRecord, id: string): HubMessage | { error: string } {
    const before = session.playlist.items.length;
    session.playlist.items = sortItems(
      session.playlist.items.filter((item) => item.id !== id),
    );
    if (session.playlist.items.length === before) {
      return { error: "not_found" };
    }
    if (session.live.currentItemId === id) {
      session.live.currentItemId = null;
      session.live.currentSlideIndex = 0;
      session.live.mediaClock = null;
    }
    return this.bumpUpdated(session);
  }

  reorder(session: SessionRecord, orderedIds: string[]): HubMessage | { error: string } {
    if (orderedIds.length !== session.playlist.items.length) {
      return { error: "invalid_reorder" };
    }
    const byId = new Map(session.playlist.items.map((i) => [i.id, i]));
    const next: PlaylistItem[] = [];
    for (const id of orderedIds) {
      const item = byId.get(id);
      if (!item) {
        return { error: "invalid_reorder" };
      }
      next.push(item);
      byId.delete(id);
    }
    if (byId.size > 0) {
      return { error: "invalid_reorder" };
    }
    session.playlist.items = sortItems(next);
    return this.bumpUpdated(session);
  }

  goLive(session: SessionRecord, itemId: string): HubMessage | { error: string } {
    const item = session.playlist.items.find((i) => i.id === itemId);
    if (!item) {
      return { error: "not_found" };
    }
    if (
      item.type !== "text" &&
      item.type !== "image" &&
      item.type !== "video" &&
      item.type !== "audio" &&
      item.type !== "bible"
    ) {
      return { error: "unsupported_type" };
    }
    session.live.currentItemId = itemId;
    session.live.currentSlideIndex = 0;
    // Julio lock: Go Live resets sceneMode to normal
    session.live.sceneMode = "normal";
    // Q3: video/audio Go Live auto-starts playback (unmuted per Q2 / payload.muted)
    if (
      (item.type === "video" || item.type === "audio") &&
      item.payload.autoplay !== false
    ) {
      session.live.mediaClock = {
        itemId,
        playing: true,
        positionMs: 0,
        rate: 1,
        updatedAt: nowIso(),
      };
    } else if (item.type === "video" || item.type === "audio") {
      session.live.mediaClock = {
        itemId,
        playing: false,
        positionMs: 0,
        rate: 1,
        updatedAt: nowIso(),
      };
    } else {
      session.live.mediaClock = null;
    }
    return this.bumpUpdated(session);
  }

  clearLive(session: SessionRecord): HubMessage {
    session.live.currentItemId = null;
    session.live.currentSlideIndex = 0;
    session.live.mediaClock = null;
    session.live.sceneMode = "normal";
    const agent = this.ensureSubtitleAgent(session);
    if (agent.enabled) {
      session.live.subtitleAgent = {
        ...agent,
        status: "idle",
        itemId: null,
        assetId: null,
        cues: [],
        hint: null,
        sourceLang: null,
        translated: false,
        updatedAt: nowIso(),
      };
    }
    return this.bumpUpdated(session);
  }


  setSceneMode(
    session: SessionRecord,
    sceneMode: SceneMode | unknown,
  ): HubMessage | { error: string } {
    if (!isSceneMode(sceneMode)) {
      return { error: "invalid_scene_mode" };
    }
    // Overlay only — does not touch currentItemId or mediaClock (Q1/Q2)
    session.live.sceneMode = sceneMode;
    return this.bumpUpdated(session);
  }

  setTheme(
    session: SessionRecord,
    patch: {
      themeId?: ThemeId;
      themeOverrides?: ThemeOverrides | null;
      themeBackground?: ThemeBackground | null;
      transitionEffect?: TransitionEffect;
    },
  ): HubMessage | { error: string } {
    if (patch.themeId !== undefined) {
      if (!isThemeId(patch.themeId)) {
        return { error: "invalid_theme_id" };
      }
      session.live.themeId = patch.themeId;
      // Switching preset clears overrides unless caller also sends overrides
      if (patch.themeOverrides === undefined) {
        session.live.themeOverrides = null;
      }
    }
    if (patch.themeOverrides !== undefined) {
      if (patch.themeOverrides === null) {
        session.live.themeOverrides = null;
      } else {
        const normalized = normalizeThemeOverrides(patch.themeOverrides);
        if (normalized === null && Object.keys(patch.themeOverrides).length > 0) {
          return { error: "invalid_theme_overrides" };
        }
        session.live.themeOverrides = normalized;
      }
    }
    if (patch.themeBackground !== undefined) {
      session.live.themeBackground = normalizeThemeBackground(
        patch.themeBackground,
      );
    }
    if (patch.transitionEffect !== undefined) {
      if (!isTransitionEffect(patch.transitionEffect)) {
        return { error: "invalid_transition_effect" };
      }
      session.live.transitionEffect = patch.transitionEffect;
    }
    if (
      patch.themeId === undefined &&
      patch.themeOverrides === undefined &&
      patch.themeBackground === undefined &&
      patch.transitionEffect === undefined
    ) {
      return { error: "empty_theme_patch" };
    }
    return this.bumpUpdated(session);
  }

  setSlide(session: SessionRecord, index: number): HubMessage | { error: string } {
    const itemId = session.live.currentItemId;
    if (!itemId) {
      return { error: "no_live_item" };
    }
    const item = session.playlist.items.find((i) => i.id === itemId);
    if (
      !item ||
      (item.type !== "text" &&
        item.type !== "bible" &&
        item.type !== "presentation")
    ) {
      return { error: "no_live_item" };
    }
    const max = Math.max(itemProjectionSlideCount(item) - 1, 0);
    const clamped = Math.min(Math.max(Math.trunc(index), 0), max);
    session.live.currentSlideIndex = clamped;
    return this.bumpUpdated(session);
  }

  setMediaClock(
    session: SessionRecord,
    patch: { playing: boolean; positionMs?: number },
  ): HubMessage | { error: string } {
    const itemId = session.live.currentItemId;
    if (!itemId) {
      return { error: "no_live_item" };
    }
    const item = session.playlist.items.find((i) => i.id === itemId);
    if (!item || (item.type !== "video" && item.type !== "audio")) {
      return { error: "no_video_live" };
    }
    const prev = session.live.mediaClock;
    const positionMs =
      typeof patch.positionMs === "number" && Number.isFinite(patch.positionMs)
        ? Math.max(0, patch.positionMs)
        : (prev?.positionMs ?? 0);
    const clock: MediaClock = {
      itemId,
      playing: patch.playing,
      positionMs,
      rate: prev?.rate ?? 1,
      updatedAt: nowIso(),
    };
    session.live.mediaClock = clock;
    return this.bumpUpdated(session);
  }

  private requireAsset(
    session: SessionRecord,
    item: PlaylistItem,
  ): { error: string } | null {
    if (item.type === "image") {
      const asset = assetStore.get(session.id, item.payload.assetId);
      if (!asset || asset.kind !== "image") {
        return { error: "unknown_asset" };
      }
    }
    if (item.type === "video" && item.payload.sourceKind === "file") {
      const asset = assetStore.get(session.id, item.payload.assetId);
      if (!asset || asset.kind !== "video") {
        return { error: "unknown_asset" };
      }
    }
    if (item.type === "audio") {
      const asset = assetStore.get(session.id, item.payload.assetId);
      if (!asset || asset.kind !== "audio") {
        return { error: "unknown_asset" };
      }
    }
    if (item.type === "presentation") {
      for (const slide of item.payload.slides) {
        const asset = assetStore.get(session.id, slide.assetId);
        if (!asset || asset.kind !== "image") {
          return { error: "unknown_asset" };
        }
      }
    }
    return null;
  }

  eventsAfter(session: SessionRecord, lastSeq: number): HubMessage[] | null {
    if (session.eventBuffer.length === 0) {
      return null;
    }
    const oldest = session.eventBuffer[0]?.seq ?? 0;
    if (lastSeq + 1 < oldest) {
      return null;
    }
    return session.eventBuffer.filter((m) => m.seq > lastSeq);
  }
}

export const sessionStore = new SessionStore();
