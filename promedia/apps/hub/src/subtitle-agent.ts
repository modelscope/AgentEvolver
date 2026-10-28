import type {
  AudioPlaylistItem,
  HubMessage,
  PlaylistItem,
  SubtitleAgentState,
  SubtitleFontSize,
  SubtitleLeadMs,
  SubtitleTargetLang,
  SubtitleTimingMode,
  VideoPlaylistItem,
} from "@promedia/shared";
import {
  CANCIONERO_DEFAULT_LEAD_MS,
  DEFAULT_SUBTITLE_LEAD_MS,
  emptySubtitleAgentState,
  normalizeSubtitleLeadMs,
} from "@promedia/shared";
import { rm } from "node:fs/promises";
import { assetStore } from "./asset-store.js";
import { broadcastSession } from "./broadcast.js";
import {
  ffmpegAvailable,
  mediaFileExists,
  transcribeLocalFile,
  whisperCliAvailable,
} from "./subtitle-asr.js";
import { isMissingBinHint, missingBinHintFromError } from "./subtitle-bins.js";
import { tryLocalCaptionTracks } from "./subtitle-local-tracks.js";
import { ensureSubtitleDirs } from "./subtitle-paths.js";
import { translateCuesLocal } from "./subtitle-translate.js";
import { filterSpeechCues } from "./subtitle-cue-text.js";
import { cuesLikelyInLanguage } from "./subtitle-lang-detect.js";
import {
  preserveNativeCaptionCues,
  shapeLyricLikeCues,
} from "./subtitle-lyric-shape.js";
import {
  alignCancioneroLinesToTiming,
  buildKaraokeAlignPrompt,
  findCancioneroSong,
  packKaraokeDisplayCues,
  probeMediaDurationMs,
  songCancioneroLines,
  songToSubtitleCues,
  titleFromAudioFilename,
} from "./subtitle-cancionero.js";
import {
  ensureYoutubeAudio,
  expandCaptionPreferLangs,
  fetchYoutubeMeta,
  tryYoutubeCaptions,
  ytDlpAvailable,
} from "./subtitle-youtube.js";
import type { SessionRecord } from "./session-store.js";
import { sessionStore } from "./session-store.js";

/** Cap lang tag to 2-letter base (`es-419` → `es`). */
function langBase(lang: string | null | undefined): string | null {
  if (!lang) return null;
  const l = lang.toLowerCase();
  if (l.startsWith("zh")) return "zh";
  return l.slice(0, 2);
}

/** In-flight jobs keyed by sessionId — cancel by generation bump. */
const jobGeneration = new Map<string, number>();

/**
 * Hard YouTube / extract failures: do not tight-loop retries on focus/Go Live.
 * Allow another attempt after backoff or when the user toggles the agent on.
 */
const HARD_YT_HINT_PREFIXES = [
  "youtube_download_failed",
  "youtube_cache_path_error",
  "youtube_cache_missing_after_download",
  "youtube_age_restricted",
  "youtube_blocked",
  "youtube_private",
  "youtube_unavailable",
  "youtube_invalid",
  "yt_dlp_spawn_failed",
  "ffmpeg_extract_failed",
] as const;

const YT_HARD_FAIL_BACKOFF_MS = 30_000;
const ytHardFailUntil = new Map<string, number>();

function bumpJob(sessionId: string): number {
  const next = (jobGeneration.get(sessionId) ?? 0) + 1;
  jobGeneration.set(sessionId, next);
  return next;
}

function isCurrent(sessionId: string, gen: number): boolean {
  return jobGeneration.get(sessionId) === gen;
}

function isHardYoutubeHint(hint: string | null | undefined): boolean {
  if (!hint) return false;
  return HARD_YT_HINT_PREFIXES.some(
    (p) => hint === p || hint.startsWith(`${p}:`) || hint.startsWith(`${p}_`),
  );
}

function markYtHardFail(sessionId: string): void {
  ytHardFailUntil.set(sessionId, Date.now() + YT_HARD_FAIL_BACKOFF_MS);
}

function clearYtHardFail(sessionId: string): void {
  ytHardFailUntil.delete(sessionId);
}

function ytHardFailInBackoff(sessionId: string): boolean {
  const until = ytHardFailUntil.get(sessionId);
  if (!until) return false;
  if (Date.now() >= until) {
    ytHardFailUntil.delete(sessionId);
    return false;
  }
  return true;
}

function patchAndBroadcast(
  session: SessionRecord,
  patch: Partial<SubtitleAgentState>,
): void {
  const prev = session.live.subtitleAgent ?? emptySubtitleAgentState();
  session.live.subtitleAgent = {
    ...prev,
    ...patch,
    updatedAt: new Date().toISOString(),
  };
  const message = sessionStore.touchSubtitleAgent(session);
  broadcastSession(session.id, message);
}

type AvPlaylistItem = VideoPlaylistItem | AudioPlaylistItem;

type ResolveOk = {
  ok: true;
  item: AvPlaylistItem;
  assetId: string;
  /** Absolute path once media is local (file or after yt-dlp). */
  mediaPath: string | null;
  youtube: null | { videoId: string; sourceUrl: string };
  /** Upload basename for sidecar caption lookup (local file/audio only). */
  originalName: string | null;
};

type ResolveFail = {
  ok: false;
  reason:
    | "no_item"
    | "not_video"
    | "no_asset"
    | "missing_file"
    | "bad_youtube";
  item: PlaylistItem | null;
};

function resolveMediaSource(
  session: SessionRecord,
  itemId: string | null | undefined,
): ResolveOk | ResolveFail {
  const id = itemId || session.live.currentItemId || null;
  if (!id) {
    return { ok: false, reason: "no_item", item: null };
  }
  const item = session.playlist.items.find((i) => i.id === id) ?? null;
  if (!item) {
    return { ok: false, reason: "no_item", item: null };
  }

  // Guion Audio tab (#76) — same ASR / sidecar path as local video files.
  if (item.type === "audio") {
    const assetId = item.payload.assetId;
    const asset = assetStore.get(session.id, assetId);
    if (!asset || asset.kind !== "audio") {
      return { ok: false, reason: "no_asset", item };
    }
    let mediaPath: string;
    try {
      mediaPath = assetStore.getAdapter().resolvePath(asset.storageKey);
    } catch {
      return { ok: false, reason: "missing_file", item };
    }
    return {
      ok: true,
      item,
      assetId,
      mediaPath,
      youtube: null,
      originalName: asset.originalName ?? null,
    };
  }

  if (item.type !== "video") {
    return { ok: false, reason: "not_video", item };
  }

  if (item.payload.sourceKind === "youtube") {
    const videoId = item.payload.videoId?.trim();
    const sourceUrl = item.payload.sourceUrl?.trim();
    if (!videoId) {
      return { ok: false, reason: "bad_youtube", item };
    }
    return {
      ok: true,
      item,
      assetId: `yt_${videoId}`,
      mediaPath: null,
      youtube: {
        videoId,
        sourceUrl: sourceUrl || `https://www.youtube.com/watch?v=${videoId}`,
      },
      originalName: null,
    };
  }

  const assetId = item.payload.assetId;
  const asset = assetStore.get(session.id, assetId);
  if (!asset || asset.kind !== "video") {
    return { ok: false, reason: "no_asset", item };
  }
  let mediaPath: string;
  try {
    mediaPath = assetStore.getAdapter().resolvePath(asset.storageKey);
  } catch {
    return { ok: false, reason: "missing_file", item };
  }
  return {
    ok: true,
    item,
    assetId,
    mediaPath,
    youtube: null,
    originalName: asset.originalName ?? null,
  };
}

function hintForReason(
  reason: ResolveFail["reason"],
  item: PlaylistItem | null,
): string {
  switch (reason) {
    case "not_video":
      if (item?.type === "text" || item?.type === "bible") {
        return "text_only_no_audio";
      }
      return "not_local_av";
    case "no_item":
      return "no_item";
    case "no_asset":
    case "missing_file":
      return "missing_local_file";
    case "bad_youtube":
      return "youtube_invalid";
    default:
      return "unsupported";
  }
}

/**
 * Finish job from caption/ASR cues.
 *
 * Native path (YouTube/sidecar matching UI lang): keep clocks + words as-is,
 * never Marian, exact timing (no intro-hold). ASR fallback may shape + assist.
 */
async function finishWithCues(options: {
  sessionId: string;
  gen: number;
  itemId: string;
  assetId: string;
  cues: import("@promedia/shared").SubtitleCue[];
  sourceLang: string | null;
  targetLang: SubtitleTargetLang;
  alreadyEnglish: boolean;
  /** Official captions / sidecar — skip shape + MT when lang matches UI. */
  nativeTrack?: boolean;
  preserveLyricsSource?: boolean;
  hint?: string | null;
  timingMode?: SubtitleTimingMode;
  /** Optional path default (e.g. cancionero −1s). */
  leadMs?: SubtitleLeadMs;
}): Promise<boolean> {
  const {
    sessionId,
    gen,
    itemId,
    assetId,
    targetLang,
    alreadyEnglish,
    preserveLyricsSource,
    nativeTrack,
  } = options;
  const speech = filterSpeechCues(options.cues);
  let cues = nativeTrack
    ? preserveNativeCaptionCues(speech)
    : shapeLyricLikeCues(speech);
  let translated = alreadyEnglish && targetLang === "en";
  let hint = options.hint ?? null;
  let sourceLang = options.sourceLang;
  const srcBase = langBase(sourceLang) || (alreadyEnglish ? "en" : null);
  const tgtBase = langBase(targetLang) || targetLang;
  const sameLang =
    (srcBase !== null && srcBase === tgtBase) ||
    (speech.length > 0 && cuesLikelyInLanguage(speech, tgtBase));

  // Native matching-lang captions: never Marian, never reshape timings again.
  const skipMt =
    nativeTrack === true ||
    sameLang ||
    (preserveLyricsSource === true && sameLang);

  if (!cues.length) return false;

  if (!skipMt && (targetLang !== "en" || !alreadyEnglish)) {
    const live = sessionStore.get(sessionId);
    if (!live?.live.subtitleAgent?.enabled || !isCurrent(sessionId, gen)) {
      return false;
    }
    patchAndBroadcast(live, { status: "translating" });
    const mt = await translateCuesLocal({
      cues,
      sourceLang,
      targetLang,
      alreadyEnglish,
      // Cross-lang only reaches here; preserve still blocks Marian on lyrics.
      preserveLyricsSource: preserveLyricsSource === true,
    });
    cues = shapeLyricLikeCues(filterSpeechCues(mt.cues));
    translated = mt.translated || translated;
    if (mt.note && !mt.translated) {
      hint =
        mt.note === "mt_lyrics_preserved" ||
        mt.note.startsWith("mt_unavailable")
          ? "mt_source_only"
          : mt.note;
    }
    if (!sourceLang && alreadyEnglish) {
      sourceLang = "en";
    }
  } else if (sameLang && !sourceLang) {
    sourceLang = tgtBase;
  }

  if (!isCurrent(sessionId, gen)) return false;
  const liveFinal = sessionStore.get(sessionId);
  if (!liveFinal?.live.subtitleAgent?.enabled) return false;
  if (!cues.length) return false;

  const timingMode: SubtitleTimingMode =
    options.timingMode ?? (nativeTrack ? "exact" : "assisted");

  if (nativeTrack && !hint) {
    hint = "native_captions";
  }

  clearYtHardFail(sessionId);
  const leadPatch =
    options.leadMs !== undefined
      ? { leadMs: normalizeSubtitleLeadMs(options.leadMs) }
      : {};
  patchAndBroadcast(liveFinal, {
    status: "ready",
    cues,
    sourceLang,
    translated,
    timingMode,
    hint: hint ?? (nativeTrack ? "native_captions" : null),
    itemId,
    assetId,
    ...leadPatch,
  });
  return true;
}

function isStickyMissingBin(agent: SubtitleAgentState | null | undefined): boolean {
  return Boolean(
    agent && agent.status === "error" && isMissingBinHint(agent.hint),
  );
}

function isStickyHardYoutube(
  agent: SubtitleAgentState | null | undefined,
): boolean {
  return Boolean(
    agent && agent.status === "error" && isHardYoutubeHint(agent.hint),
  );
}

/** Cancionero miss is intentional — do not fall through to Whisper on Go Live. */
function isStickyCancioneroMiss(
  agent: SubtitleAgentState | null | undefined,
): boolean {
  return Boolean(
    agent &&
      agent.status === "error" &&
      (agent.hint === "cancionero_no_match" ||
        agent.hint === "cancionero_empty" ||
        agent.hint === "cancionero_no_songs"),
  );
}

/**
 * Timing-only probe for karaoke forced align. Captions/ASR clocks decide
 * *when* to reveal each Biblioteca line — never rewrite the text.
 */
async function fetchCancioneroTimingCues(options: {
  sessionId: string;
  gen: number;
  targetLang: SubtitleTargetLang;
  resolved: ResolveOk;
  /** Official cancionero lines → Whisper `-prompt` (clocks only). */
  alignPrompt?: string | null;
  onStatus?: (
    status: "downloading" | "extracting" | "transcribing",
    hint: string,
  ) => void;
}): Promise<{ cues: import("@promedia/shared").SubtitleCue[]; assetId: string } | null> {
  const { sessionId, gen, targetLang, resolved } = options;
  let mediaPath = resolved.mediaPath;
  let assetId = resolved.assetId;

  if (resolved.youtube) {
    options.onStatus?.("downloading", "cancionero_sync");
    const preferLangs = expandCaptionPreferLangs([targetLang]);
    try {
      const caps = await tryYoutubeCaptions({
        videoId: resolved.youtube.videoId,
        sourceUrl: resolved.youtube.sourceUrl,
        preferLangs,
      });
      if (!isCurrent(sessionId, gen)) return null;
      if (caps && caps.cues.length > 0) {
        const speech = filterSpeechCues(caps.cues);
        if (speech.length > 0) {
          console.log(
            `[subtitle] cancionero timing from YouTube captions kind=${caps.trackKind} n=${speech.length}`,
          );
          return { cues: speech, assetId: caps.assetId };
        }
      }
    } catch (err) {
      console.warn(
        "[subtitle] cancionero timing captions failed:",
        err instanceof Error ? err.message : err,
      );
    }

    options.onStatus?.("downloading", "cancionero_sync");
    try {
      const yt = await ensureYoutubeAudio({
        videoId: resolved.youtube.videoId,
        sourceUrl: resolved.youtube.sourceUrl,
      });
      if (!isCurrent(sessionId, gen)) return null;
      mediaPath = yt.mediaPath;
      assetId = yt.assetId;
    } catch (err) {
      console.warn(
        "[subtitle] cancionero timing YT audio failed:",
        err instanceof Error ? err.message : err,
      );
      return null;
    }
  } else if (mediaPath) {
    options.onStatus?.("extracting", "cancionero_sync");
    try {
      const local = await tryLocalCaptionTracks({
        assetId,
        mediaPath,
        originalName: resolved.originalName,
      });
      if (!isCurrent(sessionId, gen)) return null;
      if (local && local.cues.length > 0) {
        const speech = filterSpeechCues(local.cues);
        if (speech.length > 0) {
          console.log(
            `[subtitle] cancionero timing from local captions origin=${local.origin} n=${speech.length}`,
          );
          return { cues: speech, assetId };
        }
      }
    } catch (err) {
      console.warn(
        "[subtitle] cancionero timing local captions failed:",
        err instanceof Error ? err.message : err,
      );
    }
  }

  if (!mediaPath) return null;
  const exists = await mediaFileExists(mediaPath);
  if (!exists) return null;

  options.onStatus?.("transcribing", "cancionero_sync");
  try {
    const asr = await transcribeLocalFile({
      mediaPath,
      assetId,
      translateToEn: false,
      targetLang,
      // Karaoke: prompt biases clocks; ASR words never reach the overlay.
      prompt: options.alignPrompt ?? null,
      onStatus: (st) => options.onStatus?.(st, "cancionero_sync"),
    });
    if (!isCurrent(sessionId, gen)) return null;
    const speech = filterSpeechCues(asr.cues);
    if (speech.length === 0) return null;
    console.log(
      `[subtitle] karaoke timing from ASR n=${speech.length} prompt=${Boolean(options.alignPrompt)}`,
    );
    return { cues: speech, assetId };
  } catch (err) {
    console.warn(
      "[subtitle] karaoke timing ASR failed:",
      err instanceof Error ? err.message : err,
    );
    return null;
  }
}

/**
 * Guion Audio + YouTube — karaoke when the Biblioteca cancionero matches.
 * Audio miss → sticky modal (no Whisper-as-lyrics for songs).
 * YouTube miss → return false so captions / Whisper handle general videos.
 * Returns true when the job should stop (karaoke hit or sticky Audio miss).
 */
async function tryCancioneroLyricPath(options: {
  sessionId: string;
  gen: number;
  targetLang: SubtitleTargetLang;
  resolved: ResolveOk;
}): Promise<boolean> {
  const { sessionId, gen, targetLang, resolved } = options;
  const item = resolved.item;
  const isAudio = item.type === "audio";
  const isYoutube =
    item.type === "video" && item.payload.sourceKind === "youtube";
  if (!isAudio && !isYoutube) {
    return false;
  }

  const titles: string[] = [];
  const artists: string[] = [];
  let durationMs = 0;

  if (isAudio) {
    const payloadTitle =
      typeof item.payload.title === "string" ? item.payload.title.trim() : "";
    if (payloadTitle) titles.push(payloadTitle);
    const fromFile = titleFromAudioFilename(resolved.originalName);
    if (fromFile) titles.push(fromFile);
    if (resolved.mediaPath) {
      durationMs = await probeMediaDurationMs(resolved.mediaPath);
    }
  } else if (isYoutube && resolved.youtube) {
    const payloadTitle =
      typeof item.payload.title === "string" ? item.payload.title.trim() : "";
    if (payloadTitle) titles.push(payloadTitle);

    const liveLookup = sessionStore.get(sessionId);
    if (!liveLookup?.live.subtitleAgent?.enabled) return true;
    patchAndBroadcast(liveLookup, {
      enabled: true,
      targetLang,
      status: "downloading",
      itemId: item.id,
      assetId: resolved.assetId,
      hint: "cancionero_lookup",
      cues: [],
      translated: false,
      sourceLang: null,
    });

    const meta = await fetchYoutubeMeta({
      videoId: resolved.youtube.videoId,
      sourceUrl: resolved.youtube.sourceUrl,
    });
    if (!isCurrent(sessionId, gen)) return true;
    if (meta?.title) titles.push(meta.title);
    if (meta?.track) titles.push(meta.track);
    if (meta?.artist) artists.push(meta.artist);
    if (meta?.durationMs) durationMs = meta.durationMs;
  }

  const match = await findCancioneroSong({ titles, artists });
  if (!isCurrent(sessionId, gen)) return true;

  if (!match.ok) {
    // YouTube without cancionero → fall through to captions / Whisper ASR
    // (sermons, teachings, general videos). Audio stays sticky modal.
    if (isYoutube) {
      console.warn(
        `[subtitle] cancionero miss for YouTube (${match.reason}) — captions/Whisper fallback tried=${JSON.stringify(match.tried)}`,
      );
      return false;
    }
    const live = sessionStore.get(sessionId);
    if (!live?.live.subtitleAgent?.enabled) return true;
    const hint =
      match.reason === "no_songs"
        ? "cancionero_no_songs"
        : match.reason === "no_query"
          ? "cancionero_no_match"
          : "cancionero_no_match";
    console.warn(
      `[subtitle] cancionero miss (${match.reason}) tried=${JSON.stringify(match.tried)}`,
    );
    patchAndBroadcast(live, {
      enabled: true,
      targetLang,
      status: "error",
      itemId: item.id,
      assetId: resolved.assetId,
      cues: [],
      translated: false,
      sourceLang: null,
      hint,
      timingMode: "exact",
    });
    return true;
  }

  const lines = songCancioneroLines(match.song);
  if (!lines.length) {
    const live = sessionStore.get(sessionId);
    if (!live?.live.subtitleAgent?.enabled) return true;
    patchAndBroadcast(live, {
      enabled: true,
      targetLang,
      status: "error",
      itemId: item.id,
      assetId: resolved.assetId,
      cues: [],
      translated: false,
      sourceLang: null,
      hint: "cancionero_empty",
      timingMode: "exact",
    });
    return true;
  }

  // Karaoke forced align: captions/ASR clocks only — Biblioteca text exact.
  const liveSync = sessionStore.get(sessionId);
  if (!liveSync?.live.subtitleAgent?.enabled) return true;
  const alignPrompt = buildKaraokeAlignPrompt(lines);
  const timing = await fetchCancioneroTimingCues({
    sessionId,
    gen,
    targetLang,
    resolved,
    alignPrompt,
    onStatus: (status, hint) => {
      if (!isCurrent(sessionId, gen)) return;
      const live = sessionStore.get(sessionId);
      if (!live?.live.subtitleAgent?.enabled) return;
      patchAndBroadcast(live, {
        status,
        hint,
        itemId: item.id,
        assetId: resolved.assetId,
        cues: [],
        translated: false,
        sourceLang: null,
      });
    },
  });
  if (!isCurrent(sessionId, gen)) return true;

  let lineCues = songToSubtitleCues(match.song, durationMs);
  let synced = false;
  let assetId = resolved.assetId;
  if (timing && timing.cues.length > 0) {
    const aligned = alignCancioneroLinesToTiming(
      lines,
      timing.cues,
      durationMs,
    );
    lineCues = aligned.cues;
    synced = aligned.synced;
    assetId = timing.assetId;
    console.log(
      `[subtitle] karaoke align synced=${synced} aligned=${aligned.alignedCount}/${lines.length}`,
    );
  }

  // Prefer 1–2 / max 2 display lines (exact Biblioteca text, newlines kept).
  const cues = packKaraokeDisplayCues(lineCues);

  if (!cues.length) {
    const live = sessionStore.get(sessionId);
    if (!live?.live.subtitleAgent?.enabled) return true;
    patchAndBroadcast(live, {
      enabled: true,
      targetLang,
      status: "error",
      itemId: item.id,
      assetId,
      cues: [],
      translated: false,
      sourceLang: null,
      hint: "cancionero_empty",
      timingMode: "exact",
    });
    return true;
  }

  console.log(
    `[subtitle] karaoke match score=${match.score} q="${match.matchedQuery}" song="${match.song.title}" lines=${lineCues.length} cues=${cues.length} synced=${synced}`,
  );

  const ok = await finishWithCues({
    sessionId,
    gen,
    itemId: item.id,
    assetId,
    cues,
    sourceLang: targetLang,
    targetLang,
    alreadyEnglish: targetLang === "en",
    // Exact Biblioteca text; exact timingMode so we wait for the voice
    // (no intro-hold race). leadMs −1s reveals each group 1s before onset.
    nativeTrack: true,
    preserveLyricsSource: true,
    timingMode: "exact",
    hint: synced ? "cancionero_match_synced" : "cancionero_match",
    leadMs: CANCIONERO_DEFAULT_LEAD_MS,
  });
  return ok || true;
}

async function missingBinStillAbsent(hint: string): Promise<boolean> {
  if (hint === "yt_dlp_missing") {
    return !(await ytDlpAvailable());
  }
  if (hint === "ffmpeg_missing") {
    return !(await ffmpegAvailable());
  }
  if (hint === "whisper_cpp_missing") {
    return !(await whisperCliAvailable());
  }
  return true;
}

/**
 * Apply enable/lang patch and kick off local ASR when appropriate.
 * Always returns a HubMessage (session.updated).
 */
export function setSubtitleAgent(
  session: SessionRecord,
  patch: {
    enabled?: boolean;
    targetLang?: SubtitleTargetLang;
    fontSize?: SubtitleFontSize;
    leadMs?: SubtitleLeadMs;
    itemId?: string;
  },
): HubMessage {
  const prev = session.live.subtitleAgent ?? emptySubtitleAgentState();
  const enabled = patch.enabled ?? prev.enabled;
  const targetLang = patch.targetLang ?? prev.targetLang;
  const fontSize = patch.fontSize ?? prev.fontSize ?? "md";
  const leadMs =
    patch.leadMs !== undefined
      ? normalizeSubtitleLeadMs(patch.leadMs)
      : normalizeSubtitleLeadMs(prev.leadMs ?? DEFAULT_SUBTITLE_LEAD_MS);
  const wasOff = !prev.enabled;
  const stickyMissing = isStickyMissingBin(prev);
  // Size / lead-only: persist without restarting ASR.
  const displayOnly =
    (patch.fontSize !== undefined || patch.leadMs !== undefined) &&
    patch.enabled === undefined &&
    patch.targetLang === undefined &&
    patch.itemId === undefined;

  if (!enabled) {
    bumpJob(session.id);
    clearYtHardFail(session.id);
    session.live.subtitleAgent = {
      ...emptySubtitleAgentState(targetLang),
      enabled: false,
      status: "off",
      fontSize,
      leadMs,
      updatedAt: new Date().toISOString(),
    };
    return sessionStore.touchSubtitleAgent(session);
  }

  if (displayOnly) {
    session.live.subtitleAgent = {
      ...prev,
      enabled: true,
      fontSize,
      leadMs,
      updatedAt: new Date().toISOString(),
    };
    return sessionStore.touchSubtitleAgent(session);
  }

  // Keep missing-binary / hard YT / cancionero-miss visible when focus/lang
  // changes; only clear on toggle-on (wasOff) so Control shows the hint
  // instead of retry spam (and never invents lyrics via Whisper after a miss).
  const stickyHardYt = isStickyHardYoutube(prev);
  const stickyCancionero = isStickyCancioneroMiss(prev);
  const keepSticky =
    (stickyMissing || stickyHardYt || stickyCancionero) && !wasOff;
  if (wasOff) {
    clearYtHardFail(session.id);
  }

  session.live.subtitleAgent = {
    ...prev,
    enabled: true,
    targetLang,
    fontSize,
    leadMs,
    status: keepSticky
      ? "error"
      : prev.status === "ready" && prev.targetLang === targetLang
        ? "ready"
        : "idle",
    hint: keepSticky ? prev.hint : null,
    updatedAt: new Date().toISOString(),
  };

  const message = sessionStore.touchSubtitleAgent(session);

  // Fire-and-forget local job (does not block WS).
  // force=true only when enabling from off (user toggle) — retries missing bins / YT.
  void runSubtitleJob(session.id, patch.itemId, { force: wasOff });

  return message;
}

/** Re-run when Go Live changes the A/V item under an active agent. */
export function maybeRefreshSubtitleAgent(session: SessionRecord): void {
  const agent = session.live.subtitleAgent;
  if (!agent?.enabled) return;
  // Missing yt-dlp/ffmpeg: do not tight-loop; quiet re-check inside the job.
  void runSubtitleJob(session.id, session.live.currentItemId ?? undefined, {
    force: false,
  });
}

async function runSubtitleJob(
  sessionId: string,
  preferredItemId?: string,
  options?: { force?: boolean },
): Promise<void> {
  const force = options?.force === true;
  const sessionPeek = sessionStore.get(sessionId);
  const agentPeek = sessionPeek?.live.subtitleAgent;
  if (
    !force &&
    agentPeek &&
    isStickyMissingBin(agentPeek) &&
    agentPeek.hint &&
    (await missingBinStillAbsent(agentPeek.hint))
  ) {
    // Still missing — fail once; wait for toggle-off/on or deps to appear.
    return;
  }
  if (
    !force &&
    agentPeek &&
    isStickyHardYoutube(agentPeek) &&
    ytHardFailInBackoff(sessionId)
  ) {
    // Same hard YT failure — wait for backoff or user toggle-on.
    return;
  }
  if (!force && agentPeek && isStickyCancioneroMiss(agentPeek)) {
    // Cancionero miss — wait for toggle-on or library update (no Whisper).
    return;
  }

  const gen = bumpJob(sessionId);
  const session = sessionStore.get(sessionId);
  if (!session) return;

  const agent = session.live.subtitleAgent;
  if (!agent?.enabled) return;

  const targetLang = agent.targetLang;
  const resolved = resolveMediaSource(session, preferredItemId);

  if (!resolved.ok) {
    if (!isCurrent(sessionId, gen)) return;
    patchAndBroadcast(session, {
      enabled: true,
      targetLang,
      status: "unsupported",
      itemId: resolved.item?.id ?? null,
      assetId: null,
      sourceLang: null,
      translated: false,
      cues: [],
      hint: hintForReason(resolved.reason, resolved.item),
    });
    return;
  }

  // Same asset already ready for this target — keep cues.
  if (
    agent.status === "ready" &&
    agent.assetId === resolved.assetId &&
    agent.targetLang === targetLang &&
    agent.cues.length > 0
  ) {
    if (agent.itemId !== resolved.item.id) {
      patchAndBroadcast(session, { itemId: resolved.item.id });
    }
    return;
  }

  let mediaPath: string | null = resolved.mediaPath;
  let assetId = resolved.assetId;

  try {
    await ensureSubtitleDirs();
    if (!isCurrent(sessionId, gen)) return;

    // Karaoke when cancionero matches (Audio / YouTube).
    // Audio miss → sticky modal. YouTube miss → captions/Whisper below.
    // Local file video skips this path and uses sidecar / Whisper.
    const cancioneroDone = await tryCancioneroLyricPath({
      sessionId,
      gen,
      targetLang,
      resolved,
    });
    if (cancioneroDone) return;
    if (!isCurrent(sessionId, gen)) return;

    if (resolved.youtube) {
      patchAndBroadcast(session, {
        enabled: true,
        targetLang,
        status: "downloading",
        itemId: resolved.item.id,
        assetId,
        hint: "youtube_captions",
        cues: [],
        translated: false,
        sourceLang: null,
      });

      // Native agent: only captions matching UI language (no EN→Marian fallback).
      // Missing matching track → Whisper `-l <targetLang>` below.
      const preferLangs = expandCaptionPreferLangs([targetLang]);
      const caps = await tryYoutubeCaptions({
        videoId: resolved.youtube.videoId,
        sourceUrl: resolved.youtube.sourceUrl,
        preferLangs,
      });
      if (!isCurrent(sessionId, gen)) return;

      if (caps && caps.cues.length > 0) {
        const capBase = langBase(caps.sourceLang);
        const tgt = langBase(targetLang);
        // Only accept tracks that match the UI language (es/es-419…).
        if (capBase && tgt && capBase === tgt) {
          const ok = await finishWithCues({
            sessionId,
            gen,
            itemId: resolved.item.id,
            assetId: caps.assetId,
            cues: caps.cues,
            sourceLang: caps.sourceLang,
            targetLang,
            alreadyEnglish: false,
            nativeTrack: true,
            preserveLyricsSource: true,
            timingMode: "exact",
            hint:
              caps.trackKind === "manual"
                ? "native_captions"
                : "native_captions_auto",
          });
          if (ok) return;
        } else {
          console.warn(
            `[subtitle] skipping cross-lang YouTube captions (${caps.sourceLang}) for UI=${targetLang} — ASR -l ${targetLang}`,
          );
        }
        console.warn(
          "[subtitle] YouTube captions unused — falling back to Whisper ASR in target language",
        );
      }

      patchAndBroadcast(session, {
        enabled: true,
        targetLang,
        status: "downloading",
        itemId: resolved.item.id,
        assetId,
        hint: "youtube_audio",
        cues: [],
        translated: false,
        sourceLang: null,
      });

      const yt = await ensureYoutubeAudio({
        videoId: resolved.youtube.videoId,
        sourceUrl: resolved.youtube.sourceUrl,
      });
      if (!isCurrent(sessionId, gen)) return;
      mediaPath = yt.mediaPath;
      assetId = yt.assetId;
    }

    if (!mediaPath) {
      if (!isCurrent(sessionId, gen)) return;
      patchAndBroadcast(session, {
        enabled: true,
        targetLang,
        status: "error",
        itemId: resolved.item.id,
        assetId,
        cues: [],
        hint: "missing_local_file",
      });
      return;
    }

    const exists = await mediaFileExists(mediaPath);
    if (!exists) {
      if (!isCurrent(sessionId, gen)) return;
      patchAndBroadcast(session, {
        enabled: true,
        targetLang,
        status: "unsupported",
        itemId: resolved.item.id,
        assetId,
        cues: [],
        hint: "missing_local_file",
      });
      return;
    }

    // Local alternate agent (no Whisper): sidecars in data/subtitles/sidecars/
    // or embedded softsubs via ffmpeg. Prefer before ASR — reliable on Mac when
    // whisper-cli/ggml is missing or music-only ASR returns empty cues.
    if (!resolved.youtube) {
      patchAndBroadcast(session, {
        enabled: true,
        targetLang,
        status: "extracting",
        itemId: resolved.item.id,
        assetId,
        hint: "local_captions",
        cues: [],
        translated: false,
        sourceLang: null,
      });
      const local = await tryLocalCaptionTracks({
        assetId,
        mediaPath,
        originalName: resolved.originalName,
      });
      if (!isCurrent(sessionId, gen)) return;
      if (local && local.cues.length > 0) {
        const localBase = langBase(local.sourceLang);
        const tgt = langBase(targetLang);
        const langMatches =
          !local.sourceLang ||
          !localBase ||
          !tgt ||
          localBase === tgt ||
          cuesLikelyInLanguage(local.cues, tgt);
        if (langMatches) {
          const ok = await finishWithCues({
            sessionId,
            gen,
            itemId: resolved.item.id,
            assetId,
            cues: local.cues,
            sourceLang: local.sourceLang ?? tgt,
            targetLang,
            alreadyEnglish: false,
            nativeTrack: true,
            preserveLyricsSource: true,
            timingMode: "exact",
            hint:
              local.origin === "sidecar"
                ? "local_sidecar"
                : "local_embedded",
          });
          if (ok) return;
        }
      }
    }

    patchAndBroadcast(session, {
      enabled: true,
      targetLang,
      status: "extracting",
      itemId: resolved.item.id,
      assetId,
      hint: null,
      cues: [],
      translated: false,
      sourceLang: null,
    });

    // Prefer Whisper translate→EN when target is English (one model pass).
    const translateToEn = targetLang === "en";

    const asr = await transcribeLocalFile({
      mediaPath,
      assetId,
      translateToEn,
      targetLang,
      onStatus: (st) => {
        if (!isCurrent(sessionId, gen)) return;
        const live = sessionStore.get(sessionId);
        if (!live?.live.subtitleAgent?.enabled) return;
        patchAndBroadcast(live, { status: st, hint: null });
      },
    });

    if (!isCurrent(sessionId, gen)) return;
    const liveAfterAsr = sessionStore.get(sessionId);
    if (!liveAfterAsr?.live.subtitleAgent?.enabled) return;

    const okAsr = await finishWithCues({
      sessionId,
      gen,
      itemId: resolved.item.id,
      assetId,
      cues: asr.cues,
      sourceLang: asr.sourceLang ?? (asr.whisperTranslatedToEn ? "en" : targetLang),
      targetLang,
      alreadyEnglish: asr.whisperTranslatedToEn,
      // ASR fallback: shape + assisted intro-hold; skip Marian when same lang.
      nativeTrack: false,
      preserveLyricsSource: false,
      timingMode: "assisted",
      hint: "asr_fallback",
    });
    if (okAsr) return;

    // Job "succeeded" but every cue was filtered (e.g. only [MÚSICA]) or ASR
    // returned nothing — do NOT show a green "listo" with a blank overlay.
    if (!isCurrent(sessionId, gen)) return;
    const liveEmpty = sessionStore.get(sessionId);
    if (!liveEmpty?.live.subtitleAgent?.enabled) return;
    const hasWhisper = await whisperCliAvailable();
    patchAndBroadcast(liveEmpty, {
      status: "error",
      cues: [],
      sourceLang: asr.sourceLang,
      translated: false,
      hint: hasWhisper ? "no_speech" : "whisper_cpp_missing",
      itemId: resolved.item.id,
      assetId,
    });
  } catch (err) {
    const raw = err instanceof Error ? err.message : "asr_failed";
    const missingHint = missingBinHintFromError(err);
    const hint = missingHint ?? raw.slice(0, 180);
    // Log missing binaries once; sticky state prevents retry spam.
    if (missingHint) {
      console.error(`[subtitle] job failed: ${missingHint}`);
    } else if (hint.startsWith("ffmpeg_extract_failed")) {
      console.error("[subtitle] job failed: ffmpeg_extract_failed");
    } else {
      console.error("[subtitle] job failed:", raw);
    }
    if (isHardYoutubeHint(hint)) {
      markYtHardFail(sessionId);
    }
    // Drop bad YouTube cache so the next user toggle re-downloads once.
    if (
      hint.startsWith("ffmpeg_extract_failed") &&
      mediaPath &&
      resolved.youtube
    ) {
      void rm(mediaPath, { force: true }).catch(() => undefined);
    }
    if (!isCurrent(sessionId, gen)) return;
    const live = sessionStore.get(sessionId);
    if (!live) return;
    // Prefer stable hint codes for Control (strip long stderr tails).
    const uiHint = hint.startsWith("ffmpeg_extract_failed")
      ? "ffmpeg_extract_failed"
      : hint.startsWith("whisper_cpp_failed")
        ? "whisper_cpp_failed"
        : hint;
    patchAndBroadcast(live, {
      status: "error",
      cues: [],
      hint: uiHint,
      itemId: resolved.item.id,
      assetId: resolved.assetId,
    });
  }
}
