import type { SubtitleCue } from "@promedia/shared";
import { cleanCueText, isNonSpeechCueText } from "./subtitle-cue-text.js";
import { cuesLookLikeLyrics } from "./subtitle-lang-detect.js";

/**
 * Gold-standard lyric cues (e.g. Julio’s Abigail Araujo “Cuán grande es Él”
 * manual YouTube track) are short cancionero lines (~20–55 chars), punctuated,
 * timed to sung phrases — not ASR paragraphs or one-word karaoke flashes.
 *
 * Shape Whisper / auto-caption fallbacks toward that density without changing
 * overlay lead (#77) or intro-hold (#80).
 */

/** Target characters per projected lyric line (Spanish hymn / chorus density). */
export const LYRIC_TARGET_CHARS = 48;
/** Soft ceiling before we split a long ASR paragraph. */
export const LYRIC_SPLIT_CHARS = 64;
/** Merge micro-cues until we reach roughly this many characters. */
export const LYRIC_MERGE_CHARS = 40;

const WORD_RE = /[\p{L}\p{N}'’¿¡]+/gu;

function speechOnly(cues: SubtitleCue[]): SubtitleCue[] {
  const out: SubtitleCue[] = [];
  for (const cue of cues) {
    const text = cleanCueText(cue.text);
    if (!text || isNonSpeechCueText(text)) continue;
    out.push({ ...cue, text });
  }
  return out;
}

function reId(cues: SubtitleCue[]): SubtitleCue[] {
  return cues.map((c, i) => ({ ...c, id: `c${i}` }));
}

/** Light lyric punctuation: collapse spaces, keep ¿¡…, drop trailing comma spam. */
export function normalizeLyricPunctuation(text: string): string {
  let t = cleanCueText(text);
  if (!t) return "";
  // Space before Spanish inverted marks if glued mid-line.
  t = t.replace(/\s*([¿¡])/g, " $1").replace(/\s+/g, " ").trim();
  // Drop dangling connectors at end of a fragment.
  t = t.replace(/[,:;]\s*$/u, "").trim();
  // Capitalize first letter when the line starts with a word (cancionero feel).
  if (/^[\p{Ll}]/u.test(t)) {
    t = t.charAt(0).toLocaleUpperCase("es") + t.slice(1);
  }
  return t;
}

function median(nums: number[]): number {
  if (!nums.length) return 0;
  const s = [...nums].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)] ?? 0;
}

function wordCount(text: string): number {
  return (text.match(WORD_RE) || []).length;
}

/**
 * Split a long paragraph into lyric-sized chunks on punctuation / word wraps,
 * distributing the parent cue’s time window proportionally by character weight.
 */
export function splitLongLyricText(
  text: string,
  startMs: number,
  endMs: number,
  maxChars = LYRIC_SPLIT_CHARS,
): Array<{ text: string; startMs: number; endMs: number }> {
  const cleaned = normalizeLyricPunctuation(text);
  if (!cleaned) return [];
  if (cleaned.length <= maxChars) {
    return [{ text: cleaned, startMs, endMs }];
  }

  const parts: string[] = [];
  // Prefer phrase breaks: . ! ? ; : — … or commas when chunks stay long.
  const rough = cleaned
    .split(/(?<=[.!?…])\s+|(?<=[;:—])\s+|(?<=,)\s+(?=.{12,})/u)
    .map((p) => p.trim())
    .filter(Boolean);

  const pushWrapped = (chunk: string) => {
    if (chunk.length <= maxChars) {
      parts.push(chunk);
      return;
    }
    const words = chunk.split(/\s+/).filter(Boolean);
    let buf = "";
    for (const w of words) {
      const next = buf ? `${buf} ${w}` : w;
      if (next.length > maxChars && buf) {
        parts.push(buf);
        buf = w;
      } else {
        buf = next;
      }
    }
    if (buf) parts.push(buf);
  };

  for (const piece of rough.length ? rough : [cleaned]) {
    pushWrapped(piece);
  }

  const totalChars = parts.reduce((s, p) => s + Math.max(1, p.length), 0);
  const span = Math.max(500, endMs - startMs);
  const out: Array<{ text: string; startMs: number; endMs: number }> = [];
  let cursor = startMs;
  for (let i = 0; i < parts.length; i++) {
    const piece = normalizeLyricPunctuation(parts[i]!);
    if (!piece) continue;
    const weight = Math.max(1, piece.length) / totalChars;
    const dur = Math.max(400, Math.round(span * weight));
    let segEnd = cursor + dur;
    if (i === parts.length - 1) segEnd = Math.max(segEnd, endMs);
    if (segEnd <= cursor) segEnd = cursor + 400;
    out.push({ text: piece, startMs: cursor, endMs: segEnd });
    cursor = segEnd;
  }
  return out;
}

/**
 * Merge one-word / flash cues into cancionero-length lines (auto karaoke → lyrics).
 */
export function mergeMicroLyricCues(
  cues: SubtitleCue[],
  targetChars = LYRIC_MERGE_CHARS,
): SubtitleCue[] {
  if (cues.length < 4) return cues.map((c) => ({
    ...c,
    text: normalizeLyricPunctuation(c.text),
  }));

  const lengths = cues.map((c) => cleanCueText(c.text).length);
  const words = cues.map((c) => wordCount(c.text));
  const durs = cues.map((c) => Math.max(0, c.endMs - c.startMs));
  const micro =
    median(lengths) <= 18 ||
    median(words) <= 2 ||
    (median(durs) > 0 && median(durs) < 900);
  if (!micro) {
    return cues.map((c) => ({
      ...c,
      text: normalizeLyricPunctuation(c.text),
    }));
  }

  const merged: SubtitleCue[] = [];
  let bufText = "";
  let bufStart = 0;
  let bufEnd = 0;

  const flush = () => {
    const text = normalizeLyricPunctuation(bufText);
    if (!text) {
      bufText = "";
      return;
    }
    merged.push({
      id: `m${merged.length}`,
      startMs: bufStart,
      endMs: Math.max(bufEnd, bufStart + 400),
      text,
    });
    bufText = "";
  };

  for (const cue of cues) {
    const t = cleanCueText(cue.text);
    if (!t) continue;
    if (!bufText) {
      bufText = t;
      bufStart = cue.startMs;
      bufEnd = cue.endMs;
      continue;
    }
    const gap = cue.startMs - bufEnd;
    const next = `${bufText} ${t}`;
    // New line on a long pause or when the buffer is already a full lyric line.
    if (gap > 1400 || (bufText.length >= targetChars && gap > 250)) {
      flush();
      bufText = t;
      bufStart = cue.startMs;
      bufEnd = cue.endMs;
      continue;
    }
    if (next.length > LYRIC_SPLIT_CHARS && bufText.length >= Math.floor(targetChars * 0.7)) {
      flush();
      bufText = t;
      bufStart = cue.startMs;
      bufEnd = cue.endMs;
      continue;
    }
    bufText = next;
    bufEnd = Math.max(bufEnd, cue.endMs);
  }
  flush();
  return merged;
}

/**
 * Ensure cue windows meet at edges so lines are not late vs the previous end
 * (Whisper often leaves a dead gap after a segment while the next phrase
 * already started). Does **not** pull the first cue to t=0 — intro-hold does.
 */
export function tightenLyricCueStarts(cues: SubtitleCue[]): SubtitleCue[] {
  if (cues.length < 2) return cues;
  const out: SubtitleCue[] = [{ ...cues[0]! }];
  for (let i = 1; i < cues.length; i++) {
    const prev = out[i - 1]!;
    const cur = cues[i]!;
    let start = cur.startMs;
    const gap = start - prev.endMs;
    // Close small “late” gaps so the next line appears with the sung onset.
    if (gap > 0 && gap <= 750) {
      start = prev.endMs;
    }
    let end = Math.max(cur.endMs, start + 400);
    // Avoid overlapping previous text for more than a blink.
    if (start < prev.endMs - 80) {
      start = prev.endMs;
      end = Math.max(end, start + 400);
    }
    out.push({ ...cur, startMs: start, endMs: end });
  }
  return out;
}

/**
 * Official / sidecar captions for the **native** agent: keep VTT/SRT clocks
 * and wording as-is. Only drop non-speech labels and light space cleanup —
 * never merge/split/tighten (those shift times vs YouTube lyric videos).
 */
export function preserveNativeCaptionCues(cues: SubtitleCue[]): SubtitleCue[] {
  const speech = speechOnly(cues);
  if (!speech.length) return [];
  return reId(
    speech.map((c) => ({
      ...c,
      // Clean tags/entities only — do not capitalize or re-punctuate lyrics.
      text: cleanCueText(c.text),
    })),
  );
}

/**
 * Shape ASR / weak auto-caption fallbacks toward lyric-track density.
 * Native matching-lang tracks must use {@link preserveNativeCaptionCues}
 * instead — do not call this on official YouTube manuals.
 */
export function shapeLyricLikeCues(cues: SubtitleCue[]): SubtitleCue[] {
  const speech = speechOnly(cues);
  if (!speech.length) return [];

  // Already gold-standard density — keep boundaries, polish wording only.
  if (cuesLookLikeLyrics(speech)) {
    return reId(
      tightenLyricCueStarts(
        speech.map((c) => ({
          ...c,
          text: normalizeLyricPunctuation(c.text),
        })),
      ),
    );
  }

  const merged = mergeMicroLyricCues(speech);
  const split: SubtitleCue[] = [];
  for (const cue of merged) {
    const pieces = splitLongLyricText(cue.text, cue.startMs, cue.endMs);
    for (const p of pieces) {
      split.push({
        id: `s${split.length}`,
        startMs: p.startMs,
        endMs: p.endMs,
        text: p.text,
      });
    }
  }

  return reId(tightenLyricCueStarts(split));
}

/**
 * Quality score for choosing among YouTube caption files (higher = better).
 * Rewards manual tracks and cancionero-shaped cue density.
 */
export function scoreCaptionCueQuality(
  cues: SubtitleCue[],
  trackKind: "manual" | "auto" | "unknown",
): number {
  const speech = speechOnly(cues);
  if (!speech.length) return -1_000_000;
  let score = speech.length * 2;
  if (trackKind === "manual") score += 500;
  else if (trackKind === "unknown") score += 50;
  if (cuesLookLikeLyrics(speech)) score += 200;
  const lens = speech.map((c) => cleanCueText(c.text).length);
  const mid = median(lens);
  // Prefer hymn-line lengths over sermons or one-word flashes.
  if (mid >= 12 && mid <= 55) score += 80;
  else if (mid > 90) score -= 40;
  else if (mid < 6) score -= 30;
  return score;
}
