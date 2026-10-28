import type { SubtitleCue } from "@promedia/shared";

/**
 * Whisper (esp. tiny) and some ASR tracks emit non-speech sound labels instead
 * of lyrics — e.g. `[MUSIC]`, `[MÚSICA]`, `(applause)`. Those are not useful as
 * cancionero cues; prefer real transcript text over bracket labels.
 *
 * Pure music glyphs (♪♪) with no words also count as non-speech. Lyrics that
 * merely include a ♪ among words are kept.
 */
const NON_SPEECH_LABEL_RE =
  /^\s*[\[\(\{]?\s*(?:music(?:\s+playing)?|m[uú]sica(?:\s+(?:de\s+fondo|sonando))?|blank[\s_]?audio|silence|silencio|applause|aplausos|laughter|risas|noise|ruido|inaudible|speaking(?:\s+in)?\s+foreign(?:\s+language)?|foreign|♪+|♫+)\s*[\]\)\}]?\s*$/iu;

/** Strip YouTube / WebVTT karaoke tags and collapse whitespace. */
export function cleanCueText(raw: string): string {
  return raw
    .replace(/<\/?[^>]+>/g, " ")
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/\s+/g, " ")
    .trim();
}

/** True when the cue is only a sound / music label, not spoken or sung words. */
export function isNonSpeechCueText(text: string): boolean {
  const t = cleanCueText(text);
  if (!t) return true;
  if (NON_SPEECH_LABEL_RE.test(t)) return true;
  // Bracket-only leftovers after cleanup, e.g. "[]" / "( )"
  if (/^[\[\(\{\s\]\)\}]+$/.test(t)) return true;
  return false;
}

/**
 * Drop non-speech labels; re-id remaining cues. Empty input → empty output.
 * Prefer this over showing `[MÚSICA]` as if it were a lyric line.
 */
export function filterSpeechCues(cues: SubtitleCue[]): SubtitleCue[] {
  const kept: SubtitleCue[] = [];
  let i = 0;
  for (const cue of cues) {
    const text = cleanCueText(cue.text);
    if (!text || isNonSpeechCueText(text)) continue;
    kept.push({
      ...cue,
      id: `c${i++}`,
      text,
    });
  }
  return kept;
}

/** Fraction of cues that look like real speech/lyrics (0–1). */
export function speechCueRatio(cues: SubtitleCue[]): number {
  if (!cues.length) return 0;
  const speech = filterSpeechCues(cues).length;
  return speech / cues.length;
}
