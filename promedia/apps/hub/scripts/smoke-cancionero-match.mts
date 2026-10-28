/**
 * Smoke: karaoke cancionero exact lines / forced align / −1s lead / pack 2–3.
 * Usage: pnpm --filter @promedia/hub exec tsx scripts/smoke-cancionero-match.mts
 */
import type { LibrarySongPayload, SubtitleCue } from "@promedia/shared";
import {
  CANCIONERO_DEFAULT_LEAD_MS,
  LYRIC_OVERLAY_MAX_LINES,
  LYRIC_OVERLAY_PREFERRED_LINES,
  cueAtPositionWithLeadMs,
} from "@promedia/shared";
import {
  alignCancioneroLinesToTiming,
  buildKaraokeAlignPrompt,
  flattenTimingTokens,
  normalizeMatchTitle,
  packKaraokeDisplayCues,
  scoreSongMatch,
  songCancioneroLines,
  songLyricChunks,
  songLyricPhrases,
  songToSubtitleCues,
  splitLyricPhrases,
  titleFromAudioFilename,
  titleVariants,
} from "../src/subtitle-cancionero.ts";

function assert(cond: unknown, msg: string): asserts cond {
  if (!cond) throw new Error(msg);
}

const song: LibrarySongPayload = {
  title: "Cuán grande es Él",
  artist: "Tradicional",
  stanzas: [
    {
      id: "s1",
      lines: ["Señor mi Dios", "al contemplar los cielos", "te adoro a Ti"],
    },
    {
      id: "s2",
      lines: ["Cuán grande es Él", "Cuán grande es Él"],
    },
  ],
};

assert(
  normalizeMatchTitle("Cuán grande es Él (Official Video).mp3") ===
    "cuán grande es él",
  "normalize strips junk+ext",
);
assert(
  titleFromAudioFilename("Mi Cancion.m4a") === "Mi Cancion",
  "audio filename strip",
);
assert(
  titleVariants("Artist - Song Title").includes("song title"),
  "title variants split",
);

const exact = scoreSongMatch(
  song,
  "Cuán grande es Él",
  "Tradicional",
);
assert(exact >= 100, `exact+artist score got ${exact}`);

const ytLike = scoreSongMatch(
  song,
  "Cuán Grande Es Él | Official Lyric Video",
);
assert(ytLike >= 70, `youtube-like title score got ${ytLike}`);

const miss = scoreSongMatch(song, "Shape of You");
assert(miss < 55, `unrelated should miss, got ${miss}`);

// Exact Biblioteca lines — do not invent punctuation splits.
const longLine =
  "Señor mi Dios, al contemplar los cielos, el firmamento y las estrellas mil";
const phrases = splitLyricPhrases(longLine);
assert(phrases.length === 1, `exact line stays 1 phrase, got ${phrases.length}`);
assert(phrases[0] === longLine, "exact line text preserved");

const embedded = splitLyricPhrases("línea uno\nlínea dos");
assert(embedded.length === 2, "embedded newlines → exact lines");
assert(embedded[0] === "línea uno" && embedded[1] === "línea dos", "newline split");

const flat = songLyricPhrases(song);
assert(flat.length === 5, `expected 5 exact lines, got ${flat.length}`);
assert(flat[0] === "Señor mi Dios", "first exact line");
assert(flat[1] === "al contemplar los cielos", "second exact line");

const chunks = songLyricChunks(song);
assert(chunks.length === 5, `one cue per exact line, got ${chunks.length}`);
assert(
  chunks.every((c) => !c.includes("\n")),
  "chunks are single exact lines (no re-merge)",
);
assert(
  JSON.stringify(chunks) === JSON.stringify(songCancioneroLines(song)),
  "chunks === cancionero lines",
);

const cues = songToSubtitleCues(song, 120_000);
assert(cues.length === chunks.length, "cue count = exact lines");
assert(cues[0]!.startMs >= 8_000, "equal-time fallback waits through intro");
assert(cues[cues.length - 1]!.endMs >= 100_000, "last end near duration");
assert(
  cues.every((c) => c.text.trim().length > 0),
  "all cues have text",
);
assert(
  cues.every((c, i) => c.text === chunks[i]),
  "cue text is exact cancionero line",
);

// Voice sync: ASR/caption clocks → cancionero text only.
const timing: SubtitleCue[] = [
  { id: "t0", startMs: 12_000, endMs: 14_000, text: "Señor mi Dios" },
  {
    id: "t1",
    startMs: 14_500,
    endMs: 17_000,
    text: "al contemplar los cielos",
  },
  { id: "t2", startMs: 17_500, endMs: 19_000, text: "te adoro a Ti" },
  { id: "t3", startMs: 22_000, endMs: 24_000, text: "Cuán grande es Él" },
  { id: "t4", startMs: 24_500, endMs: 26_500, text: "Cuán grande es Él" },
];
const aligned = alignCancioneroLinesToTiming(chunks, timing, 60_000);
assert(aligned.synced, "enough lines voice-aligned");
assert(aligned.alignedCount >= 4, `alignedCount got ${aligned.alignedCount}`);
assert(aligned.cues.length === 5, "aligned cue count");
assert(
  aligned.cues.every((c, i) => c.text === chunks[i]),
  "aligned cues keep Biblioteca text",
);
assert(aligned.cues[0]!.startMs === 12_000, "first line locks to voice");
assert(aligned.cues[1]!.startMs === 14_500, "second line locks to voice");
// Must not race: before voice onset (with −1s lead) earlier lines only.
const exactOpts = { holdIntro: false } as const;
assert(
  cueAtPositionWithLeadMs(aligned.cues, 10_500, 0, exactOpts)?.text == null,
  "exact mode: blank before first voice moment",
);
assert(
  cueAtPositionWithLeadMs(
    aligned.cues,
    11_000,
    CANCIONERO_DEFAULT_LEAD_MS,
    exactOpts,
  )?.text === "Señor mi Dios",
  "−1s lead shows first line 1s before voice",
);
assert(
  cueAtPositionWithLeadMs(
    aligned.cues,
    13_500,
    CANCIONERO_DEFAULT_LEAD_MS,
    exactOpts,
  )?.text === "al contemplar los cielos",
  "−1s lead advances to next line 1s before its voice",
);
assert(
  cueAtPositionWithLeadMs(aligned.cues, 14_500, 0, exactOpts)?.text ===
    "al contemplar los cielos",
  "without lead, next line appears at its voice onset",
);

assert(CANCIONERO_DEFAULT_LEAD_MS === -1000, "cancionero default lead −1s");

// Karaoke-gen: token stream + prompt + 2–3 line display pack.
const stream = flattenTimingTokens(timing);
assert(stream.length >= 10, `token stream got ${stream.length}`);
assert(stream[0]!.token === "senor" || stream[0]!.token.length > 0, "stream tokens");

const prompt = buildKaraokeAlignPrompt(chunks);
assert(prompt.includes("Señor") || prompt.toLowerCase().includes("senor") || prompt.length > 10, "align prompt");
assert(prompt.length <= 220, "prompt truncated");

const packed = packKaraokeDisplayCues(aligned.cues);
assert(packed.length === Math.ceil(5 / LYRIC_OVERLAY_PREFERRED_LINES), `packed groups got ${packed.length}`);
assert(
  packed.every((c) => c.text.split("\n").length <= LYRIC_OVERLAY_MAX_LINES),
  "packed cues ≤ max lines",
);
assert(
  packed[0]!.text.split("\n").every((line, i) => line === chunks[i]),
  "packed keeps exact Biblioteca lines",
);
assert(packed[0]!.startMs === aligned.cues[0]!.startMs, "pack start = first line onset");
assert(
  cueAtPositionWithLeadMs(
    packed,
    11_000,
    CANCIONERO_DEFAULT_LEAD_MS,
    exactOpts,
  )?.text.split("\n")[0] === "Señor mi Dios",
  "−1s lead on packed karaoke cue",
);

// ASR typos must not rewrite cancionero text.
const messy: SubtitleCue[] = [
  { id: "m0", startMs: 12_000, endMs: 14_000, text: "senor mi dios" },
  { id: "m1", startMs: 14_500, endMs: 17_000, text: "al contemplar los cielos" },
  { id: "m2", startMs: 17_500, endMs: 19_000, text: "te adoro a ti" },
  { id: "m3", startMs: 22_000, endMs: 24_000, text: "cuan grande es el" },
  { id: "m4", startMs: 24_500, endMs: 26_500, text: "cuan grande es el" },
];
const messyAligned = alignCancioneroLinesToTiming(chunks, messy, 60_000);
assert(
  messyAligned.cues.every((c, i) => c.text === chunks[i]),
  "messy ASR never rewrites Biblioteca text",
);
assert(messyAligned.synced, "messy ASR still syncs via normalize");

console.log("SMOKE_OK cancionero-match");
