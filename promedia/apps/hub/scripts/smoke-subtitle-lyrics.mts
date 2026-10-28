/**
 * Smoke: non-speech cue filter + YouTube caption name/rank helpers.
 * Usage: pnpm --filter @promedia/hub exec tsx scripts/smoke-subtitle-lyrics.mts
 */
import { mkdir, writeFile, rm } from "node:fs/promises";
import path from "node:path";
import {
  cleanCueText,
  filterSpeechCues,
  isNonSpeechCueText,
  speechCueRatio,
} from "../src/subtitle-cue-text.ts";
import { ensureSubtitleDirs } from "../src/subtitle-paths.ts";
import {
  cuesLikelyInLanguage,
  cuesLookLikeLyrics,
  inferDominantLanguage,
} from "../src/subtitle-lang-detect.ts";
import {
  mergeMicroLyricCues,
  normalizeLyricPunctuation,
  preserveNativeCaptionCues,
  shapeLyricLikeCues,
  splitLongLyricText,
} from "../src/subtitle-lyric-shape.ts";
import {
  cachedCaptionSatisfiesPick,
  expandCaptionPreferLangs,
  findYoutubeCaptionFiles,
  pickBestYoutubeCaptionTrack,
  rankYoutubeCaptionLang,
  youtubeCaptionKindFromLangTag,
  youtubeCaptionLangFromName,
} from "../src/subtitle-youtube.ts";
import {
  parseWhisperCppVttOrSrt,
  whisperCppDefaultMinBytes,
  whisperCppDefaultModelUrl,
} from "../src/subtitle-whisper-cpp.ts";
import {
  cueAtPositionWithLeadMs,
  type SubtitleCue,
} from "@promedia/shared";

function assert(cond: unknown, msg: string): asserts cond {
  if (!cond) throw new Error(msg);
}

async function main(): Promise<void> {
  assert(isNonSpeechCueText("[MUSIC]"), "MUSIC label");
  assert(isNonSpeechCueText("[MÚSICA]"), "MÚSICA label");
  assert(isNonSpeechCueText("(music)"), "paren music");
  assert(isNonSpeechCueText("♪♪"), "glyph only");
  assert(isNonSpeechCueText("  [ Blank_Audio ]  "), "blank audio");
  assert(!isNonSpeechCueText("The club isn't the best place to find a lover"), "lyric");
  assert(
    !isNonSpeechCueText("♪ The club isn't the best place ♪"),
    "lyric with music glyphs",
  );

  const filtered = filterSpeechCues([
    { id: "a", startMs: 0, endMs: 1000, text: "[MUSIC]" },
    { id: "b", startMs: 1000, endMs: 2000, text: "  Hello world  " },
    { id: "c", startMs: 2000, endMs: 3000, text: "[MÚSICA]" },
    { id: "d", startMs: 3000, endMs: 4000, text: "Shape of you" },
  ]);
  assert(filtered.length === 2, `expected 2 speech cues, got ${filtered.length}`);
  assert(filtered[0]!.text === "Hello world", "clean first lyric");
  assert(filtered[1]!.text === "Shape of you", "second lyric");
  assert(filtered[0]!.id === "c0" && filtered[1]!.id === "c1", "re-id");

  assert(speechCueRatio([{ id: "x", startMs: 0, endMs: 1, text: "[MUSIC]" }]) === 0, "ratio 0");
  assert(
    speechCueRatio([
      { id: "x", startMs: 0, endMs: 1, text: "[MUSIC]" },
      { id: "y", startMs: 1, endMs: 2, text: "hi" },
    ]) === 0.5,
    "ratio 0.5",
  );

  const cleaned = cleanCueText("<c>Hello</c> &amp; <b>you</b>");
  assert(cleaned === "Hello & you", `vtt cleanup: ${cleaned}`);

  const vtt = `WEBVTT

00:00:00.000 --> 00:00:02.000
[MUSIC]

00:00:02.000 --> 00:00:04.000
The club isn't the best place to find a lover

00:00:04.000 --> 00:00:06.000
[MÚSICA]
`;
  const parsed = filterSpeechCues(parseWhisperCppVttOrSrt(vtt));
  assert(parsed.length === 1, `vtt filter expected 1, got ${parsed.length}`);
  assert(
    parsed[0]!.text.includes("club isn't the best"),
    `unexpected lyric: ${parsed[0]?.text}`,
  );

  // YouTube-style short clocks + cue settings must not yield zero cues.
  const shortForm = filterSpeechCues(
    parseWhisperCppVttOrSrt(`WEBVTT

00:00.080 --> 00:02.320 align:start position:0%
The club isn't the best place to find a lover
`),
  );
  assert(shortForm.length === 1, "short MM:SS.mmm VTT");
  assert(shortForm[0]!.startMs === 80, `short start ${shortForm[0]!.startMs}`);

  assert(
    youtubeCaptionLangFromName("JGw1FRCpL94", "JGw1FRCpL94.en.vtt") === "en",
    "en.vtt lang",
  );
  assert(
    youtubeCaptionLangFromName("JGw1FRCpL94", "JGw1FRCpL94.es-ES.vtt") ===
      "es-es",
    "es-ES lang",
  );
  assert(
    youtubeCaptionLangFromName("JGw1FRCpL94", "JGw1FRCpL94.wav") === null,
    "wav not caption",
  );
  assert(
    rankYoutubeCaptionLang("es", ["es", "en"]) <
      rankYoutubeCaptionLang("en", ["es", "en"]),
    "prefer es over en when target es",
  );
  assert(
    rankYoutubeCaptionLang("es", ["es", "en"], "auto") <
      rankYoutubeCaptionLang("en", ["es", "en"], "manual"),
    "matching ES (even auto) beats manual EN when UI is ES",
  );
  assert(
    rankYoutubeCaptionLang("es", ["es", "en"], "manual") <
      rankYoutubeCaptionLang("es", ["es", "en"], "auto"),
    "manual ES beats auto ES",
  );
  assert(
    youtubeCaptionKindFromLangTag("en-orig") === "auto",
    "en-orig is auto",
  );
  const picked = pickBestYoutubeCaptionTrack(
    [
      { lang: "en", kind: "auto" },
      { lang: "es", kind: "auto" },
      { lang: "en", kind: "manual" },
      { lang: "es", kind: "manual" },
    ],
    ["es", "en"],
  );
  assert(
    picked?.lang === "es" && picked.kind === "manual",
    `expected manual es, got ${picked?.kind}:${picked?.lang}`,
  );

  // Stale auto/unknown must not block a manual pick (same videoId.es.vtt name).
  assert(
    !cachedCaptionSatisfiesPick(
      { lang: "es", trackKind: "unknown" },
      { lang: "es", kind: "manual" },
    ),
    "unknown cache must not satisfy manual pick",
  );
  assert(
    !cachedCaptionSatisfiesPick(
      { lang: "es", trackKind: "auto" },
      { lang: "es", kind: "manual" },
    ),
    "auto cache must not satisfy manual pick",
  );
  assert(
    cachedCaptionSatisfiesPick(
      { lang: "es", trackKind: "manual" },
      { lang: "es", kind: "manual" },
    ),
    "manual cache satisfies manual pick",
  );

  const expanded = expandCaptionPreferLangs(["es"]);
  assert(expanded[0] === "es" || expanded.includes("es"), "expand keeps es");
  assert(expanded.includes("es-419"), "expand Spanish regional");
  // Native agent must NOT auto-append English (that fed Marian on missing ES).
  assert(!expanded.includes("en"), "expand es must not force en fallback");

  // Gold-standard hymn lines (Cuán grande es Él density) stay short + timed.
  const hymn: SubtitleCue[] = [
    { id: "a", startMs: 12_000, endMs: 15_000, text: "Señor mi Dios" },
    {
      id: "b",
      startMs: 15_000,
      endMs: 18_500,
      text: "Al contemplar los cielos",
    },
    {
      id: "c",
      startMs: 18_500,
      endMs: 22_000,
      text: "El firmamento y las estrellas mil",
    },
    {
      id: "d",
      startMs: 22_000,
      endMs: 26_000,
      text: "Al oír tu voz en los potentes truenos",
    },
    {
      id: "e",
      startMs: 26_000,
      endMs: 30_000,
      text: "Y ver brillar el sol en su cenit",
    },
    {
      id: "f",
      startMs: 30_000,
      endMs: 34_000,
      text: "Mi corazón entona la canción",
    },
    {
      id: "g",
      startMs: 34_000,
      endMs: 38_000,
      text: "Cuán grande es Él, cuán grande es Él",
    },
    {
      id: "h",
      startMs: 38_000,
      endMs: 42_000,
      text: "Mi corazón entona la canción",
    },
  ];
  // Native path: clocks + wording unchanged (no tighten / capitalize).
  const nativeHymn = preserveNativeCaptionCues(hymn);
  assert(nativeHymn.length === hymn.length, "native keeps cue count");
  assert(
    nativeHymn[0]!.startMs === 12_000 && nativeHymn[0]!.text === "Señor mi Dios",
    "native preserves first cue clock+text",
  );
  assert(
    cueAtPositionWithLeadMs(nativeHymn, 0, 0, { holdIntro: false }) === null,
    "exact timing: blank before first official startMs",
  );
  assert(
    cueAtPositionWithLeadMs(nativeHymn, 12_500, 0, { holdIntro: false })?.text ===
      "Señor mi Dios",
    "exact timing: first lyric at official window",
  );

  const shapedHymn = shapeLyricLikeCues(hymn);
  assert(shapedHymn.length >= 6, `hymn cues kept, got ${shapedHymn.length}`);
  assert(
    shapedHymn[0]!.text.toLowerCase().includes("señor"),
    `hymn wording: ${shapedHymn[0]?.text}`,
  );
  // Assisted ASR path still uses intro-hold.
  assert(
    cueAtPositionWithLeadMs(shapedHymn, 0, 0, { holdIntro: true })?.text
      .toLowerCase()
      .includes("señor"),
    "assisted intro-hold shows first hymn line at t=0",
  );
  assert(
    cueAtPositionWithLeadMs(shapedHymn, 0, -2000, { holdIntro: true })?.text
      .toLowerCase()
      .includes("señor"),
    "Adelante −2 still works with shaped cues",
  );

  assert(
    normalizeLyricPunctuation("señor mi dios") === "Señor mi dios",
    "capitalize lyric line",
  );
  const split = splitLongLyricText(
    "Señor mi Dios al contemplar los cielos el firmamento y las estrellas mil al oír tu voz en los potentes truenos y ver brillar el sol en su cenit",
    0,
    20_000,
  );
  assert(split.length >= 2, `long lyric split, got ${split.length}`);
  assert(
    split.every((s) => s.text.length <= 70),
    "split pieces stay line-sized",
  );

  const micro = mergeMicroLyricCues([
    { id: "1", startMs: 0, endMs: 400, text: "Cuán" },
    { id: "2", startMs: 400, endMs: 800, text: "grande" },
    { id: "3", startMs: 800, endMs: 1200, text: "es" },
    { id: "4", startMs: 1200, endMs: 1800, text: "Él" },
    { id: "5", startMs: 2000, endMs: 2400, text: "cuán" },
    { id: "6", startMs: 2400, endMs: 2800, text: "grande" },
    { id: "7", startMs: 2800, endMs: 3200, text: "es" },
    { id: "8", startMs: 3200, endMs: 3800, text: "Él" },
  ]);
  assert(micro.length <= 4, `micro merge → lines, got ${micro.length}`);
  assert(
    micro.some((c) => /grande/i.test(c.text)),
    "merged micro keeps wording",
  );

  // Already-Spanish must not look like English for MT skip.
  const esCues = [
    {
      id: "1",
      startMs: 0,
      endMs: 1000,
      text: "El Señor es mi pastor y nada me faltará en la vida",
    },
    {
      id: "2",
      startMs: 1000,
      endMs: 2000,
      text: "Porque Él está conmigo para siempre en el amor",
    },
    {
      id: "3",
      startMs: 2000,
      endMs: 3000,
      text: "Gracias Dios por tu gracia y por el corazón nuevo",
    },
  ];
  assert(inferDominantLanguage(esCues) === "es", "infer Spanish");
  assert(cuesLikelyInLanguage(esCues, "es"), "likely Spanish");
  assert(!cuesLikelyInLanguage(esCues, "en"), "not English");

  const lyricCues = Array.from({ length: 8 }, (_, i) => ({
    id: `l${i}`,
    startMs: i * 1000,
    endMs: i * 1000 + 900,
    text: "Shape of you love",
  }));
  assert(cuesLookLikeLyrics(lyricCues), "short dense lines = lyrics");

  assert(
    whisperCppDefaultModelUrl("ggml-base.bin").endsWith("/ggml-base.bin"),
    "default URL base",
  );
  assert(
    whisperCppDefaultMinBytes("ggml-base.bin") >= 140 * 1024 * 1024,
    "base min bytes",
  );
  assert(
    whisperCppDefaultMinBytes("ggml-tiny.bin") <
      whisperCppDefaultMinBytes("ggml-base.bin"),
    "tiny min < base min",
  );

  const dirs = await ensureSubtitleDirs();
  const id = "smokeCap01";
  const enPath = path.join(dirs.youtubeCache, `${id}.en.vtt`);
  const esPath = path.join(dirs.youtubeCache, `${id}.es.vtt`);
  await mkdir(dirs.youtubeCache, { recursive: true });
  await writeFile(
    enPath,
    `WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nEnglish line\n`,
  );
  await writeFile(
    esPath,
    `WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nLínea en español\n`,
  );
  const found = await findYoutubeCaptionFiles(dirs.youtubeCache, id, [
    "es",
    "en",
  ]);
  assert(found.length >= 2, "expected both caption files");
  assert(found[0]!.lang === "es", `prefer es first, got ${found[0]?.lang}`);
  await rm(enPath, { force: true });
  await rm(esPath, { force: true });

  console.log("SMOKE_OK");
}

main().catch((err) => {
  console.error("[smoke-subtitle-lyrics] FAILED:", err);
  process.exit(1);
});
