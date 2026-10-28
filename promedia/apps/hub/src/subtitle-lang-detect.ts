import type { SubtitleCue } from "@promedia/shared";
import { cleanCueText, isNonSpeechCueText } from "./subtitle-cue-text.js";

/**
 * Lightweight cue-language heuristics so we do not Marian-mangle text that is
 * already in the target language (e.g. Spanish captions mis-tagged as `en`,
 * or Whisper `sourceLang` wrong after `-l auto`).
 *
 * Not a full LID model — enough stopword + diacritic signal for church/song
 * subtitle volumes on CPU without extra deps.
 */

const WORD_RE = /[\p{L}\p{N}']+/gu;

const ES_STOP = new Set([
  "el",
  "la",
  "los",
  "las",
  "de",
  "que",
  "y",
  "en",
  "un",
  "una",
  "es",
  "por",
  "para",
  "con",
  "no",
  "se",
  "del",
  "al",
  "lo",
  "como",
  "más",
  "pero",
  "sus",
  "le",
  "ya",
  "o",
  "este",
  "sí",
  "porque",
  "esta",
  "entre",
  "cuando",
  "muy",
  "sin",
  "sobre",
  "también",
  "me",
  "hasta",
  "hay",
  "donde",
  "quien",
  "desde",
  "todo",
  "nos",
  "durante",
  "uno",
  "ni",
  "contra",
  "otros",
  "ese",
  "eso",
  "ante",
  "ellos",
  "e",
  "esto",
  "mí",
  "antes",
  "algunos",
  "qué",
  "unos",
  "yo",
  "otro",
  "otras",
  "otra",
  "él",
  "tanto",
  "esa",
  "estos",
  "mucho",
  "quienes",
  "nada",
  "muchos",
  "cual",
  "poco",
  "ella",
  "estar",
  "estas",
  "algunas",
  "algo",
  "nosotros",
  "mi",
  "mis",
  "tú",
  "te",
  "ti",
  "tu",
  "tus",
  "ellas",
  "dios",
  "señor",
  "jesús",
  "amor",
  "vida",
  "corazón",
  "gracias",
  "amén",
]);

const EN_STOP = new Set([
  "the",
  "and",
  "you",
  "that",
  "was",
  "for",
  "are",
  "with",
  "his",
  "they",
  "this",
  "have",
  "from",
  "one",
  "had",
  "word",
  "but",
  "not",
  "what",
  "all",
  "were",
  "when",
  "your",
  "can",
  "said",
  "there",
  "each",
  "which",
  "she",
  "do",
  "how",
  "their",
  "if",
  "will",
  "up",
  "other",
  "about",
  "out",
  "many",
  "then",
  "them",
  "these",
  "so",
  "some",
  "her",
  "would",
  "make",
  "like",
  "him",
  "into",
  "time",
  "has",
  "look",
  "two",
  "more",
  "write",
  "go",
  "see",
  "number",
  "no",
  "way",
  "could",
  "people",
  "my",
  "than",
  "first",
  "water",
  "been",
  "call",
  "who",
  "oil",
  "its",
  "now",
  "find",
  "long",
  "down",
  "day",
  "did",
  "get",
  "come",
  "made",
  "may",
  "part",
  "lord",
  "god",
  "jesus",
  "love",
  "heart",
  "praise",
  "amen",
]);

const PT_STOP = new Set([
  "de",
  "a",
  "o",
  "que",
  "e",
  "do",
  "da",
  "em",
  "um",
  "para",
  "é",
  "com",
  "não",
  "uma",
  "os",
  "no",
  "se",
  "na",
  "por",
  "mais",
  "as",
  "dos",
  "como",
  "mas",
  "foi",
  "ao",
  "ele",
  "das",
  "tem",
  "à",
  "seu",
  "sua",
  "ou",
  "ser",
  "quando",
  "muito",
  "há",
  "nos",
  "já",
  "está",
  "eu",
  "também",
  "só",
  "pelo",
  "pela",
  "até",
  "isso",
  "ela",
  "entre",
  "era",
  "depois",
  "sem",
  "mesmo",
  "aos",
  "ter",
  "seus",
  "quem",
  "nas",
  "me",
  "esse",
  "eles",
  "estão",
  "você",
  "tinha",
  "foram",
  "essa",
  "num",
  "nem",
  "suas",
  "meu",
  "às",
  "minha",
  "têm",
  "numa",
  "pelos",
  "elas",
  "havia",
  "seja",
  "qual",
  "será",
  "nós",
  "tenho",
  "lhe",
  "deles",
  "essas",
  "esses",
  "pelas",
  "este",
  "fosse",
  "dele",
  "tu",
  "te",
  "vocês",
  "vos",
  "lhes",
  "meus",
  "minhas",
  "teu",
  "tua",
  "teus",
  "tuas",
  "nosso",
  "nossa",
  "nossos",
  "nossas",
  "dela",
  "delas",
  "esta",
  "estes",
  "estas",
  "aquele",
  "aquela",
  "aqueles",
  "aquelas",
  "isto",
  "aquilo",
  "deus",
  "senhor",
  "jesus",
  "amor",
  "vida",
  "coração",
  "graças",
  "amém",
]);

const FR_STOP = new Set([
  "le",
  "de",
  "un",
  "être",
  "et",
  "à",
  "il",
  "avoir",
  "ne",
  "je",
  "son",
  "que",
  "se",
  "qui",
  "ce",
  "dans",
  "en",
  "du",
  "elle",
  "au",
  "pour",
  "pas",
  "que",
  "vous",
  "par",
  "sur",
  "faire",
  "plus",
  "dire",
  "me",
  "on",
  "mon",
  "lui",
  "nous",
  "comme",
  "mais",
  "pouvoir",
  "avec",
  "tout",
  "y",
  "aller",
  "voir",
  "savoir",
  "en",
  "quoi",
  "leur",
  "si",
  "deux",
  "sans",
  "où",
  "ou",
  "très",
  "dieu",
  "seigneur",
  "jésus",
  "amour",
  "vie",
  "cœur",
  "merci",
  "amen",
]);

function wordsOf(text: string): string[] {
  const out: string[] = [];
  const m = text.toLowerCase().match(WORD_RE);
  if (!m) return out;
  for (const w of m) {
    if (w.length >= 2) out.push(w);
  }
  return out;
}

function scoreLang(
  words: string[],
  stop: Set<string>,
  diacriticBonus: number,
): number {
  let hits = 0;
  for (const w of words) {
    if (stop.has(w)) hits += 1;
  }
  return hits + diacriticBonus;
}

/**
 * Infer dominant language among es/en/pt/fr from cue texts.
 * Returns null when signal is weak or mixed.
 */
export function inferDominantLanguage(
  cues: Array<Pick<SubtitleCue, "text">>,
): string | null {
  const texts = cues
    .map((c) => cleanCueText(c.text))
    .filter((t) => t && !isNonSpeechCueText(t));
  if (!texts.length) return null;

  const joined = texts.join(" ");
  const words = wordsOf(joined);
  if (words.length < 4) return null;

  const diacEs = (joined.match(/[áéíóúüñ¿¡]/gi) || []).length;
  const diacPt = (joined.match(/[ãõçáéíóúâêô]/gi) || []).length;
  const diacFr = (joined.match(/[àâæçéèêëïîôœùûüÿ]/gi) || []).length;

  const scores: Record<string, number> = {
    es: scoreLang(words, ES_STOP, diacEs * 0.75),
    en: scoreLang(words, EN_STOP, 0),
    pt: scoreLang(words, PT_STOP, diacPt * 0.5),
    fr: scoreLang(words, FR_STOP, diacFr * 0.5),
  };

  const ranked = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  const best = ranked[0]!;
  const second = ranked[1]!;
  // Need a clear winner vs runner-up and enough absolute hits.
  if (best[1] < 3) return null;
  if (best[1] < second[1] * 1.25 && best[1] - second[1] < 2) return null;
  return best[0];
}

/**
 * True when cues already read as `lang` (skip Marian re-translation).
 */
export function cuesLikelyInLanguage(
  cues: Array<Pick<SubtitleCue, "text">>,
  lang: string,
): boolean {
  const inferred = inferDominantLanguage(cues);
  if (!inferred) return false;
  const want = lang.toLowerCase().slice(0, 2);
  return inferred === want;
}

/**
 * Short, dense timed lines → likely song lyrics (Marian often mangles these).
 */
export function cuesLookLikeLyrics(
  cues: Array<Pick<SubtitleCue, "text" | "startMs" | "endMs">>,
): boolean {
  const speech = cues.filter(
    (c) => cleanCueText(c.text) && !isNonSpeechCueText(c.text),
  );
  if (speech.length < 6) return false;
  const lengths = speech.map((c) => cleanCueText(c.text).length);
  lengths.sort((a, b) => a - b);
  const mid = lengths[Math.floor(lengths.length / 2)] ?? 0;
  const avg =
    lengths.reduce((s, n) => s + n, 0) / Math.max(1, lengths.length);
  // Lyric lines are short; spoken sermons/talks tend longer.
  return mid <= 55 && avg <= 70;
}
