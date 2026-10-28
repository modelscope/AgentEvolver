import type { SubtitleCue, SubtitleTargetLang } from "@promedia/shared";
import { filterSpeechCues, isNonSpeechCueText } from "./subtitle-cue-text.js";
import {
  cuesLikelyInLanguage,
  cuesLookLikeLyrics,
  inferDominantLanguage,
} from "./subtitle-lang-detect.js";
import {
  applyTransformersCacheEnv,
  ensureSubtitleDirs,
} from "./subtitle-paths.js";

/**
 * Tiny local MT via Marian/OPUS ONNX (Xenova) when a pair model exists.
 * Phase 1 staged: if no model for the pair, return source cues unchanged.
 *
 * Supported pairs (downloaded on first use into data/models/):
 *   es↔en, en↔es, pt↔en, en↔pt, fr↔en, en↔fr, zh↔en, en↔zh
 * Cascades via English when needed (e.g. es→pt = es→en→pt).
 */

type TranslatePipeline = (
  text: string | string[],
  options?: Record<string, unknown>,
) => Promise<Array<{ translation_text?: string }> | { translation_text?: string }>;

const pipeCache = new Map<string, Promise<TranslatePipeline | null>>();

const DIRECT_MODELS: Record<string, string> = {
  "es-en": "Xenova/opus-mt-es-en",
  "en-es": "Xenova/opus-mt-en-es",
  "pt-en": "Xenova/opus-mt-ROMANCE-en",
  "fr-en": "Xenova/opus-mt-ROMANCE-en",
  "en-pt": "Xenova/opus-mt-en-ROMANCE",
  "en-fr": "Xenova/opus-mt-en-ROMANCE",
  "zh-en": "Xenova/opus-mt-zh-en",
  "en-zh": "Xenova/opus-mt-en-zh",
};

function langKey(lang: string): string {
  const l = lang.toLowerCase();
  if (l.startsWith("zh")) return "zh";
  if (l.startsWith("pt")) return "pt";
  if (l.startsWith("es")) return "es";
  if (l.startsWith("fr")) return "fr";
  if (l.startsWith("en")) return "en";
  return l.slice(0, 2);
}

function targetKey(lang: SubtitleTargetLang): string {
  return langKey(lang);
}

async function loadPair(src: string, tgt: string): Promise<TranslatePipeline | null> {
  const key = `${src}-${tgt}`;
  const modelId = DIRECT_MODELS[key];
  if (!modelId) return null;

  let pending = pipeCache.get(key);
  if (!pending) {
    pending = (async () => {
      try {
        const dirs = await ensureSubtitleDirs();
        const transformers = await import("@huggingface/transformers");
        applyTransformersCacheEnv(transformers.env, dirs.models);
        const pipe = await transformers.pipeline("translation", modelId);
        return pipe as unknown as TranslatePipeline;
      } catch (err) {
        console.warn(
          `[subtitle] MT model unavailable ${key}:`,
          err instanceof Error ? err.message : err,
        );
        return null;
      }
    })();
    pipeCache.set(key, pending);
  }
  return pending;
}

async function translateTexts(
  pipe: TranslatePipeline,
  texts: string[],
): Promise<string[]> {
  const out: string[] = [];
  // Batch small groups to limit memory.
  const batchSize = 8;
  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize);
    const result = await pipe(batch);
    const rows = Array.isArray(result) ? result : [result];
    for (let j = 0; j < batch.length; j++) {
      const row = rows[j] as { translation_text?: string } | undefined;
      out.push((row?.translation_text ?? batch[j]!).trim() || batch[j]!);
    }
  }
  return out;
}

async function translateViaPair(
  texts: string[],
  src: string,
  tgt: string,
): Promise<string[] | null> {
  if (src === tgt) return texts;
  const direct = await loadPair(src, tgt);
  if (direct) {
    return translateTexts(direct, texts);
  }
  // Cascade through English when both legs exist.
  if (src !== "en" && tgt !== "en") {
    const toEn = await loadPair(src, "en");
    const fromEn = await loadPair("en", tgt);
    if (toEn && fromEn) {
      const mid = await translateTexts(toEn, texts);
      return translateTexts(fromEn, mid);
    }
  }
  return null;
}

export type TranslateResult = {
  cues: SubtitleCue[];
  translated: boolean;
  /** Human note when MT skipped. */
  note: string | null;
};

/**
 * Translate ASR cues to targetLang with local Marian models.
 * Never calls cloud APIs. On failure, returns source cues (stage A).
 *
 * Guards against bad text:
 * - Do not re-translate cues that already read as the target language
 *   (Spanish captions / Whisper mis-tagged as `en`).
 * - Optionally keep YouTube lyric lines in the source language — Marian
 *   often mangles sung poetry worse than showing accurate EN/ES originals.
 */
export async function translateCuesLocal(options: {
  cues: SubtitleCue[];
  sourceLang: string | null;
  targetLang: SubtitleTargetLang;
  /** Already English from Whisper translate task. */
  alreadyEnglish?: boolean;
  /**
   * When true (YouTube caption path), skip Marian for lyric-shaped cues so
   * we do not mangle accurate uploaded/auto lyrics into bad MT.
   */
  preserveLyricsSource?: boolean;
  onStatus?: () => void;
}): Promise<TranslateResult> {
  const {
    cues,
    targetLang,
    alreadyEnglish,
    preserveLyricsSource,
    onStatus,
  } = options;
  if (!cues.length) {
    return { cues, translated: false, note: null };
  }

  const tgt = targetKey(targetLang);
  let src = options.sourceLang ? langKey(options.sourceLang) : null;
  if (alreadyEnglish) {
    src = "en";
  }

  // Trust cue text over a wrong Whisper/YT lang tag: already-Spanish must not
  // go through en→es Marian (that is a common "texto equivocado" failure).
  const speechPreview = filterSpeechCues(cues);
  if (speechPreview.length && cuesLikelyInLanguage(speechPreview, tgt)) {
    return { cues: speechPreview, translated: false, note: null };
  }
  const inferred = speechPreview.length
    ? inferDominantLanguage(speechPreview)
    : null;
  if (inferred && (!src || src !== inferred)) {
    // Prefer LID over a conflicting metadata tag when signal is clear.
    if (!src || src !== tgt) {
      src = inferred;
    }
  }

  if (src && src === tgt) {
    return {
      cues: speechPreview.length ? speechPreview : cues,
      translated: alreadyEnglish === true && tgt === "en",
      note: null,
    };
  }

  // Unknown source: if Whisper already produced English, translate en→target.
  // Otherwise keep source cues visible (do NOT assume Spanish — that made
  // src===tgt for Spanish UI and skipped en→es, leaving wrong-language text;
  // with empty/garbage EN-forced ASR it looked like "Spanish never appears").
  if (!src) {
    if (alreadyEnglish) {
      src = "en";
    } else {
      return {
        cues: speechPreview.length ? speechPreview : cues,
        translated: false,
        note: tgt === "es" ? null : `mt_unavailable:unknown->${tgt}`,
      };
    }
  }

  if (src === tgt) {
    return {
      cues: speechPreview.length ? speechPreview : cues,
      translated: false,
      note: null,
    };
  }

  // Skip MT for sound labels — Marian turns `[MUSIC]` into `[MÚSICA]` which
  // looks like a Spanish lyric line but is not. Prefer dropping them.
  const speechCues = speechPreview;
  if (!speechCues.length) {
    return { cues: [], translated: false, note: null };
  }

  // YouTube lyrics: accurate source language beats mangled Marian poetry.
  if (
    preserveLyricsSource &&
    cuesLookLikeLyrics(speechCues) &&
    src !== tgt
  ) {
    console.warn(
      `[subtitle] skipping Marian for lyric-shaped cues (${src}->${tgt}) — keeping source text`,
    );
    return {
      cues: speechCues,
      translated: false,
      note: "mt_lyrics_preserved",
    };
  }

  onStatus?.();
  const texts = speechCues.map((c) => c.text);
  try {
    const translated = await translateViaPair(texts, src, tgt);
    if (!translated) {
      return {
        cues: speechCues,
        translated: false,
        note: `mt_unavailable:${src}->${tgt}`,
      };
    }
    return {
      cues: speechCues.map((c, i) => {
        const next = (translated[i] ?? c.text).trim();
        // If MT somehow reintroduced a music bracket label, keep source words.
        const text =
          next && !isNonSpeechCueText(next) ? next : c.text;
        return { ...c, text };
      }),
      translated: true,
      note: null,
    };
  } catch (err) {
    console.warn(
      "[subtitle] local MT failed:",
      err instanceof Error ? err.message : err,
    );
    return {
      cues: speechCues,
      translated: false,
      note: "mt_failed",
    };
  }
}
