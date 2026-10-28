/**
 * Minimal Whisper pipeline load into data/models (needs network once).
 * Usage: pnpm exec tsx scripts/smoke-whisper-load.mts  (from apps/hub)
 */
import path from "node:path";
import { readdir } from "node:fs/promises";
import {
  applyTransformersCacheEnv,
  ensureSubtitleDirs,
  subtitleModelsRoot,
} from "../src/subtitle-paths.ts";

async function main(): Promise<void> {
  const dirs = await ensureSubtitleDirs();
  const modelId =
    process.env.SUBTITLE_WHISPER_MODEL?.trim() || "Xenova/whisper-tiny";
  const revision =
    process.env.SUBTITLE_WHISPER_REVISION?.trim() || "main";
  console.log("[load] models root:", dirs.models);

  const transformers = await import("@huggingface/transformers");
  applyTransformersCacheEnv(transformers.env, dirs.models);
  console.log("[load] cacheDir:", transformers.env.cacheDir);
  console.log("[load] localModelPath:", transformers.env.localModelPath);

  const t0 = Date.now();
  const pipe = await transformers.pipeline(
    "automatic-speech-recognition",
    modelId,
    {
      revision,
      dtype: "q8",
    },
  );
  console.log("[load] pipeline ready in", Date.now() - t0, "ms");

  const audio = new Float32Array(16000); // 1s silence @ 16k
  const result = await pipe(audio, {
    return_timestamps: true,
    chunk_length_s: 30,
    sampling_rate: 16000,
  });
  const text = String((result as { text?: string })?.text ?? "");
  console.log("[load] text sample:", text.slice(0, 80) || "(empty/no_speech)");

  const modelDir = path.join(subtitleModelsRoot(), ...modelId.split("/"));
  let listing: string[] = [];
  try {
    listing = await readdir(path.join(modelDir, "onnx"));
    console.log("[load] onnx entries:", listing.slice(0, 12));
  } catch {
    listing = await readdir(dirs.models);
    console.log("[load] models root entries:", listing.slice(0, 12));
  }
  console.log("SMOKE_LOAD_OK");
}

main().catch((err) => {
  console.error("[load] FAILED:", err instanceof Error ? err.message : err);
  process.exit(1);
});
