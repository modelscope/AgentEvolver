/**
 * Smoke: transformers cacheDir/localModelPath + Whisper error classification.
 * Does not download the model (fast). Full ASR: node scripts/verify-subtitle-asr.mjs from promedia/.
 */
import path from "node:path";
import {
  applyTransformersCacheEnv,
  clearCorruptModelCache,
  ensureSubtitleDirs,
  promediaRoot,
  subtitleModelsRoot,
} from "../src/subtitle-paths.ts";
import { classifyWhisperLoadError } from "../src/subtitle-asr.ts";

async function main(): Promise<void> {
  const root = promediaRoot();
  const dirs = await ensureSubtitleDirs();
  console.log("promediaRoot:", root);
  console.log("models:", dirs.models);
  if (dirs.models !== subtitleModelsRoot()) {
    throw new Error("models mismatch");
  }
  if (!dirs.models.includes(`${path.sep}data${path.sep}models`)) {
    throw new Error(`unexpected models root: ${dirs.models}`);
  }

  const fakeEnv = {
    cacheDir: null as string | null,
    localModelPath: "/wrong/",
    allowRemoteModels: false,
    allowLocalModels: false,
    useFSCache: false,
    useFS: false,
  };
  applyTransformersCacheEnv(fakeEnv, dirs.models);
  console.log("cacheDir:", fakeEnv.cacheDir);
  console.log("localModelPath:", fakeEnv.localModelPath);
  if (fakeEnv.cacheDir !== path.resolve(dirs.models)) {
    throw new Error("cacheDir not set to models root");
  }
  if (!String(fakeEnv.localModelPath).startsWith(path.resolve(dirs.models))) {
    throw new Error("localModelPath not under models root");
  }
  if (!fakeEnv.useFSCache || !fakeEnv.allowRemoteModels || !fakeEnv.useFS) {
    throw new Error("transformers flags not enabled");
  }

  // Confirm real package env accepts the same apply (no pipeline load).
  const transformers = await import("@huggingface/transformers");
  applyTransformersCacheEnv(transformers.env, dirs.models);
  if (transformers.env.cacheDir !== path.resolve(dirs.models)) {
    throw new Error("real env.cacheDir not applied");
  }

  const c1 = classifyWhisperLoadError(
    new Error("Unable to get model file path or buffer."),
  );
  const c2 = classifyWhisperLoadError(new Error("fetch failed"));
  const c3 = classifyWhisperLoadError(
    new Error("getaddrinfo ENOTFOUND huggingface.co"),
  );
  console.log({ c1, c2, c3 });
  if (c1 !== "model_path_error") throw new Error(`c1=${c1}`);
  if (c2 !== "model_download_failed") throw new Error(`c2=${c2}`);
  if (c3 !== "model_download_failed") throw new Error(`c3=${c3}`);

  const clearedMissing = await clearCorruptModelCache(
    dirs.models,
    "Xenova/whisper-tiny-definitely-missing",
  );
  console.log("clear missing:", clearedMissing);
  if (clearedMissing.removedModelDir) {
    throw new Error("clearCorrupt must not wipe a non-existent model dir");
  }
  const cleared = await clearCorruptModelCache(
    dirs.models,
    "Xenova/whisper-tiny",
  );
  console.log("clear:", cleared);

  // cwd-independence (same contract as youtube smoke)
  process.chdir(path.join(root, "apps", "hub"));
  const dirs2 = await ensureSubtitleDirs();
  if (dirs2.models !== dirs.models) {
    throw new Error("models path drifted after chdir");
  }

  console.log("SMOKE_OK");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
