/**
 * Smoke: ggml download coalesce + incomplete rejection + language `-l` helper.
 * Usage: pnpm --filter @promedia/hub exec tsx scripts/smoke-ggml-download.mts
 *
 * Network: uses a tiny fixture URL with SUBTITLE_WHISPER_CPP_MIN_BYTES=8 so we
 * do not pull the real ~75 MB ggml in CI.
 */
import { createServer } from "node:http";
import { mkdir, readFile, stat, writeFile, rm } from "node:fs/promises";
import path from "node:path";
import {
  ensureWhisperCppModel,
  fileLooksComplete,
  parseWhisperCppResult,
  removeIncompleteWhisperCppModel,
  resetWhisperCppDownloadInflight,
  whisperCliLanguageArg,
  whisperCppDownloadInflightCount,
  whisperCppModelPath,
} from "../src/subtitle-whisper-cpp.ts";
import { ensureSubtitleDirs } from "../src/subtitle-paths.ts";

async function withTempHttpFixture(
  body: Buffer,
): Promise<{ url: string; close: () => Promise<void> }> {
  const server = createServer((_req, res) => {
    res.writeHead(200, {
      "Content-Type": "application/octet-stream",
      "Content-Length": String(body.length),
    });
    res.end(body);
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const addr = server.address();
  if (!addr || typeof addr === "string") {
    throw new Error("no server port");
  }
  return {
    url: `http://127.0.0.1:${addr.port}/ggml-tiny.bin`,
    close: () =>
      new Promise((resolve, reject) =>
        server.close((err) => (err ? reject(err) : resolve())),
      ),
  };
}

async function main(): Promise<void> {
  // --- pure helpers ---
  if (whisperCliLanguageArg(null) !== "auto") {
    throw new Error("null hint must be auto (whisper-cli defaults to en!)");
  }
  if (whisperCliLanguageArg("es") !== "es") {
    throw new Error("es hint");
  }
  if (whisperCliLanguageArg("es-ES") !== "es") {
    throw new Error("es-ES → es");
  }
  if (whisperCliLanguageArg("zh-CN") !== "zh") {
    throw new Error("zh-CN → zh");
  }

  const parsed = parseWhisperCppResult(
    JSON.stringify({
      result: { language: "es" },
      transcription: [
        {
          text: " Hola mundo ",
          offsets: { from: 0, to: 1200 },
        },
      ],
    }),
  );
  if (parsed.language !== "es") {
    throw new Error(`expected language es, got ${parsed.language}`);
  }
  if (parsed.cues.length !== 1 || parsed.cues[0]!.text !== "Hola mundo") {
    throw new Error(`cues bad: ${JSON.stringify(parsed.cues)}`);
  }

  // --- incomplete rejection ---
  const dirs = await ensureSubtitleDirs();
  const smokeDir = path.join(dirs.models, "whisper-smoke-ggml");
  await mkdir(smokeDir, { recursive: true });
  const prevModel = process.env.SUBTITLE_WHISPER_CPP_MODEL;
  const prevUrl = process.env.SUBTITLE_WHISPER_CPP_MODEL_URL;
  const prevMin = process.env.SUBTITLE_WHISPER_CPP_MIN_BYTES;

  process.env.SUBTITLE_WHISPER_CPP_MODEL = "smoke-ggml-tiny.bin";
  process.env.SUBTITLE_WHISPER_CPP_MIN_BYTES = "32";

  // Point model path under models/whisper via default whisperCppModelPath —
  // it uses subtitleModelsRoot()/whisper/<name>. Write an undersized leftover.
  const modelPath = whisperCppModelPath(dirs.models);
  await mkdir(path.dirname(modelPath), { recursive: true });
  await writeFile(modelPath, Buffer.alloc(8));
  if (await fileLooksComplete(modelPath)) {
    throw new Error("8-byte file must not look complete with MIN=32");
  }
  const removed = await removeIncompleteWhisperCppModel(modelPath);
  if (!removed) {
    throw new Error("expected incomplete ggml to be removed");
  }

  // --- coalesced download ---
  const payload = Buffer.alloc(64, 7);
  const fixture = await withTempHttpFixture(payload);
  process.env.SUBTITLE_WHISPER_CPP_MODEL_URL = fixture.url;
  process.env.SUBTITLE_WHISPER_CPP_MIN_BYTES = "32";
  resetWhisperCppDownloadInflight();

  let downloadLogs = 0;
  const origLog = console.log;
  console.log = (...args: unknown[]) => {
    const line = args.map(String).join(" ");
    if (line.includes("whisper.cpp downloading ggml")) {
      downloadLogs += 1;
    }
    origLog.apply(console, args);
  };

  try {
    const results = await Promise.all([
      ensureWhisperCppModel(),
      ensureWhisperCppModel(),
      ensureWhisperCppModel(),
    ]);
    if (downloadLogs !== 1) {
      throw new Error(
        `expected exactly 1 download log, got ${downloadLogs} (coalesce failed)`,
      );
    }
    if (whisperCppDownloadInflightCount() !== 0) {
      throw new Error("inflight map should be clear after settle");
    }
    const out = results[0]!;
    if (!(await fileLooksComplete(out))) {
      throw new Error("model not complete after ensure");
    }
    const st = await stat(out);
    if (st.size !== payload.length) {
      throw new Error(`size ${st.size} != ${payload.length}`);
    }
    const roundTrip = await readFile(out);
    if (!roundTrip.equals(payload)) {
      throw new Error("model bytes mismatch");
    }
    // Second wave: must not re-download when complete.
    downloadLogs = 0;
    await Promise.all([ensureWhisperCppModel(), ensureWhisperCppModel()]);
    if (downloadLogs !== 0) {
      throw new Error("complete model must not re-download");
    }
  } finally {
    console.log = origLog;
    await fixture.close();
    await rm(modelPath, { force: true }).catch(() => undefined);
    await rm(`${modelPath}.lock`, { force: true }).catch(() => undefined);
    resetWhisperCppDownloadInflight();
    if (prevModel === undefined) delete process.env.SUBTITLE_WHISPER_CPP_MODEL;
    else process.env.SUBTITLE_WHISPER_CPP_MODEL = prevModel;
    if (prevUrl === undefined) delete process.env.SUBTITLE_WHISPER_CPP_MODEL_URL;
    else process.env.SUBTITLE_WHISPER_CPP_MODEL_URL = prevUrl;
    if (prevMin === undefined) delete process.env.SUBTITLE_WHISPER_CPP_MIN_BYTES;
    else process.env.SUBTITLE_WHISPER_CPP_MIN_BYTES = prevMin;
    await rm(smokeDir, { recursive: true, force: true }).catch(() => undefined);
  }

  console.log("SMOKE_OK (ggml download coalesce + language helpers)");
}

main().catch((err) => {
  console.error("[smoke-ggml-download] FAILED:", err);
  process.exit(1);
});
