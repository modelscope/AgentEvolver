import { access, rm, writeFile, stat } from "node:fs/promises";
import path from "node:path";
import {
  ensureSubtitleDirs,
  promediaRoot,
  subtitleDataRoot,
} from "../src/subtitle-paths.ts";
import {
  classifyYoutubeError,
  findCachedAudio,
  youtubeYtDlpOutputArgs,
} from "../src/subtitle-youtube.ts";

async function main(): Promise<void> {
  const root = promediaRoot();
  const data = subtitleDataRoot();
  console.log("promediaRoot:", root);
  console.log("subtitleDataRoot:", data);
  if (!root.endsWith(`${path.sep}promedia`) && !root.endsWith("/promedia")) {
    // Allow trailing slash variants
    if (!/[/\\]promedia[/\\]?$/.test(root)) {
      throw new Error(`unexpected promediaRoot: ${root}`);
    }
  }

  const dirs = await ensureSubtitleDirs();
  console.log("youtubeCache:", dirs.youtubeCache);
  await access(dirs.youtubeCache);
  const st = await stat(dirs.youtubeCache);
  console.log("youtubeCache isDir:", st.isDirectory());
  if (!st.isDirectory()) throw new Error("youtube-cache not a directory");

  const args = youtubeYtDlpOutputArgs(dirs.youtubeCache);
  console.log("yt-dlp args:", JSON.stringify(args));
  if (args[0] !== "-P" || args[2] !== "-o" || args[3] !== "%(id)s.%(ext)s") {
    throw new Error("bad output template args");
  }
  if (!path.isAbsolute(args[1]!)) throw new Error("cache -P must be absolute");
  if (!args[1]!.includes("youtube-cache")) {
    throw new Error("expected youtube-cache in -P");
  }

  const id = "3sur4BmjQt8";
  const fakeM4a = path.join(dirs.youtubeCache, `${id}.m4a`);
  const fakeWav = path.join(dirs.youtubeCache, `${id}.wav`);
  // Prefer wav over m4a when both exist; skip tiny/empty partials (<64 bytes).
  await writeFile(fakeM4a, "x".repeat(80));
  let found = await findCachedAudio(dirs.youtubeCache, id);
  console.log("findCachedAudio m4a:", found);
  if (found !== fakeM4a) throw new Error("glob resolve failed (m4a)");
  await writeFile(fakeWav, "y".repeat(80));
  found = await findCachedAudio(dirs.youtubeCache, id);
  console.log("findCachedAudio prefers wav:", found);
  if (found !== fakeWav) throw new Error("glob resolve failed (wav preferred)");
  await rm(fakeM4a);
  await rm(fakeWav);

  process.chdir(path.join(root, "apps", "hub"));
  const dirs2 = await ensureSubtitleDirs();
  console.log("after chdir cwd:", process.cwd());
  console.log("youtubeCache still:", dirs2.youtubeCache);
  if (dirs2.youtubeCache !== dirs.youtubeCache) {
    throw new Error("path drifted after chdir");
  }

  // ffmpeg postprocessor ENOENT must not look like a cache mkdir bug.
  const ffmpegGone = classifyYoutubeError(
    "ERROR: Postprocessing: ffprobe and ffmpeg not found. Please install or provide the path using --ffmpeg-location",
  );
  console.log("classify ffmpeg missing:", ffmpegGone);
  if (ffmpegGone !== "ffmpeg_missing") {
    throw new Error(`expected ffmpeg_missing, got ${ffmpegGone}`);
  }
  const renameOnly = classifyYoutubeError(
    "ERROR: unable to rename file: [Errno 2] No such file or directory",
  );
  console.log("classify cache path:", renameOnly);
  if (renameOnly !== "youtube_cache_path_error") {
    throw new Error(`expected youtube_cache_path_error, got ${renameOnly}`);
  }

  console.log("SMOKE_OK");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
