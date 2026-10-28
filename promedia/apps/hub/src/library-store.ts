import { mkdir, readdir, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { randomBytes } from "node:crypto";
import type {
  BiblePayload,
  LibraryCategory,
  LibraryItem,
  LibrarySongPayload,
  TextPayload,
} from "@promedia/shared";
import { LIBRARY_CATEGORIES, songToTextPayload } from "@promedia/shared";
import { assetStore } from "./asset-store.js";
import {
  importBibleFile,
  looksLikeBibleFile,
} from "./bible-import.js";
import {
  canonToDiskJson,
  clearLocalCanonCache,
  libraryItemToLocalVersion,
  metaFromCanon,
  type LocalBibleVersion,
} from "./bible/local-canon.js";
import {
  importPresentationFile,
  isMacroPresentationName,
  type RenderedSlide,
} from "./presentation-import.js";
import {
  looksLikeQueleaSongXml,
  parseQueleaQsp,
  parseQueleaSongFile,
  songIdentityKey,
} from "./quelea-import.js";

function nowIso(): string {
  return new Date().toISOString();
}

function newId(): string {
  return randomBytes(10).toString("hex");
}

/**
 * Prefer monorepo `promedia/data/library` (gitignored) so imports work the same
 * whether the hub is started from repo root or `apps/hub` via pnpm filter.
 */
function libraryRoot(): string {
  if (process.env.LIBRARY_ROOT?.trim()) {
    return process.env.LIBRARY_ROOT.trim();
  }
  // src|dist → apps/hub → apps → promedia
  const promediaRoot = path.resolve(
    path.dirname(fileURLToPath(import.meta.url)),
    "../../..",
  );
  return path.join(promediaRoot, "data", "library");
}

function indexPath(): string {
  return path.join(libraryRoot(), "index.json");
}

function assertSafeId(id: string): void {
  if (!id || id.includes("..") || id.includes("/") || id.includes("\\")) {
    throw new Error("invalid_id");
  }
}

function isCategory(value: string): value is LibraryCategory {
  return (LIBRARY_CATEGORIES as readonly string[]).includes(value);
}

function mimeFromName(name: string): string {
  const lower = name.toLowerCase();
  if (lower.endsWith(".png")) return "image/png";
  if (lower.endsWith(".jpg") || lower.endsWith(".jpeg")) return "image/jpeg";
  if (lower.endsWith(".gif")) return "image/gif";
  if (lower.endsWith(".webp")) return "image/webp";
  if (lower.endsWith(".svg")) return "image/svg+xml";
  if (lower.endsWith(".mp4")) return "video/mp4";
  if (lower.endsWith(".webm")) return "video/webm";
  if (lower.endsWith(".mov")) return "video/quicktime";
  if (lower.endsWith(".mp3")) return "audio/mpeg";
  if (lower.endsWith(".m4a")) return "audio/mp4";
  if (lower.endsWith(".aac")) return "audio/aac";
  if (lower.endsWith(".wav")) return "audio/wav";
  if (lower.endsWith(".flac")) return "audio/flac";
  if (lower.endsWith(".ogg") || lower.endsWith(".oga")) return "audio/ogg";
  if (lower.endsWith(".opus")) return "audio/opus";
  if (lower.endsWith(".pdf")) return "application/pdf";
  if (lower.endsWith(".pptx")) {
    return "application/vnd.openxmlformats-officedocument.presentationml.presentation";
  }
  if (lower.endsWith(".ppsx")) {
    return "application/vnd.openxmlformats-officedocument.presentationml.slideshow";
  }
  if (lower.endsWith(".potx")) {
    return "application/vnd.openxmlformats-officedocument.presentationml.template";
  }
  if (lower.endsWith(".odp")) {
    return "application/vnd.oasis.opendocument.presentation";
  }
  if (lower.endsWith(".ppt")) return "application/vnd.ms-powerpoint";
  if (lower.endsWith(".qsp")) return "application/zip";
  if (lower.endsWith(".xml") || lower.endsWith(".xmm")) return "application/xml";
  return "application/octet-stream";
}

function classifyUpload(
  fileName: string,
  mime: string,
  opts?: { preferBiblias?: boolean; bytes?: Buffer },
): LibraryCategory | { error: string; message: string } {
  const lower = fileName.toLowerCase();
  if (isMacroPresentationName(fileName)) {
    return {
      error: "macros_not_supported",
      message:
        "No se admiten presentaciones con macros (.pptm / .ppsm) por seguridad. Guardá una copia sin macros (.pptx / .ppsx) o exportá a PDF e importá de nuevo.",
    };
  }
  if (lower.endsWith(".qsp") || looksLikeQueleaSongXml(fileName, opts?.bytes)) {
    return "canciones";
  }
  const bibleLikely =
    opts?.preferBiblias ||
    looksLikeBibleFile(fileName, opts?.bytes) ||
    lower.endsWith(".xmm");
  if (bibleLikely && (lower.endsWith(".xml") || lower.endsWith(".xmm") || lower.endsWith(".pdf"))) {
    return "biblias";
  }
  if (
    lower.endsWith(".pptx") ||
    lower.endsWith(".ppt") ||
    lower.endsWith(".ppsx") ||
    lower.endsWith(".potx") ||
    lower.endsWith(".odp") ||
    lower.endsWith(".pdf") ||
    mime.includes("presentation") ||
    mime === "application/pdf"
  ) {
    return "presentaciones";
  }
  if (
    mime.startsWith("image/") ||
    /\.(png|jpe?g|gif|webp|svg)$/i.test(lower)
  ) {
    return "imagenes";
  }
  if (
    mime.startsWith("video/") ||
    /\.(mp4|webm|mov)$/i.test(lower)
  ) {
    return "videos";
  }
  if (
    mime.startsWith("audio/") ||
    /\.(mp3|m4a|aac|wav|flac|ogg|oga|opus)$/i.test(lower)
  ) {
    return "audios";
  }
  if (/\.(txt|md)$/i.test(lower) || mime.startsWith("text/")) {
    return "textos";
  }
  if (/\.xml$/i.test(lower)) {
    // Generic XML without Bible markers → textos (legacy)
    return "textos";
  }
  return {
    error: "unsupported_type",
    message:
      "Tipo no soportado para la biblioteca. Subí imagen, vídeo, audio, texto, Biblia (.xml/.xmm/.pdf), cancionero Quelea (.qsp/.xml) o presentación (.pptx/.pdf/…).",
  };
}

/** Default folder Julio (and operators) can drop Bible files into — gitignored under data/. */
export function bibleImportFolder(): string {
  return (
    process.env.BIBLE_IMPORT_DIR?.trim() ||
    path.join(libraryRoot(), "bibles", "import")
  );
}

/** Drop folder for Quelea .qsp packs / .xml songs — gitignored under data/. */
export function songImportFolder(): string {
  return (
    process.env.SONG_IMPORT_DIR?.trim() ||
    path.join(libraryRoot(), "songs", "import")
  );
}

/**
 * Create the on-disk library tree so Finder shows drop folders after first hub
 * launch — no manual `mkdir`. Content stays gitignored; we only create empty dirs.
 *
 * Layout under `data/library/`:
 * - `files/` — stored library items (songs, presentations, media, texts, …)
 * - `bibles/import/` — operator drop folder for Bible XML/XMM/PDF
 * - `songs/import/` — operator drop folder for Quelea .qsp / song .xml
 */
export async function ensureLibraryLayout(): Promise<{
  root: string;
  bibleImport: string;
  songImport: string;
}> {
  const root = libraryRoot();
  const bibleImport = bibleImportFolder();
  const songImport = songImportFolder();
  await mkdir(root, { recursive: true });
  await mkdir(path.join(root, "files"), { recursive: true });
  await mkdir(path.join(root, "bibles"), { recursive: true });
  await mkdir(bibleImport, { recursive: true });
  await mkdir(path.join(root, "songs"), { recursive: true });
  await mkdir(songImport, { recursive: true });
  return { root, bibleImport, songImport };
}

/**
 * File-backed local media library (JSON index + files under data/library).
 * Cloud sync is out of scope for Phase 1.
 */
export class LibraryStore {
  private items = new Map<string, LibraryItem>();
  private loaded = false;
  private layoutReady = false;

  private async ensureLayout(): Promise<void> {
    if (this.layoutReady) {
      return;
    }
    await ensureLibraryLayout();
    this.layoutReady = true;
  }

  private async ensureLoaded(): Promise<void> {
    await this.ensureLayout();
    if (this.loaded) {
      return;
    }
    this.loaded = true;
    try {
      const raw = await readFile(indexPath(), "utf8");
      const parsed = JSON.parse(raw) as { items?: LibraryItem[] };
      for (const item of parsed.items ?? []) {
        if (item?.id && item.category && item.title) {
          this.items.set(item.id, item);
        }
      }
    } catch {
      // missing = empty
    }
  }

  private async persist(): Promise<void> {
    const root = libraryRoot();
    await mkdir(root, { recursive: true });
    const items = [...this.items.values()].sort((a, b) =>
      a.updatedAt < b.updatedAt ? 1 : -1,
    );
    await writeFile(
      indexPath(),
      JSON.stringify({ items }, null, 2),
      "utf8",
    );
  }

  private resolveUnderLibrary(relativePath: string): string {
    if (
      !relativePath ||
      relativePath.includes("..") ||
      path.isAbsolute(relativePath)
    ) {
      throw new Error("invalid_path");
    }
    const abs = path.resolve(libraryRoot(), relativePath);
    const root = path.resolve(libraryRoot());
    if (abs !== root && !abs.startsWith(root + path.sep)) {
      throw new Error("path_escape");
    }
    return abs;
  }

  async list(opts?: {
    category?: string;
    q?: string;
  }): Promise<LibraryItem[]> {
    await this.ensureLoaded();
    let list = [...this.items.values()];
    if (opts?.category && isCategory(opts.category)) {
      list = list.filter((i) => i.category === opts.category);
    }
    const q = opts?.q?.trim().toLowerCase();
    if (q) {
      list = list.filter((i) => {
        const hay = [
          i.title,
          i.song?.artist,
          i.song?.title,
          i.original?.originalName,
          i.bible?.reference,
          i.bibleVersion?.versionId,
          i.bibleVersion?.shortLabel,
          i.bibleVersion?.label,
        ]
          .filter(Boolean)
          .join(" ")
          .toLowerCase();
        return hay.includes(q);
      });
    }
    return list.sort((a, b) => (a.updatedAt < b.updatedAt ? 1 : -1));
  }

  async get(id: string): Promise<LibraryItem | undefined> {
    await this.ensureLoaded();
    return this.items.get(id);
  }

  async createSong(
    song: LibrarySongPayload,
  ): Promise<LibraryItem | { error: string; message: string }> {
    await this.ensureLoaded();
    const title = song.title.trim();
    if (!title) {
      return { error: "missing_title", message: "La canción necesita un título." };
    }
    const stanzas = (song.stanzas ?? [])
      .map((s) => ({
        id: s.id || newId(),
        lines: (s.lines ?? []).map((l) => String(l)),
      }))
      .filter((s) => s.lines.some((l) => l.trim().length > 0) || true);
    if (stanzas.length === 0) {
      stanzas.push({ id: newId(), lines: [""] });
    }
    const id = newId();
    const ts = nowIso();
    const item: LibraryItem = {
      id,
      category: "canciones",
      title,
      createdAt: ts,
      updatedAt: ts,
      song: {
        title,
        artist: song.artist?.trim() || undefined,
        stanzas,
      },
    };
    this.items.set(id, item);
    await this.persist();
    return item;
  }

  async updateSong(
    id: string,
    song: LibrarySongPayload,
  ): Promise<LibraryItem | { error: string; message: string }> {
    await this.ensureLoaded();
    const existing = this.items.get(id);
    if (!existing || existing.category !== "canciones") {
      return { error: "not_found", message: "Canción no encontrada." };
    }
    const title = song.title.trim();
    if (!title) {
      return { error: "missing_title", message: "La canción necesita un título." };
    }
    const updated: LibraryItem = {
      ...existing,
      title,
      updatedAt: nowIso(),
      song: {
        title,
        artist: song.artist?.trim() || undefined,
        stanzas: (song.stanzas ?? []).map((s) => ({
          id: s.id || newId(),
          lines: (s.lines ?? []).map((l) => String(l)),
        })),
      },
    };
    this.items.set(id, updated);
    await this.persist();
    return updated;
  }

  async saveText(opts: {
    title: string;
    text: TextPayload;
  }): Promise<LibraryItem | { error: string; message: string }> {
    await this.ensureLoaded();
    const title = opts.title.trim() || opts.text.title?.trim() || "Texto";
    const id = newId();
    const ts = nowIso();
    const item: LibraryItem = {
      id,
      category: "textos",
      title,
      createdAt: ts,
      updatedAt: ts,
      text: opts.text,
    };
    this.items.set(id, item);
    await this.persist();
    return item;
  }

  async saveBible(
    payload: BiblePayload,
  ): Promise<LibraryItem | { error: string; message: string }> {
    await this.ensureLoaded();
    const title =
      payload.reference?.trim() ||
      `${payload.bookName} ${payload.chapter}` ||
      "Biblia";
    const id = newId();
    const ts = nowIso();
    const item: LibraryItem = {
      id,
      category: "biblias",
      title,
      createdAt: ts,
      updatedAt: ts,
      bible: payload,
    };
    this.items.set(id, item);
    await this.persist();
    return item;
  }

  private async writeSlides(
    itemId: string,
    slides: RenderedSlide[],
  ): Promise<LibraryItem["slides"]> {
    const out: NonNullable<LibraryItem["slides"]> = [];
    for (const slide of slides) {
      const rel = path.join(
        "files",
        itemId,
        "slides",
        slide.fileName,
      );
      const abs = this.resolveUnderLibrary(rel);
      await mkdir(path.dirname(abs), { recursive: true });
      await writeFile(abs, slide.bytes);
      out.push({
        id: slide.id,
        relativePath: rel.split(path.sep).join("/"),
        mime: slide.mime,
      });
    }
    return out;
  }

  async listLocalBibleVersions(): Promise<LocalBibleVersion[]> {
    await this.ensureLoaded();
    const out: LocalBibleVersion[] = [];
    for (const item of this.items.values()) {
      if (item.category !== "biblias" || !item.bibleVersion) {
        continue;
      }
      const local = libraryItemToLocalVersion(item, (rel) =>
        this.resolveUnderLibrary(rel),
      );
      if (local) {
        out.push(local);
      }
    }
    return out.sort((a, b) => a.shortLabel.localeCompare(b.shortLabel, "es"));
  }

  async getLocalBibleVersion(
    versionId: string,
  ): Promise<LocalBibleVersion | undefined> {
    const key = versionId.trim().toLowerCase();
    const all = await this.listLocalBibleVersions();
    return all.find((v) => v.versionId.toLowerCase() === key);
  }

  private async persistBibleImport(opts: {
    bytes: Buffer;
    originalName: string;
    mime: string;
    title?: string;
  }): Promise<LibraryItem | { error: string; message: string }> {
    const imported = await importBibleFile({
      bytes: opts.bytes,
      originalName: opts.originalName,
      mime: opts.mime,
    });
    if (!imported.ok) {
      return { error: imported.error, message: imported.message };
    }
    const canon = imported.canon;

    // Replace any previous library entry for the same versionId.
    for (const existing of [...this.items.values()]) {
      if (
        existing.category === "biblias" &&
        existing.bibleVersion?.versionId.toLowerCase() ===
          canon.versionId.toLowerCase()
      ) {
        this.items.delete(existing.id);
        await rm(path.join(libraryRoot(), "files", existing.id), {
          recursive: true,
          force: true,
        }).catch(() => undefined);
      }
    }
    clearLocalCanonCache(canon.versionId);

    const id = newId();
    const ext = path.extname(opts.originalName) || "";
    const originalRel = path
      .join("files", id, `original${ext}`)
      .split(path.sep)
      .join("/");
    const canonRel = path
      .join("files", id, "canon.json")
      .split(path.sep)
      .join("/");
    const absOriginal = this.resolveUnderLibrary(originalRel);
    const absCanon = this.resolveUnderLibrary(canonRel);
    await mkdir(path.dirname(absOriginal), { recursive: true });
    await writeFile(absOriginal, opts.bytes);
    await writeFile(absCanon, canonToDiskJson(canon), "utf8");

    const ts = nowIso();
    const item: LibraryItem = {
      id,
      category: "biblias",
      title: opts.title?.trim() || `${canon.shortLabel} — ${canon.label}`,
      createdAt: ts,
      updatedAt: ts,
      bibleVersion: metaFromCanon(canon, canonRel),
      original: {
        relativePath: originalRel,
        mime: opts.mime,
        originalName: opts.originalName,
        size: opts.bytes.byteLength,
      },
    };
    this.items.set(id, item);
    await this.persist();
    return item;
  }

  /**
   * Upsert a song by title+author (case-insensitive). Used by Quelea import so
   * re-importing the same pack updates lyrics instead of duplicating.
   */
  async upsertSongByIdentity(
    song: LibrarySongPayload,
  ): Promise<
    | { item: LibraryItem; action: "created" | "updated" }
    | { error: string; message: string }
  > {
    await this.ensureLoaded();
    const title = song.title.trim();
    if (!title) {
      return { error: "missing_title", message: "La canción necesita un título." };
    }
    const key = songIdentityKey(title, song.artist);
    for (const existing of this.items.values()) {
      if (existing.category !== "canciones" || !existing.song) {
        continue;
      }
      if (
        songIdentityKey(existing.song.title, existing.song.artist) === key
      ) {
        const updated = await this.updateSong(existing.id, song);
        if ("error" in updated) {
          return updated;
        }
        return { item: updated, action: "updated" };
      }
    }
    const created = await this.createSong(song);
    if ("error" in created) {
      return created;
    }
    return { item: created, action: "created" };
  }

  async importQueleaUpload(opts: {
    bytes: Buffer;
    originalName: string;
  }): Promise<{
    imported: LibraryItem[];
    updated: LibraryItem[];
    skipped: Array<{ file: string; error: string; message: string }>;
  }> {
    await this.ensureLoaded();
    const originalName = opts.originalName || "canciones.qsp";
    const lower = originalName.toLowerCase();
    const imported: LibraryItem[] = [];
    const updated: LibraryItem[] = [];
    const skipped: Array<{ file: string; error: string; message: string }> = [];

    if (lower.endsWith(".qsp")) {
      const pack = await parseQueleaQsp({
        bytes: opts.bytes,
        originalName,
      });
      if (!pack.ok) {
        return {
          imported,
          updated,
          skipped: [
            {
              file: originalName,
              error: pack.error,
              message: pack.message,
            },
          ],
        };
      }
      skipped.push(...pack.skipped);
      for (const entry of pack.songs) {
        const result = await this.upsertSongByIdentity(entry.song);
        if ("error" in result) {
          skipped.push({
            file: entry.sourceName || entry.song.title,
            error: result.error,
            message: result.message,
          });
          continue;
        }
        if (result.action === "updated") {
          updated.push(result.item);
        } else {
          imported.push(result.item);
        }
      }
      return { imported, updated, skipped };
    }

    const parsed = await parseQueleaSongFile({
      bytes: opts.bytes,
      originalName,
    });
    if (!parsed.ok) {
      return {
        imported,
        updated,
        skipped: [
          {
            file: originalName,
            error: parsed.error,
            message: parsed.message,
          },
        ],
      };
    }
    const result = await this.upsertSongByIdentity(parsed.song);
    if ("error" in result) {
      skipped.push({
        file: originalName,
        error: result.error,
        message: result.message,
      });
    } else if (result.action === "updated") {
      updated.push(result.item);
    } else {
      imported.push(result.item);
    }
    return { imported, updated, skipped };
  }

  /**
   * Import Quelea .qsp / .xml from the operator folder
   * (default: data/library/songs/import/).
   */
  async importSongsFromFolder(opts?: {
    folder?: string;
  }): Promise<{
    folder: string;
    imported: LibraryItem[];
    updated: LibraryItem[];
    skipped: Array<{ file: string; error: string; message: string }>;
  }> {
    await this.ensureLoaded();
    const folder = opts?.folder?.trim() || songImportFolder();
    await mkdir(folder, { recursive: true });
    let entries: string[] = [];
    try {
      entries = await readdir(folder);
    } catch {
      return { folder, imported: [], updated: [], skipped: [] };
    }
    const imported: LibraryItem[] = [];
    const updated: LibraryItem[] = [];
    const skipped: Array<{ file: string; error: string; message: string }> = [];
    for (const name of entries.sort()) {
      const lower = name.toLowerCase();
      if (!lower.endsWith(".qsp") && !lower.endsWith(".xml")) {
        continue;
      }
      const abs = path.join(folder, name);
      let bytes: Buffer;
      try {
        bytes = await readFile(abs);
      } catch (err) {
        skipped.push({
          file: name,
          error: "read_failed",
          message: err instanceof Error ? err.message : String(err),
        });
        continue;
      }
      if (
        lower.endsWith(".xml") &&
        !looksLikeQueleaSongXml(name, bytes)
      ) {
        skipped.push({
          file: name,
          error: "not_quelea_song",
          message: "No parece una canción Quelea; se omitió.",
        });
        continue;
      }
      const result = await this.importQueleaUpload({
        bytes,
        originalName: name,
      });
      imported.push(...result.imported);
      updated.push(...result.updated);
      skipped.push(...result.skipped);
    }
    return { folder, imported, updated, skipped };
  }

  async uploadFile(opts: {
    bytes: Buffer;
    originalName: string;
    mime?: string;
    title?: string;
    /** Force Bible import path (Biblioteca → Biblias). */
    preferBiblias?: boolean;
  }): Promise<LibraryItem | { error: string; message: string }> {
    await this.ensureLoaded();
    const originalName = opts.originalName || "archivo";
    const mime = opts.mime || mimeFromName(originalName);
    const classified = classifyUpload(originalName, mime, {
      preferBiblias: opts.preferBiblias,
      bytes: opts.bytes,
    });
    if (typeof classified !== "string") {
      return classified;
    }
    const category = classified;

    if (category === "biblias") {
      return this.persistBibleImport({
        bytes: opts.bytes,
        originalName,
        mime,
        title: opts.title,
      });
    }

    if (category === "canciones") {
      const bulk = await this.importQueleaUpload({
        bytes: opts.bytes,
        originalName,
      });
      const first = bulk.imported[0] ?? bulk.updated[0];
      if (!first) {
        const skip = bulk.skipped[0];
        return {
          error: skip?.error || "import_failed",
          message:
            skip?.message ||
            "No se pudo importar la canción o el pack Quelea.",
        };
      }
      // Multipart upload historically returns one item; for .qsp packs the
      // dedicated /library/songs/import endpoint returns the full summary.
      return first;
    }

    const id = newId();
    const ext = path.extname(originalName) || "";
    const originalRel = path
      .join("files", id, `original${ext}`)
      .split(path.sep)
      .join("/");
    const absOriginal = this.resolveUnderLibrary(originalRel);
    await mkdir(path.dirname(absOriginal), { recursive: true });
    await writeFile(absOriginal, opts.bytes);

    const ts = nowIso();
    let item: LibraryItem = {
      id,
      category,
      title:
        opts.title?.trim() ||
        originalName.replace(/\.[^.]+$/, "") ||
        originalName,
      createdAt: ts,
      updatedAt: ts,
      original: {
        relativePath: originalRel,
        mime,
        originalName,
        size: opts.bytes.byteLength,
      },
    };

    if (category === "presentaciones") {
      const imported = await importPresentationFile({
        bytes: opts.bytes,
        originalName,
        mime,
      });
      if (!imported.ok) {
        await rm(path.join(libraryRoot(), "files", id), {
          recursive: true,
          force: true,
        }).catch(() => undefined);
        return { error: imported.error, message: imported.message };
      }
      item = {
        ...item,
        title: opts.title?.trim() || imported.title || item.title,
        slides: await this.writeSlides(id, imported.slides),
      };
    }

    this.items.set(id, item);
    await this.persist();
    return item;
  }

  /**
   * Import every .xml / .xmm / Bible-like .pdf from a local folder
   * (default: data/library/bibles/import/). Skips broken files with Spanish errors.
   */
  async importBiblesFromFolder(opts?: {
    folder?: string;
  }): Promise<{
    folder: string;
    imported: LibraryItem[];
    skipped: Array<{ file: string; error: string; message: string }>;
  }> {
    await this.ensureLoaded();
    const folder = opts?.folder?.trim() || bibleImportFolder();
    await mkdir(folder, { recursive: true });
    let entries: string[] = [];
    try {
      entries = await readdir(folder);
    } catch {
      return { folder, imported: [], skipped: [] };
    }
    const imported: LibraryItem[] = [];
    const skipped: Array<{ file: string; error: string; message: string }> = [];
    for (const name of entries.sort()) {
      const lower = name.toLowerCase();
      if (
        !lower.endsWith(".xml") &&
        !lower.endsWith(".xmm") &&
        !lower.endsWith(".pdf")
      ) {
        continue;
      }
      const abs = path.join(folder, name);
      let bytes: Buffer;
      try {
        bytes = await readFile(abs);
      } catch (err) {
        skipped.push({
          file: name,
          error: "read_failed",
          message: err instanceof Error ? err.message : String(err),
        });
        continue;
      }
      if (!looksLikeBibleFile(name, bytes) && !lower.endsWith(".xmm")) {
        skipped.push({
          file: name,
          error: "not_bible",
          message: "No parece un archivo de Biblia; se omitió.",
        });
        continue;
      }
      const result = await this.persistBibleImport({
        bytes,
        originalName: name,
        mime: mimeFromName(name),
      });
      if ("error" in result) {
        skipped.push({
          file: name,
          error: result.error,
          message: result.message,
        });
        continue;
      }
      imported.push(result);
    }
    return { folder, imported, skipped };
  }

  async remove(
    id: string,
  ): Promise<boolean | { error: string; message: string }> {
    await this.ensureLoaded();
    assertSafeId(id);
    if (!this.items.has(id)) {
      return false;
    }
      const removed = this.items.get(id);
    this.items.delete(id);
    if (removed?.bibleVersion?.versionId) {
      clearLocalCanonCache(removed.bibleVersion.versionId);
    }
    await this.persist();
    await rm(path.join(libraryRoot(), "files", id), {
      recursive: true,
      force: true,
    }).catch(() => undefined);
    return true;
  }

  async openFile(
    itemId: string,
    kind: "original" | { slideId: string },
  ): Promise<{ path: string; mime: string; downloadName: string } | null> {
    await this.ensureLoaded();
    const item = this.items.get(itemId);
    if (!item) {
      return null;
    }
    if (kind === "original") {
      if (!item.original) {
        return null;
      }
      return {
        path: this.resolveUnderLibrary(item.original.relativePath),
        mime: item.original.mime,
        downloadName: item.original.originalName,
      };
    }
    const slide = item.slides?.find((s) => s.id === kind.slideId);
    if (!slide) {
      return null;
    }
    return {
      path: this.resolveUnderLibrary(slide.relativePath),
      mime: slide.mime,
      downloadName: path.basename(slide.relativePath),
    };
  }

  /**
   * Materialize a library item into session playlist item(s) + session assets.
   */
  async addToSession(opts: {
    sessionId: string;
    libraryItemId: string;
    order: number;
  }): Promise<
    | { item: import("@promedia/shared").PlaylistItem }
    | { error: string; message: string }
  > {
    await this.ensureLoaded();
    const lib = this.items.get(opts.libraryItemId);
    if (!lib) {
      return { error: "not_found", message: "Ítem de biblioteca no encontrado." };
    }

    if (lib.category === "canciones" && lib.song) {
      const payload = songToTextPayload(lib.song);
      return {
        item: {
          id: newId(),
          type: "text",
          order: opts.order,
          payload,
        },
      };
    }

    if (lib.category === "textos" && lib.text) {
      return {
        item: {
          id: newId(),
          type: "text",
          order: opts.order,
          payload: lib.text,
        },
      };
    }

    if (lib.category === "biblias" && lib.bible) {
      return {
        item: {
          id: newId(),
          type: "bible",
          order: opts.order,
          payload: lib.bible,
        },
      };
    }

    if (lib.category === "biblias" && lib.bibleVersion) {
      return {
        error: "bible_version_not_playlist",
        message:
          `La traducción «${lib.bibleVersion.shortLabel}» quedó en la biblioteca local. Abrí Biblia en el Guion, elegí esa versión y buscá el pasaje para proyectarlo.`,
      };
    }

    if (lib.category === "imagenes" && lib.original) {
      const bytes = await readFile(
        this.resolveUnderLibrary(lib.original.relativePath),
      );
      const asset = await assetStore.putProjectionImage(
        opts.sessionId,
        bytes,
        lib.original.mime.startsWith("image/")
          ? lib.original.mime
          : "image/png",
        lib.original.originalName,
      );
      if ("error" in asset) {
        return {
          error: asset.error,
          message: "No se pudo copiar la imagen a la sesión.",
        };
      }
      return {
        item: {
          id: newId(),
          type: "image",
          order: opts.order,
          payload: {
            assetId: asset.id,
            fit: "contain",
            title: lib.title,
          },
        },
      };
    }

    if (lib.category === "videos" && lib.original) {
      const bytes = await readFile(
        this.resolveUnderLibrary(lib.original.relativePath),
      );
      const asset = await assetStore.putByMime(
        opts.sessionId,
        bytes,
        lib.original.mime,
        lib.original.originalName,
      );
      if ("error" in asset) {
        return {
          error: asset.error,
          message: "No se pudo copiar el vídeo a la sesión.",
        };
      }
      return {
        item: {
          id: newId(),
          type: "video",
          order: opts.order,
          payload: {
            sourceKind: "file",
            assetId: asset.id,
            title: lib.title,
            autoplay: true,
            muted: false,
            fit: "contain",
          },
        },
      };
    }

    if (lib.category === "audios" && lib.original) {
      const bytes = await readFile(
        this.resolveUnderLibrary(lib.original.relativePath),
      );
      const asset = await assetStore.putByMime(
        opts.sessionId,
        bytes,
        lib.original.mime,
        lib.original.originalName,
      );
      if ("error" in asset) {
        return {
          error: asset.error,
          message: "No se pudo copiar el audio a la sesión.",
        };
      }
      return {
        item: {
          id: newId(),
          type: "audio",
          order: opts.order,
          payload: {
            assetId: asset.id,
            title: lib.title,
            autoplay: true,
            loop: false,
          },
        },
      };
    }

    if (lib.category === "presentaciones") {
      if (!lib.slides || lib.slides.length === 0) {
        return {
          error: "no_slides",
          message:
            "Esta presentación no tiene diapositivas proyectables. Reimportá como .pptx o .pdf.",
        };
      }
      const slides: { id: string; assetId: string }[] = [];
      for (const slide of lib.slides) {
        const bytes = await readFile(
          this.resolveUnderLibrary(slide.relativePath),
        );
        const asset = await assetStore.putProjectionImage(
          opts.sessionId,
          bytes,
          slide.mime,
          path.basename(slide.relativePath),
        );
        if ("error" in asset) {
          return {
            error: asset.error,
            message: "No se pudieron copiar las diapositivas a la sesión.",
          };
        }
        slides.push({ id: slide.id, assetId: asset.id });
      }
      return {
        item: {
          id: newId(),
          type: "presentation",
          order: opts.order,
          payload: {
            title: lib.title,
            fit: "contain",
            slides,
            originalName: lib.original?.originalName,
            libraryItemId: lib.id,
          },
        },
      };
    }

    return {
      error: "unsupported_item",
      message: "Este ítem de biblioteca no se puede añadir al Guion aún.",
    };
  }
}

export const libraryStore = new LibraryStore();
