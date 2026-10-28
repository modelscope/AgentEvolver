import { mkdir, unlink, writeFile } from "node:fs/promises";
import path from "node:path";
import {
  extensionForMime,
  type PutMeta,
  type PutResult,
  type StorageAdapter,
} from "../types.js";

function assertSafeId(value: string, label: string): void {
  if (!value || value.includes("..") || value.includes("/") || value.includes("\\")) {
    throw new Error(`invalid_${label}`);
  }
}

/**
 * Local disk adapter: files under `<root>/<sessionId>/<assetId><ext>`.
 */
export class LocalDiskAdapter implements StorageAdapter {
  readonly name = "local";

  constructor(private readonly rootDir: string) {}

  resolvePath(storageKey: string): string {
    if (!storageKey || storageKey.includes("..")) {
      throw new Error("invalid_storage_key");
    }
    const resolved = path.resolve(this.rootDir, storageKey);
    const root = path.resolve(this.rootDir);
    if (resolved !== root && !resolved.startsWith(root + path.sep)) {
      throw new Error("path_escape");
    }
    return resolved;
  }

  async put(
    sessionId: string,
    assetId: string,
    bytes: Uint8Array | Buffer,
    meta: PutMeta,
  ): Promise<PutResult> {
    assertSafeId(sessionId, "sessionId");
    assertSafeId(assetId, "assetId");
    const ext = extensionForMime(meta.mime);
    const storageKey = path.join(sessionId, `${assetId}${ext}`);
    const abs = this.resolvePath(storageKey);
    await mkdir(path.dirname(abs), { recursive: true });
    await writeFile(abs, bytes);
    return { storageKey: storageKey.split(path.sep).join("/") };
  }

  async open(storageKey: string): Promise<{ path: string }> {
    return { path: this.resolvePath(storageKey) };
  }

  async delete(storageKey: string): Promise<void> {
    const abs = this.resolvePath(storageKey);
    try {
      await unlink(abs);
    } catch {
      // ignore missing
    }
  }
}

export function createLocalDiskAdapter(rootDir?: string): LocalDiskAdapter {
  const root =
    rootDir ??
    process.env.MEDIA_ROOT ??
    path.join(process.cwd(), "data", "media");
  return new LocalDiskAdapter(root);
}
