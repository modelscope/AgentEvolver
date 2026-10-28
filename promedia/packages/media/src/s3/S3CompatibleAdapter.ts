import type { PutMeta, PutResult, StorageAdapter } from "../types.js";
import { LocalDiskAdapter } from "../local/LocalDiskAdapter.js";

export type S3CompatibleConfig = {
  endpoint?: string;
  bucket?: string;
  region?: string;
  accessKeyId?: string;
  secretAccessKey?: string;
  /** When incomplete, fall back to local disk (LAN stays working). */
  fallbackRoot?: string;
};

/**
 * S3-compatible adapter stub (subtarea 10).
 * Full multipart S3 is OUT of MVP weight; when credentials/bucket are incomplete,
 * falls back to LocalDiskAdapter so LAN/portable keep working.
 */
export class S3CompatibleAdapter implements StorageAdapter {
  readonly name = "s3-compatible";
  private readonly configured: boolean;
  private readonly fallback: LocalDiskAdapter;
  private readonly bucket: string;

  constructor(config: S3CompatibleConfig = {}) {
    this.bucket = config.bucket ?? process.env.S3_BUCKET ?? "";
    const endpoint = config.endpoint ?? process.env.S3_ENDPOINT ?? "";
    const key =
      config.accessKeyId ?? process.env.S3_ACCESS_KEY_ID ?? "";
    const secret =
      config.secretAccessKey ?? process.env.S3_SECRET_ACCESS_KEY ?? "";
    this.configured = Boolean(this.bucket && endpoint && key && secret);
    this.fallback = new LocalDiskAdapter(
      config.fallbackRoot ??
        process.env.MEDIA_ROOT ??
        `${process.cwd()}/data/media`,
    );
    if (!this.configured) {
      console.warn(
        "[promedia/media] S3-compatible adapter: incomplete config — using local disk fallback (LAN OK).",
      );
    } else {
      console.warn(
        `[promedia/media] S3-compatible stub active for bucket=${this.bucket} (writes still use local fallback until full SDK wired).`,
      );
    }
  }

  resolvePath(storageKey: string): string {
    return this.fallback.resolvePath(storageKey);
  }

  async put(
    sessionId: string,
    assetId: string,
    bytes: Uint8Array | Buffer,
    meta: PutMeta,
  ): Promise<PutResult> {
    // Stub: persist locally with s3-style key prefix for future migration.
    const result = await this.fallback.put(sessionId, assetId, bytes, meta);
    if (this.configured) {
      // Placeholder for PutObject — documented stub keeps LAN working.
      console.info(
        `[promedia/media] s3 stub would upload s3://${this.bucket}/${result.storageKey}`,
      );
    }
    return result;
  }

  async open(storageKey: string): Promise<{ path: string; mimeHint?: string }> {
    return this.fallback.open(storageKey);
  }

  async delete(storageKey: string): Promise<void> {
    await this.fallback.delete?.(storageKey);
  }
}

export function createS3CompatibleAdapter(
  config?: S3CompatibleConfig,
): S3CompatibleAdapter {
  return new S3CompatibleAdapter(config);
}
