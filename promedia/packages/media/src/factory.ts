import { createLocalDiskAdapter } from "./local/LocalDiskAdapter.js";
import { createS3CompatibleAdapter } from "./s3/S3CompatibleAdapter.js";
import type { StorageAdapter } from "./types.js";

/**
 * Pick storage from env (subtarea 10).
 * - PROMEDIA_MODE=cloud or MEDIA_ADAPTER=s3 → S3-compatible stub (falls back to local if incomplete)
 * - otherwise local disk (LAN / portable default)
 */
export function createStorageAdapter(): StorageAdapter {
  const mode = (process.env.PROMEDIA_MODE ?? "lan").toLowerCase();
  const adapter = (process.env.MEDIA_ADAPTER ?? "").toLowerCase();
  if (mode === "cloud" || adapter === "s3" || adapter === "s3-compatible") {
    return createS3CompatibleAdapter();
  }
  return createLocalDiskAdapter();
}
