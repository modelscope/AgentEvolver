export {
  ALLOWED_AUDIO_MIMES,
  ALLOWED_IMAGE_MIMES,
  ALLOWED_VIDEO_MIMES,
  extensionForMime,
  isAllowedAudioMime,
  isAllowedImageMime,
  isAllowedVideoMime,
  isQuicktimeMime,
  mimeFromFileName,
  resolveUploadMime,
  type AllowedAudioMime,
  type AllowedImageMime,
  type AllowedVideoMime,
  type MediaAsset,
  type PutMeta,
  type PutResult,
  type StorageAdapter,
} from "./types.js";
export {
  LocalDiskAdapter,
  createLocalDiskAdapter,
} from "./local/LocalDiskAdapter.js";
export {
  S3CompatibleAdapter,
  createS3CompatibleAdapter,
  type S3CompatibleConfig,
} from "./s3/S3CompatibleAdapter.js";
export { createStorageAdapter } from "./factory.js";

export const MEDIA_PACKAGE_VERSION = "0.0.12-mvp" as const;
