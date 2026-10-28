import { randomBytes, randomInt } from "node:crypto";

/** Unambiguous alphabet for short join codes (no 0/O/1/I). */
const CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";

export function generateShortCode(length = 4): string {
  let out = "";
  for (let i = 0; i < length; i += 1) {
    out += CODE_ALPHABET[randomInt(CODE_ALPHABET.length)];
  }
  return out;
}

/** Opaque access token (≥22 URL-safe chars). Never expires. */
export function generateAccessToken(): string {
  return randomBytes(18).toString("base64url");
}

/** Short WS ticket issued after create/join (primary WS credential). */
export function generateWsTicket(): string {
  return randomBytes(9).toString("base64url");
}

export function generateSessionId(): string {
  return randomBytes(12).toString("hex");
}
