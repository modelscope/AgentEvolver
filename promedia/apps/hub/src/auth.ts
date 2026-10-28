import type { ClientRole } from "@promedia/shared";
import { sessionStore } from "./session-store.js";

export type SessionAccess =
  | { ok: true; role?: ClientRole }
  | { ok: false; status: 401 | 403; error: string };

/**
 * Validate session membership via wsTicket, short code, or opaque token.
 */
export function authorizeSessionAccess(
  sessionId: string,
  opts: {
    credential?: string | null;
    code?: string | null;
    accessToken?: string | null;
    requireRole?: ClientRole;
  },
): SessionAccess {
  const session = sessionStore.get(sessionId);
  if (!session) {
    return { ok: false, status: 401, error: "session_not_found" };
  }

  const credential = opts.credential?.trim();
  if (credential) {
    const ticket = sessionStore.getTicket(credential);
    if (!ticket || ticket.sessionId !== sessionId) {
      return { ok: false, status: 403, error: "invalid_ticket" };
    }
    if (opts.requireRole && ticket.role !== opts.requireRole) {
      return { ok: false, status: 403, error: "forbidden_role" };
    }
    return { ok: true, role: ticket.role };
  }

  const code = opts.code?.trim();
  if (code) {
    const byCode = sessionStore.findByCode(code);
    if (!byCode || byCode.id !== sessionId) {
      return { ok: false, status: 403, error: "invalid_code" };
    }
    if (opts.requireRole === "control") {
      return { ok: false, status: 403, error: "forbidden_role" };
    }
    return { ok: true };
  }

  const accessToken = opts.accessToken?.trim();
  if (accessToken) {
    const byToken = sessionStore.findByAccessToken(accessToken);
    if (!byToken || byToken.id !== sessionId) {
      return { ok: false, status: 403, error: "invalid_token" };
    }
    if (opts.requireRole === "control") {
      return { ok: false, status: 403, error: "forbidden_role" };
    }
    return { ok: true };
  }

  return { ok: false, status: 401, error: "missing_credentials" };
}
