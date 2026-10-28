import type { HubMessage } from "@promedia/shared";

export type SessionSocketHandlers = {
  onMessage: (msg: HubMessage) => void;
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (err: Event) => void;
};

export type DisconnectOptions = {
  /**
   * When false, tear down the socket without calling onClose.
   * Use during connect/reconnect so UI does not flash "closed" mid-handshake
   * and fight auto-reconnect effects.
   */
  notifyClose?: boolean;
};

/**
 * Minimal WS client: applies seq idempotency and supports resync on demand.
 */
export class SessionSocket {
  private ws: WebSocket | null = null;
  private lastSeq = -1;
  private readonly url: string;
  private readonly handlers: SessionSocketHandlers;

  constructor(url: string, handlers: SessionSocketHandlers) {
    this.url = url;
    this.handlers = handlers;
  }

  get appliedSeq(): number {
    return this.lastSeq;
  }

  connect(): void {
    this.disconnect({ notifyClose: false });
    this.bindSocket(new WebSocket(this.url));
  }

  disconnect(opts?: DisconnectOptions): void {
    if (!this.ws) {
      return;
    }
    const notifyClose = opts?.notifyClose !== false;
    const ws = this.ws;
    this.ws = null;
    const wasOpenOrConnecting =
      ws.readyState === WebSocket.OPEN ||
      ws.readyState === WebSocket.CONNECTING;
    ws.onopen = null;
    ws.onmessage = null;
    ws.onerror = null;
    ws.onclose = null;
    if (wasOpenOrConnecting) {
      try {
        ws.close();
      } catch {
        // ignore
      }
    }
    // Notify only for real teardown — otherwise UI can stay wsStatus "open"
    // while send() no-ops (handlers were cleared before the browser close event).
    // Silent during connect/reconnect so status stays "connecting".
    if (notifyClose) {
      this.handlers.onClose?.();
    }
  }

  get isOpen(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  /** @returns false if the socket is not open (caller should reconnect / warn). */
  send(type: string, payload?: unknown): boolean {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return false;
    }
    this.ws.send(JSON.stringify({ type, payload }));
    return true;
  }

  resync(): void {
    this.send("client.resync", { lastSeq: Math.max(this.lastSeq, 0) });
  }

  /**
   * Drop the socket without clearing lastSeq, then reconnect and ask for catch-up.
   */
  reconnectWithResync(): void {
    const last = this.lastSeq;
    this.disconnect({ notifyClose: false });
    this.lastSeq = last;
    this.bindSocket(new WebSocket(this.url), () => {
      this.send("client.resync", { lastSeq: Math.max(last, 0) });
    });
  }

  private bindSocket(ws: WebSocket, afterOpen?: () => void): void {
    this.ws = ws;

    ws.onopen = () => {
      this.handlers.onOpen?.();
      afterOpen?.();
    };

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(String(ev.data)) as HubMessage;
        if (typeof msg.seq === "number" && msg.seq <= this.lastSeq) {
          return;
        }
        if (typeof msg.seq === "number") {
          this.lastSeq = msg.seq;
        }
        this.handlers.onMessage(msg);
      } catch {
        // ignore malformed
      }
    };

    ws.onclose = () => {
      if (this.ws !== ws) {
        return;
      }
      this.ws = null;
      this.handlers.onClose?.();
    };

    ws.onerror = (err) => {
      if (this.ws !== ws) {
        return;
      }
      this.handlers.onError?.(err);
    };
  }
}
