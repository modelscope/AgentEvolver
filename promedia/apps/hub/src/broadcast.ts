import type { HubMessage } from "@promedia/shared";

type Broadcaster = (sessionId: string, message: HubMessage) => void;

let broadcaster: Broadcaster | null = null;

export function setSessionBroadcaster(fn: Broadcaster): void {
  broadcaster = fn;
}

export function broadcastSession(sessionId: string, message: HubMessage): void {
  broadcaster?.(sessionId, message);
}
