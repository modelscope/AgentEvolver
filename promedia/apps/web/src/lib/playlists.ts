import { hubHttpBase } from "./hub";

export type SavedPlaylistSummary = {
  id: string;
  name: string;
  createdAt: string;
  updatedAt: string;
  itemCount: number;
};

export async function listSavedPlaylists(): Promise<SavedPlaylistSummary[]> {
  const res = await fetch(`${hubHttpBase()}/api/v1/playlists`);
  if (!res.ok) {
    throw new Error(`list_playlists_${res.status}`);
  }
  const data = (await res.json()) as { playlists: SavedPlaylistSummary[] };
  return data.playlists;
}

export async function saveSessionPlaylist(
  sessionId: string,
  credential: string,
  name: string,
  id?: string,
): Promise<{ id: string; name: string }> {
  const res = await fetch(
    `${hubHttpBase()}/api/v1/sessions/${encodeURIComponent(sessionId)}/playlists`,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${credential}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ name, id }),
    },
  );
  if (!res.ok) {
    const body = (await res.json().catch(() => ({}))) as { error?: string };
    throw new Error(body.error ?? `save_failed_${res.status}`);
  }
  const data = (await res.json()) as {
    playlist: { id: string; name: string };
  };
  return data.playlist;
}

export async function loadSavedPlaylist(
  sessionId: string,
  credential: string,
  playlistId: string,
): Promise<{ skipped: number }> {
  const res = await fetch(
    `${hubHttpBase()}/api/v1/sessions/${encodeURIComponent(sessionId)}/playlists/${encodeURIComponent(playlistId)}/load`,
    {
      method: "POST",
      headers: { Authorization: `Bearer ${credential}` },
    },
  );
  if (!res.ok) {
    const body = (await res.json().catch(() => ({}))) as { error?: string };
    throw new Error(body.error ?? `load_failed_${res.status}`);
  }
  const data = (await res.json()) as { skipped?: number };
  return { skipped: data.skipped ?? 0 };
}

export async function deleteSavedPlaylist(id: string): Promise<void> {
  const res = await fetch(
    `${hubHttpBase()}/api/v1/playlists/${encodeURIComponent(id)}`,
    { method: "DELETE" },
  );
  if (!res.ok) {
    throw new Error(`delete_failed_${res.status}`);
  }
}
