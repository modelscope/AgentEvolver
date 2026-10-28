import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { randomBytes } from "node:crypto";
import type { PlaylistItem, SavedPlaylist } from "@promedia/shared";
import { validatePlaylistItem } from "@promedia/item-types/validate";

function nowIso(): string {
  return new Date().toISOString();
}

function newId(): string {
  return randomBytes(10).toString("hex");
}

function dataPath(): string {
  return (
    process.env.PLAYLISTS_PATH ??
    path.join(process.cwd(), "data", "playlists.json")
  );
}

/**
 * File-backed SavedPlaylist registry (MVP; SQLite-ready shape).
 */
export class PlaylistStore {
  private items = new Map<string, SavedPlaylist>();
  private loaded = false;

  private async ensureLoaded(): Promise<void> {
    if (this.loaded) {
      return;
    }
    this.loaded = true;
    try {
      const raw = await readFile(dataPath(), "utf8");
      const parsed = JSON.parse(raw) as { playlists?: SavedPlaylist[] };
      for (const p of parsed.playlists ?? []) {
        if (p?.id && p.name) {
          this.items.set(p.id, p);
        }
      }
    } catch {
      // missing file = empty
    }
  }

  private async persist(): Promise<void> {
    const file = dataPath();
    await mkdir(path.dirname(file), { recursive: true });
    const playlists = [...this.items.values()].sort((a, b) =>
      a.name.localeCompare(b.name),
    );
    await writeFile(file, JSON.stringify({ playlists }, null, 2), "utf8");
  }

  async list(): Promise<SavedPlaylist[]> {
    await this.ensureLoaded();
    return [...this.items.values()].sort((a, b) =>
      a.updatedAt < b.updatedAt ? 1 : -1,
    );
  }

  async get(id: string): Promise<SavedPlaylist | undefined> {
    await this.ensureLoaded();
    return this.items.get(id);
  }

  async save(opts: {
    name: string;
    items: unknown[];
    id?: string;
  }): Promise<SavedPlaylist | { error: string }> {
    await this.ensureLoaded();
    const name = opts.name.trim();
    if (!name) {
      return { error: "empty_name" };
    }
    const validated: PlaylistItem[] = [];
    for (const raw of opts.items) {
      const item = validatePlaylistItem(raw);
      if (!item) {
        return { error: "invalid_item" };
      }
      validated.push(item);
    }
    const existing = opts.id ? this.items.get(opts.id) : undefined;
    const id = existing?.id ?? newId();
    const record: SavedPlaylist = {
      id,
      name,
      createdAt: existing?.createdAt ?? nowIso(),
      updatedAt: nowIso(),
      items: validated.map((item, order) => ({ ...item, order })),
    };
    this.items.set(id, record);
    await this.persist();
    return record;
  }

  async remove(id: string): Promise<boolean> {
    await this.ensureLoaded();
    const ok = this.items.delete(id);
    if (ok) {
      await this.persist();
    }
    return ok;
  }
}

export const playlistStore = new PlaylistStore();
