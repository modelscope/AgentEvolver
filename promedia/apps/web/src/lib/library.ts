import type {
  BiblePayload,
  LibraryItem,
  LibrarySongPayload,
  PlaylistItem,
  TextPayload,
} from "@promedia/shared";
import { hubHttpBase } from "./hub";

export async function listLibraryItems(opts?: {
  category?: string;
  q?: string;
}): Promise<LibraryItem[]> {
  const q = new URLSearchParams();
  if (opts?.category) {
    q.set("category", opts.category);
  }
  if (opts?.q?.trim()) {
    q.set("q", opts.q.trim());
  }
  const qs = q.toString();
  const res = await fetch(
    `${hubHttpBase()}/api/v1/library/items${qs ? `?${qs}` : ""}`,
  );
  if (!res.ok) {
    throw new Error(`library_list_failed_${res.status}`);
  }
  const data = (await res.json()) as { items: LibraryItem[] };
  return data.items ?? [];
}

export async function createLibrarySong(
  song: LibrarySongPayload,
): Promise<LibraryItem> {
  const res = await fetch(`${hubHttpBase()}/api/v1/library/songs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(song),
  });
  const body = (await res.json().catch(() => ({}))) as {
    item?: LibraryItem;
    error?: string;
    message?: string;
  };
  if (!res.ok || !body.item) {
    throw new Error(body.message || body.error || `song_create_${res.status}`);
  }
  return body.item;
}

export async function updateLibrarySong(
  id: string,
  song: LibrarySongPayload,
): Promise<LibraryItem> {
  const res = await fetch(
    `${hubHttpBase()}/api/v1/library/songs/${encodeURIComponent(id)}`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(song),
    },
  );
  const body = (await res.json().catch(() => ({}))) as {
    item?: LibraryItem;
    error?: string;
    message?: string;
  };
  if (!res.ok || !body.item) {
    throw new Error(body.message || body.error || `song_update_${res.status}`);
  }
  return body.item;
}

export async function saveLibraryText(opts: {
  title: string;
  text: TextPayload;
}): Promise<LibraryItem> {
  const res = await fetch(`${hubHttpBase()}/api/v1/library/text`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(opts),
  });
  const body = (await res.json().catch(() => ({}))) as {
    item?: LibraryItem;
    error?: string;
    message?: string;
  };
  if (!res.ok || !body.item) {
    throw new Error(body.message || body.error || `text_save_${res.status}`);
  }
  return body.item;
}

export async function saveLibraryBible(
  payload: BiblePayload,
): Promise<LibraryItem> {
  const res = await fetch(`${hubHttpBase()}/api/v1/library/bible`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = (await res.json().catch(() => ({}))) as {
    item?: LibraryItem;
    error?: string;
    message?: string;
  };
  if (!res.ok || !body.item) {
    throw new Error(body.message || body.error || `bible_save_${res.status}`);
  }
  return body.item;
}

export async function uploadLibraryFile(
  file: File,
  title?: string,
  opts?: { category?: string },
): Promise<LibraryItem> {
  const form = new FormData();
  form.append("file", file);
  if (title?.trim()) {
    form.append("title", title.trim());
  }
  if (opts?.category?.trim()) {
    form.append("category", opts.category.trim());
  }
  const res = await fetch(`${hubHttpBase()}/api/v1/library/upload`, {
    method: "POST",
    body: form,
  });
  const body = (await res.json().catch(() => ({}))) as {
    item?: LibraryItem;
    error?: string;
    message?: string;
  };
  if (!res.ok || !body.item) {
    throw new Error(
      body.message || body.error || `library_upload_${res.status}`,
    );
  }
  return body.item;
}

export type BibleFolderImportResult = {
  folder: string;
  importedCount: number;
  skippedCount: number;
  imported: Array<{
    id: string;
    title: string;
    versionId?: string;
    shortLabel?: string;
    verseCount?: number;
    bookCount?: number;
  }>;
  skipped: Array<{ file: string; error: string; message: string }>;
};

/** Import Bibles from hub data/library/bibles/import/ (operator local files). */
export async function importBiblesFromFolder(): Promise<BibleFolderImportResult> {
  const res = await fetch(
    `${hubHttpBase()}/api/v1/library/bibles/import-folder`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    },
  );
  const body = (await res.json().catch(() => ({}))) as BibleFolderImportResult & {
    error?: string;
    message?: string;
  };
  if (!res.ok) {
    throw new Error(
      body.message || body.error || `bible_import_folder_${res.status}`,
    );
  }
  return body;
}

export type SongImportResult = {
  folder?: string;
  importedCount: number;
  updatedCount: number;
  skippedCount: number;
  imported: Array<{
    id: string;
    title: string;
    artist?: string;
    stanzaCount?: number;
  }>;
  updated: Array<{
    id: string;
    title: string;
    artist?: string;
    stanzaCount?: number;
  }>;
  skipped: Array<{ file: string; error: string; message: string }>;
};

/** Multipart upload of a Quelea .qsp pack or single song .xml. */
export async function importQueleaSongsFile(
  file: File,
): Promise<SongImportResult> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${hubHttpBase()}/api/v1/library/songs/import`, {
    method: "POST",
    body: form,
  });
  const body = (await res.json().catch(() => ({}))) as SongImportResult & {
    error?: string;
    message?: string;
  };
  if (!res.ok) {
    throw new Error(
      body.message || body.error || `song_import_${res.status}`,
    );
  }
  return body;
}

/** Import Quelea packs/XML from hub data/library/songs/import/. */
export async function importSongsFromFolder(): Promise<SongImportResult> {
  const res = await fetch(
    `${hubHttpBase()}/api/v1/library/songs/import-folder`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    },
  );
  const body = (await res.json().catch(() => ({}))) as SongImportResult & {
    error?: string;
    message?: string;
  };
  if (!res.ok) {
    throw new Error(
      body.message || body.error || `song_import_folder_${res.status}`,
    );
  }
  return body;
}

export async function deleteLibraryItem(id: string): Promise<void> {
  const res = await fetch(
    `${hubHttpBase()}/api/v1/library/items/${encodeURIComponent(id)}`,
    { method: "DELETE" },
  );
  if (!res.ok) {
    const body = (await res.json().catch(() => ({}))) as {
      error?: string;
      message?: string;
    };
    throw new Error(body.message || body.error || `delete_${res.status}`);
  }
}

export async function addLibraryItemToSession(opts: {
  sessionId: string;
  credential: string;
  libraryItemId: string;
}): Promise<PlaylistItem> {
  const res = await fetch(
    `${hubHttpBase()}/api/v1/sessions/${encodeURIComponent(opts.sessionId)}/library/${encodeURIComponent(opts.libraryItemId)}/add`,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${opts.credential}`,
      },
    },
  );
  const body = (await res.json().catch(() => ({}))) as {
    item?: PlaylistItem;
    error?: string;
    message?: string;
  };
  if (!res.ok || !body.item) {
    throw new Error(body.message || body.error || `add_failed_${res.status}`);
  }
  return body.item;
}

export function libraryOriginalUrl(itemId: string): string {
  return `${hubHttpBase()}/api/v1/library/items/${encodeURIComponent(itemId)}/original`;
}

export const LIBRARY_UPLOAD_ACCEPT =
  ".png,.jpg,.jpeg,.gif,.webp,.svg,.mp4,.webm,.mov,.mp3,.m4a,.aac,.wav,.flac,.ogg,.oga,.opus,.txt,.md,.xml,.xmm,.qsp,.pdf,.pptx,.ppt,.ppsx,.potx,.odp,image/*,video/*,audio/*,application/pdf,application/vnd.openxmlformats-officedocument.presentationml.presentation,application/vnd.openxmlformats-officedocument.presentationml.slideshow,application/vnd.oasis.opendocument.presentation";

export const LIBRARY_BIBLE_UPLOAD_ACCEPT = ".xml,.xmm,.pdf,application/pdf,text/xml,application/xml";

export const LIBRARY_SONG_UPLOAD_ACCEPT =
  ".qsp,.xml,application/zip,application/xml,text/xml";
