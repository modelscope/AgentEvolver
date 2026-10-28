/**
 * Parse public YouTube URLs → videoId (11 chars).
 */
const ID_RE = /^[\w-]{11}$/;

export function isYouTubeVideoId(id: string): boolean {
  return ID_RE.test(id);
}

export function parseYouTubeUrl(input: string): { videoId: string; sourceUrl: string } | null {
  const raw = input.trim();
  if (!raw) {
    return null;
  }
  if (isYouTubeVideoId(raw)) {
    return { videoId: raw, sourceUrl: `https://www.youtube.com/watch?v=${raw}` };
  }

  let url: URL;
  try {
    url = new URL(raw);
  } catch {
    return null;
  }

  const host = url.hostname.replace(/^www\./, "");
  let videoId: string | null = null;

  if (host === "youtu.be") {
    videoId = url.pathname.split("/").filter(Boolean)[0] ?? null;
  } else if (host === "youtube.com" || host === "m.youtube.com" || host === "music.youtube.com") {
    if (url.pathname === "/watch") {
      videoId = url.searchParams.get("v");
    } else if (url.pathname.startsWith("/embed/")) {
      videoId = url.pathname.split("/")[2] ?? null;
    } else if (url.pathname.startsWith("/shorts/")) {
      videoId = url.pathname.split("/")[2] ?? null;
    } else if (url.pathname.startsWith("/live/")) {
      videoId = url.pathname.split("/")[2] ?? null;
    }
  }

  if (!videoId || !isYouTubeVideoId(videoId)) {
    return null;
  }
  return { videoId, sourceUrl: raw };
}

/**
 * Public oEmbed title for Guion labels (no API key). Returns null offline
 * / blocked / private videos — caller keeps `YouTube {id}` fallback.
 */
export async function fetchYouTubeDisplayTitle(
  sourceUrl: string,
): Promise<string | null> {
  const url = sourceUrl.trim();
  if (!url) return null;
  const endpoints = [
    `https://www.youtube.com/oembed?url=${encodeURIComponent(url)}&format=json`,
    `https://noembed.com/embed?url=${encodeURIComponent(url)}`,
  ];
  for (const endpoint of endpoints) {
    try {
      const res = await fetch(endpoint);
      if (!res.ok) continue;
      const data = (await res.json()) as { title?: unknown };
      if (typeof data.title === "string" && data.title.trim()) {
        return data.title.trim();
      }
    } catch {
      // try next endpoint
    }
  }
  return null;
}
