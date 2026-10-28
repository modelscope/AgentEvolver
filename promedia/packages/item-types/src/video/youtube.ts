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
