/**
 * Static YouTube poster — used on Control Live (and Preview while projecting)
 * so Output owns the only IFrame API embed for that videoId.
 */
export type YouTubeThumbProps = {
  videoId: string;
  title?: string;
  className?: string;
  caption?: string;
  /** When hub clock is playing, show a clear “on Output” affordance. */
  playing?: boolean;
  playingLabel?: string;
};

export function youtubeThumbUrl(videoId: string): string {
  return `https://i.ytimg.com/vi/${videoId}/hqdefault.jpg`;
}

export function YouTubeThumb({
  videoId,
  title,
  className,
  caption,
  playing = false,
  playingLabel = "Reproduciendo en Output",
}: YouTubeThumbProps) {
  return (
    <div
      className={className ?? "video-yt-thumb-wrap"}
      data-playing={playing ? "1" : undefined}
    >
      <img
        className="video-yt-thumb"
        src={youtubeThumbUrl(videoId)}
        alt={title?.trim() || `YouTube ${videoId}`}
        loading="lazy"
        decoding="async"
      />
      {playing ? (
        <div className="video-yt-playing-badge" aria-live="polite">
          <span className="video-yt-playing-pulse" aria-hidden="true" />
          <span>{playingLabel}</span>
        </div>
      ) : null}
      {caption ? <p className="video-caveat">{caption}</p> : null}
    </div>
  );
}
