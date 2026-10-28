import { useCallback, useEffect, useState } from "react";
import {
  buildLanControlUrls,
  fetchHubHealth,
  isLoopbackPageHost,
} from "../lib/hub";
import { useLocale } from "../locale/LocaleContext";

/**
 * Surfaces the LAN Control URL for phones (footer on the projection PC)
 * or a short “remote LAN” badge when Control is already open on a phone.
 */
export function LanRemoteHint() {
  const { t } = useLocale();
  const [urls, setUrls] = useState<string[]>([]);
  const [copied, setCopied] = useState(false);
  const onPhone = !isLoopbackPageHost();

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      const health = await fetchHubHealth();
      if (cancelled || !health?.ok) return;
      const addrs = health.lanAddresses ?? [];
      if (addrs.length === 0) return;
      setUrls(buildLanControlUrls(addrs));
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const primary = urls[0] ?? "";

  const onCopy = useCallback(() => {
    if (!primary) return;
    void navigator.clipboard?.writeText(primary).then(
      () => {
        setCopied(true);
        window.setTimeout(() => setCopied(false), 1600);
      },
      () => undefined,
    );
  }, [primary]);

  if (onPhone) {
    return (
      <>
        <span className="footer-sep">·</span>
        <span className="lan-remote-badge" title={window.location.href}>
          {t("control.lanRemoteActive")}
        </span>
      </>
    );
  }

  if (!primary) return null;

  return (
    <>
      <span className="footer-sep">·</span>
      <span className="lan-remote-hint">
        <span className="lan-remote-label">{t("control.lanRemoteHint")}</span>{" "}
        <a className="lan-remote-link" href={primary}>
          {primary}
        </a>
        <button
          type="button"
          className="btn btn-ghost lan-remote-copy"
          onClick={onCopy}
          title={t("control.lanRemoteCopy")}
        >
          {copied ? t("control.lanRemoteCopied") : t("control.lanRemoteCopy")}
        </button>
      </span>
    </>
  );
}
