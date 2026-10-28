import { Link } from "react-router-dom";
import { DEFAULT_LOCALE, t } from "@promedia/i18n";

export function HomePage() {
  const locale = DEFAULT_LOCALE;
  return (
    <main className="scaffold-page">
      <h1>{t(locale, "app.name")}</h1>
      <p>{t(locale, "home.tagline")}</p>
      <nav className="scaffold-nav">
        <Link to="/control">/control</Link>
        <Link to="/output">/output</Link>
      </nav>
    </main>
  );
}
