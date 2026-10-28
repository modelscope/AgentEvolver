# ProMedia — carpeta portable

Empaquetado runnable (hub + web) para PCs sin instalar un servidor de hosting.
Para el flujo completo (Control, **Proyectar**, Guion, Output), ver el README principal en `promedia/README.md`.

## Empaquetar (MVP)

Desde `promedia/`:

```bash
pnpm build
./portable/pack.sh
```

Esto deja en `portable/dist/`:

```text
portable/dist/
  hub/          # apps/hub/dist + package hints
  web/          # apps/web/dist (estáticos)
  data/         # media + playlists (vacío)
  start.sh      # arranca hub sirviendo API; abre nota para servir web
  README.md
```

## Arranque

```bash
cd portable/dist
./start.sh
```

Por defecto:

- Hub: `http://localhost:8787` (`PROMEDIA_MODE=portable`)
- Web: servir `web/` con cualquier static server, p. ej.  
  `npx --yes serve web -l 5173`  
  Luego abrí **`http://localhost:5173/control`** (no la Home `/`).

LAN sigue siendo el camino preferido con `pnpm dev` / `Abrir-ProMedia.*` en el monorepo.
