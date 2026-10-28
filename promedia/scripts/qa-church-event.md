# ProMedia — Checklist QA iglesia / evento (MVP)

**Objetivo:** ensayo de 15–20 min antes de culto o evento.  
**Prerrequisito:** `cd promedia && pnpm build && pnpm dev` (web :5173, hub :8787).

## A. Arranque

- [ ] Hub responde `GET http://localhost:8787/health` → `ok: true`
- [ ] Control abre en http://localhost:5173/control
- [ ] **Crear sesión** → código visible + enlace opaco
- [ ] **Abrir salida** (o 2.ª pestaña `/output?code=…`) conecta WS

## B. Contenido culto / evento

- [ ] **+ Texto** — 2 slides → Preview local → Go Live → Output muestra slide 1
- [ ] Flechas / botones ← → cambian slide en Live y Output (2.ª pestaña igual)
- [ ] **+ Imagen** — jpeg/png → Preview fit → Go Live
- [ ] **+ Vídeo** archivo mp4/webm → Go Live autoplay (unmuted; clic si el browser bloquea)
- [ ] Play / Pause / scrub en Live → Output sigue (`mediaClock`)
- [ ] **+ Vídeo** URL YouTube pública → Preview embed → Go Live (necesita red)

## C. Escenas y temas

- [ ] Logo / Negro / Clear / Normal — Output refleja; vídeo **no** se pausa solo
- [ ] Go Live de otro ítem → sceneMode vuelve a **normal**
- [ ] Tema Oscuro / Claro / Acento — texto cambia color/fondo/align
- [ ] Editor: cambiar color texto → Output actualiza; “Volver al preset” limpia overrides

## D. Playlists guardadas

- [ ] Guardar playlist con nombre (ideal: solo texto + YouTube)
- [ ] Nueva sesión (o vaciar) → **Cargar** → ítems reaparecen
- [ ] Si había imágenes/vídeos de archivo de otra sesión → aviso de omitidos OK

## E. Dual salida / reconexión

- [ ] 2 Outputs (2 pestañas o PC + TV browser) ven el mismo live / tema / escena
- [ ] Recargar Output → resync snapshot (mismo estado)

## F. Idiomas (smoke rápido)

- [ ] Cadenas visibles en ES (default). Stubs EN/PT/FR/ZH en paquete i18n presentes.

## G. Atajos

- [ ] `G` Go Live · `O` Abrir salida · `1/2/3` temas · `L/B/C/N` escenas (sin estar en un input)

## H. Automatizado

```bash
pnpm --filter @promedia/hub start &
node scripts/smoke-hub-ws.mjs
# esperas SMOKE OK
```

## Resultado

- Ensayo **OK para culto/evento** si A–E + H pasan.
- Anotar fallos de red/YouTube/codecs como caveats, no como bloqueo de alcance MVP.
