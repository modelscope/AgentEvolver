#!/usr/bin/env bash
# ProMedia — doble clic / terminal (Linux) para arrancar hub + web
# y abrir Control en el navegador predeterminado.
# Mantiene la ventana abierta si algo falla.
#
# Uso:
#   ./start-linux.sh
#   o doble clic en Abrir-ProMedia.sh / ProMedia.desktop
# Primera vez: chmod +x start-linux.sh Abrir-ProMedia.sh
#
# Si un atajo del Escritorio abre UI vieja: debe apuntar a ESTE
# promedia/ del repo (después de git pull), no a un clone viejo.

set -e
cd "$(dirname "$0")"

READY_FILE=".promedia-ready-url"
LAN_URL_FILE=".promedia-lan-url"
LOCK_FILE=".promedia-dev.lock"
VITE_PORTS="5173 5174 5175 5176 5177 5178"

# Print phone Control URL (same Wi‑Fi). Prefer file from pnpm dev; else lan-ips.mjs.
print_lan_control_hint() {
  local port="${1:-5173}"
  local url ip
  if [ -f "$LAN_URL_FILE" ]; then
    url=$(tr -d '[:space:]' < "$LAN_URL_FILE" 2>/dev/null || true)
    if [ -n "$url" ]; then
      echo "Teléfono (misma Wi‑Fi) → ${url}"
      echo "  Output/proyección sigue en esta PC; el móvil es solo Control."
      return 0
    fi
  fi
  if command -v node >/dev/null 2>&1 && [ -f scripts/lan-ips.mjs ]; then
    ip=$(node scripts/lan-ips.mjs 2>/dev/null | head -n 1 || true)
    if [ -n "$ip" ]; then
      echo "Teléfono (misma Wi‑Fi) → http://${ip}:${port}/control"
      echo "  Output/proyección sigue en esta PC; el móvil es solo Control."
      return 0
    fi
  fi
  echo "Teléfono: abrí http://<IP-LAN-de-esta-PC>:${port}/control (misma Wi‑Fi)."
}

echo "=========================================="
echo "  ProMedia — carpeta: $(pwd)"
echo "=========================================="
echo ""
echo "AVISO: un alias del Escritorio debe apuntar a esta carpeta"
echo "  promedia/ del repo (no a un clone viejo)."
echo "Dejá esta ventana de Terminal abierta mientras usás ProMedia."
echo ""

if ! command -v pnpm >/dev/null 2>&1; then
  echo "ERROR: no se encontró pnpm."
  echo "Instalá Node.js 20+ y luego: npm install -g pnpm"
  echo ""
  read -r -p "Pulsá Enter para cerrar…"
  exit 1
fi

# Liberar hub (:8787) y Vite (5173–5178) solo si el listener parece node/promedia/tsx/vite.
# SOLO llamar ANTES de arrancar hijos — nunca desde open_control_when_ready.
free_promedia_port() {
  local port="$1"
  local pids pid cmd
  pids=""
  if command -v lsof >/dev/null 2>&1; then
    pids=$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null || true)
  elif command -v ss >/dev/null 2>&1; then
    # ss: extraer pids de listeners en el puerto
    pids=$(ss -ltnp "sport = :$port" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u || true)
  elif command -v fuser >/dev/null 2>&1; then
    pids=$(fuser "${port}/tcp" 2>/dev/null || true)
  fi
  if [ -z "$pids" ]; then
    return 0
  fi
  for pid in $pids; do
    cmd=$(ps -p "$pid" -o command= 2>/dev/null || true)
    case "$cmd" in
      *promedia*|*tsx*|*vite*|*@promedia/hub*|*concurrently*)
        echo "Cerrando instancia anterior en puerto ${port}… (PID ${pid})"
        kill "$pid" 2>/dev/null || true
        ;;
      *)
        echo "AVISO: el puerto ${port} está ocupado por otro proceso (PID ${pid})."
        echo "  $cmd"
        echo "  Si es un ProMedia viejo: kill ${pid}"
        ;;
    esac
  done
}

# Espera a que Control responda (nunca abre bare "/"). NUNCA libera puertos.
open_control_when_ready() {
  local max_wait="${PROMEDIA_OPEN_WAIT_S:-90}"
  local elapsed=0
  local port url
  local opened_marker=".promedia-browser-opened"
  rm -f "$opened_marker"
  echo "Cuando hub+web estén listos, se abrirá Control en el navegador…"
  while [ "$elapsed" -lt "$max_wait" ]; do
    if [ -f "$opened_marker" ]; then
      return 0
    fi
    if [ -f "$READY_FILE" ]; then
      url=$(tr -d '[:space:]' < "$READY_FILE" 2>/dev/null || true)
      if [ -n "$url" ]; then
        ready=0
        if command -v curl >/dev/null 2>&1; then
          if curl -sf -o /dev/null --connect-timeout 1 "$url" 2>/dev/null; then
            ready=1
          fi
        else
          ready=1
        fi
        if [ "$ready" -eq 1 ]; then
          echo ""
          echo "Abriendo Control → ${url}"
          port_hint=$(printf '%s' "$url" | sed -n 's/.*:\([0-9][0-9]*\)\/control.*/\1/p')
          print_lan_control_hint "${port_hint:-5173}"
          echo "Dejá esta Terminal abierta. Para detener: Ctrl+C."
          touch "$opened_marker"
          if command -v xdg-open >/dev/null 2>&1; then
            xdg-open "$url" >/dev/null 2>&1 || true
          elif command -v gio >/dev/null 2>&1; then
            gio open "$url" >/dev/null 2>&1 || true
          else
            echo "Abrí manualmente: ${url}"
          fi
          return 0
        fi
      fi
    fi
    for port in $VITE_PORTS; do
      if command -v curl >/dev/null 2>&1; then
        if curl -sf -o /dev/null --connect-timeout 1 "http://127.0.0.1:${port}/control" 2>/dev/null; then
          url="http://localhost:${port}/control"
          echo ""
          echo "Abriendo Control → ${url}"
          print_lan_control_hint "$port"
          echo "Dejá esta Terminal abierta. Para detener: Ctrl+C."
          touch "$opened_marker"
          if command -v xdg-open >/dev/null 2>&1; then
            xdg-open "$url" >/dev/null 2>&1 || true
          elif command -v gio >/dev/null 2>&1; then
            gio open "$url" >/dev/null 2>&1 || true
          else
            echo "Abrí manualmente: ${url}"
          fi
          return 0
        fi
      else
        # Sin curl: probar con bash /dev/tcp (solo puerto; aún abrimos /control).
        if (echo >/dev/tcp/127.0.0.1/"$port") >/dev/null 2>&1; then
          url="http://localhost:${port}/control"
          echo ""
          echo "Abriendo Control → ${url}"
          print_lan_control_hint "$port"
          echo "Dejá esta Terminal abierta. Para detener: Ctrl+C."
          touch "$opened_marker"
          if command -v xdg-open >/dev/null 2>&1; then
            xdg-open "$url" >/dev/null 2>&1 || true
          else
            echo "Abrí manualmente: ${url}"
          fi
          return 0
        fi
      fi
    done
    sleep 1
    elapsed=$((elapsed + 1))
  done
  echo ""
  echo "AVISO: no se detectó Control a tiempo."
  echo "Mirás en los logs la línea [dev] ready — Control http://… y abrila."
  echo "También podés abrir open-control.html (prueba 5173–5178) o ProMedia-Control.url."
  return 0
}

open_existing_only() {
  echo "ProMedia ya está en marcha${1:+ (PID $1)}."
  echo "No se reinicia ni se liberan puertos (evita matar Vite con SIGTERM 143)."
  echo "Abriendo Control en el navegador…"
  open_control_when_ready
  echo ""
  read -r -p "Pulsá Enter para cerrar esta ventana (la otra Terminal sigue con ProMedia)…"
  exit 0
}

# Segunda instancia: NO liberar puertos (evita SIGTERM 143 en Vite).
if [ -f "$LOCK_FILE" ]; then
  old_pid=$(tr -d '[:space:]' < "$LOCK_FILE" 2>/dev/null || true)
  if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
    open_existing_only "$old_pid"
  fi
  rm -f "$LOCK_FILE"
fi

rm -f "$READY_FILE" ".promedia-browser-opened"

# Atomic claim BEFORE freeing ports so a second start cannot also free/kill.
if ! (set -C; echo $$ > "$LOCK_FILE") 2>/dev/null; then
  old_pid=$(tr -d '[:space:]' < "$LOCK_FILE" 2>/dev/null || true)
  open_existing_only "${old_pid:-}"
fi

free_promedia_port 8787
for port in $VITE_PORTS; do
  free_promedia_port "$port"
done
sleep 1 2>/dev/null || sleep 1

if [ ! -d node_modules ] || [ ! -f node_modules/.modules.yaml ]; then
  echo "Instalando dependencias (pnpm install)…"
  pnpm install || {
    rm -f "$LOCK_FILE"
    echo ""
    echo "ERROR: pnpm install falló."
    read -r -p "Pulsá Enter para cerrar…"
    exit 1
  }
  echo ""
fi

OPENER_PID=""

cleanup_opener() {
  if [ -n "${OPENER_PID:-}" ] && kill -0 "$OPENER_PID" 2>/dev/null; then
    kill "$OPENER_PID" 2>/dev/null || true
    wait "$OPENER_PID" 2>/dev/null || true
  fi
  OPENER_PID=""
}

cleanup_all() {
  cleanup_opener
  rm -f "$LOCK_FILE" ".promedia-browser-opened"
}
trap cleanup_all EXIT INT TERM

open_control_when_ready &
OPENER_PID=$!

echo "Arrancando hub + web (pnpm dev)…"
echo "Esperá la línea [dev] ready — ahí está la URL REAL de Control (puede no ser :5173)."
echo "Hub health: http://localhost:8787/health"
echo "Una sola instancia a la vez. Para detener: Ctrl+C en esta ventana."
echo ""

pnpm dev || {
  cleanup_all
  echo ""
  echo "ERROR: pnpm dev terminó con fallo."
  echo "¿Estás en la carpeta correcta? Debe ser …/AgentEvolver/promedia"
  echo "Ruta actual: $(pwd)"
  echo ""
  echo "Si ves EADDRINUSE / puerto en uso:"
  echo "  ss -ltnp 'sport = :8787'   # o: lsof -nP -iTCP:8787 -sTCP:LISTEN"
  echo "  ss -ltnp 'sport = :5173'"
  echo "  kill <PID>"
  echo "Luego volvé a abrir start-linux.sh (o Abrir-ProMedia.sh)."
  echo ""
  echo "Workaround (dos Terminales):"
  echo "  Terminal A: cd …/promedia && pnpm dev:hub"
  echo "  Terminal B: cd …/promedia && pnpm dev:web"
  echo "  Luego abrí http://localhost:5173/control"
  read -r -p "Pulsá Enter para cerrar…"
  exit 1
}

cleanup_all
read -r -p "ProMedia se detuvo. Pulsá Enter para cerrar…"
