#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/portable/dist"
rm -rf "$OUT"
mkdir -p "$OUT/hub" "$OUT/web" "$OUT/data/media"

cp -R "$ROOT/apps/hub/dist/." "$OUT/hub/"
cp -R "$ROOT/apps/web/dist/." "$OUT/web/"

# Minimal runtime package for hub (workspace deps resolved from monorepo node_modules when run in-tree).
cat > "$OUT/start.sh" << 'EOF'
#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
export PORT="${PORT:-8787}"
export PROMEDIA_MODE="${PROMEDIA_MODE:-portable}"
export MEDIA_ROOT="${MEDIA_ROOT:-$DIR/data/media}"
export PLAYLISTS_PATH="${PLAYLISTS_PATH:-$DIR/data/playlists.json}"
echo "ProMedia portable hub → http://localhost:$PORT"
echo "Serve UI: npx --yes serve \"$DIR/web\" -l 5173"
echo "Then open http://localhost:5173/control"
# Prefer monorepo node if packing from source tree:
if command -v node >/dev/null; then
  NODE_PATH="${NODE_PATH:-}" node "$DIR/hub/index.js"
else
  echo "node not found"
  exit 1
fi
EOF
chmod +x "$OUT/start.sh"

cp "$ROOT/portable/README.md" "$OUT/README.md"
echo "Packed → $OUT"
