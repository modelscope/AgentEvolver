#!/usr/bin/env bash
# ProMedia — atajo de escritorio (Linux).
# Si una copia en el Escritorio abre la UI vieja, volvé a copiar
# ESTE archivo desde promedia/ después de `git pull`.
# Igual que start-linux.sh: arranca hub + web y abre Control.
# Podés copiar este archivo al Escritorio o enlazarlo desde el Dock/panel.
# Primera vez: chmod +x Abrir-ProMedia.sh start-linux.sh

cd "$(dirname "$0")"
exec bash "./start-linux.sh"
