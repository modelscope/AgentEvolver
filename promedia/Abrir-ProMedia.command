#!/bin/bash
# ProMedia — atajo de escritorio (Mac).
# Si una copia en el Escritorio abre la UI vieja (Home en vez de Control),
# volvé a copiar ESTE archivo desde promedia/ después de `git pull`.
# Igual que start-mac.command: arranca hub + web y abre Control.
# Podés copiar este archivo al Escritorio o al Dock (alias).

cd "$(dirname "$0")"
exec bash "./start-mac.command"
