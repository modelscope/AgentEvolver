@echo off
REM ProMedia — atajo de escritorio (Windows).
REM Si una copia en el Escritorio abre la UI vieja, volve a copiar
REM ESTE archivo desde promedia\ despues de git pull.
REM Igual que start-windows.bat: arranca hub + web y abre Control.
REM Podes copiar este archivo al Escritorio.

cd /d "%~dp0"
call "%~dp0start-windows.bat"
