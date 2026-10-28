@echo off
REM ProMedia — doble clic en el Explorador (Windows) para arrancar hub + web
REM y abrir Control (/control, puerto real Vite) en el navegador.
REM Delega en start-windows.ps1 (detección de puertos más fiable).
REM Ver también: Abrir-ProMedia.bat (mismo arranque, nombre claro para el Escritorio).
REM Si el atajo del Escritorio abre UI vieja: re-copia desde este promedia\ tras git pull.

cd /d "%~dp0"

where pnpm >nul 2>&1
if errorlevel 1 (
  echo ERROR: no se encontro pnpm.
  echo Instala Node.js 20+ y luego: npm install -g pnpm
  echo.
  pause
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-windows.ps1"
set EXITCODE=%ERRORLEVEL%
if not "%EXITCODE%"=="0" (
  echo.
  echo Si PowerShell bloqueo el script, ejecuta una vez:
  echo   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
  echo O usa siempre este .bat ^(ya pasa -ExecutionPolicy Bypass^).
  pause
)
exit /b %EXITCODE%
