@echo off
setlocal
cd /d "%~dp0\..\.."
python -m gemmanima.cli gui-command
if errorlevel 1 (
  echo.
  echo [GemmAnima] GUI command failed.
  pause
  exit /b 1
)
