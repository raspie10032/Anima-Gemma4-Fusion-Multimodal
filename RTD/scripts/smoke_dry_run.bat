@echo off
setlocal
cd /d "%~dp0\..\.."

set "PROMPT=%*"
if "%PROMPT%"=="" set "PROMPT=draw a bright forest anime illustration"

python -m gemmanima.cli run "%PROMPT%" --renderer dry-run --json
if errorlevel 1 (
  echo.
  echo [GemmAnima] Dry-run smoke failed.
  pause
  exit /b 1
)
