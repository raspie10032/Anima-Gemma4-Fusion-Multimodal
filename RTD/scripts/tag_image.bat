@echo off
setlocal
cd /d "%~dp0\..\.."

if "%~1"=="" (
  echo Usage: RTD\scripts\tag_image.bat path\to\image.png
  exit /b 2
)

python -m gemmanima.cli tag-image "%~1" --json
if errorlevel 1 (
  echo.
  echo [GemmAnima] Tag-image command failed.
  pause
  exit /b 1
)
