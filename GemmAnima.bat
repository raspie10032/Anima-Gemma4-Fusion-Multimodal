@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if "%~1"=="" goto gui
if /I "%~1"=="gui" goto gui
if /I "%~1"=="health" goto health
if /I "%~1"=="download" goto download
if /I "%~1"=="dry-run" goto dryrun
if /I "%~1"=="tag" goto tag
if /I "%~1"=="test" goto test
if /I "%~1"=="help" goto help
if /I "%~1"=="--help" goto help
if /I "%~1"=="/?" goto help

echo Unknown command: %~1
goto help

:gui
python -m gemmanima.cli gui-command
goto end

:health
python --version
if errorlevel 1 goto failed
python -m gemmanima.cli model-download-plan --json
if errorlevel 1 goto failed
python -m gemmanima.cli renderer-backends --json
if errorlevel 1 goto failed
python -m gemmanima.cli real-render-health --json
if errorlevel 1 goto failed
python -m gemmanima.cli gui-command --json
goto end

:download
python -m gemmanima.cli ensure-model-assets --json
goto end

:dryrun
set "PROMPT=%~2"
if "%PROMPT%"=="" set "PROMPT=draw a bright forest anime illustration"
python -m gemmanima.cli run "%PROMPT%" --renderer dry-run --json
goto end

:tag
if "%~2"=="" (
    echo Usage: GemmAnima.bat tag path\to\image.png
    exit /b 2
)
python -m gemmanima.cli tag-image "%~2" --json
goto end

:test
python -m pytest -q
goto end

:help
echo GemmAnima launcher
echo.
echo Usage:
echo   GemmAnima.bat                         Start the local GUI backend
echo   GemmAnima.bat gui                     Start the local GUI backend
echo   GemmAnima.bat health                  Print runtime/model health
echo   GemmAnima.bat download                Download or verify model assets
echo   GemmAnima.bat dry-run "prompt"        Run a dry-run image request
echo   GemmAnima.bat tag path\to\image.png   Tag one image
echo   GemmAnima.bat test                    Run the test suite
exit /b 0

:end
if errorlevel 1 goto failed
exit /b 0

:failed
echo.
echo [GemmAnima] Command failed.
pause
exit /b 1
