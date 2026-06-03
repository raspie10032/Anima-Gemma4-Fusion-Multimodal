@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "VENV_DIR=.venv"
set "PY=%VENV_DIR%\Scripts\python.exe"

if "%~1"=="" goto gui
if /I "%~1"=="bootstrap" goto bootstrap
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

:bootstrap
call :bootstrap_env
goto end

:gui
call :ensure_env
if errorlevel 1 goto failed
"%PY%" -m gemmanima.cli gui-command
goto end

:health
call :ensure_env
if errorlevel 1 goto failed
"%PY%" --version
if errorlevel 1 goto failed
"%PY%" -m gemmanima.core.dependencies
if errorlevel 1 goto failed
"%PY%" -m gemmanima.cli dependency-audit --json
if errorlevel 1 goto failed
"%PY%" -m gemmanima.cli model-download-plan --json
if errorlevel 1 goto failed
"%PY%" -m gemmanima.cli renderer-backends --json
if errorlevel 1 goto failed
"%PY%" -m gemmanima.cli real-render-health --json
if errorlevel 1 goto failed
"%PY%" -m gemmanima.cli gui-command --json
goto end

:download
call :ensure_env
if errorlevel 1 goto failed
"%PY%" -m gemmanima.cli ensure-model-assets --json
goto end

:dryrun
call :ensure_env
if errorlevel 1 goto failed
set "PROMPT=%~2"
if "%PROMPT%"=="" set "PROMPT=draw a bright forest anime illustration"
"%PY%" -m gemmanima.cli run "%PROMPT%" --renderer dry-run --json
goto end

:tag
call :ensure_env
if errorlevel 1 goto failed
if "%~2"=="" (
    echo Usage: GemmAnima.bat tag path\to\image.png
    exit /b 2
)
"%PY%" -m gemmanima.cli tag-image "%~2" --json
goto end

:test
call :ensure_env
if errorlevel 1 goto failed
"%PY%" -m pytest -q
goto end

:help
echo GemmAnima launcher
echo.
echo Usage:
echo   GemmAnima.bat bootstrap               Create/update the source checkout .venv
echo   GemmAnima.bat                         Start the local GUI backend
echo   GemmAnima.bat gui                     Start the local GUI backend
echo   GemmAnima.bat health                  Print runtime/model health
echo   GemmAnima.bat download                Download or verify model assets
echo   GemmAnima.bat dry-run "prompt"        Run a dry-run image request
echo   GemmAnima.bat tag path\to\image.png   Tag one image
echo   GemmAnima.bat test                    Run the test suite
exit /b 0

:ensure_env
if exist "%PY%" exit /b 0
echo [GemmAnima] Source checkout .venv was not found.
echo [GemmAnima] Running visible first-run bootstrap now.
call :bootstrap_env
exit /b %errorlevel%

:bootstrap_env
python --version >nul 2>nul
if errorlevel 1 (
    echo [GemmAnima] Python 3.10+ was not found on PATH.
    echo [GemmAnima] Install Python first, then run GemmAnima.bat bootstrap again.
    exit /b 1
)
if not exist "%PY%" (
    echo [GemmAnima] Creating %VENV_DIR% with system-site-packages.
    python -m venv --system-site-packages "%VENV_DIR%"
    if errorlevel 1 exit /b 1
)
echo [GemmAnima] Updating source checkout package in %VENV_DIR%.
"%PY%" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
"%PY%" -m pip install -e .
if errorlevel 1 exit /b 1
"%PY%" -m pip install pytest
if errorlevel 1 exit /b 1
echo [GemmAnima] Dependency audit:
"%PY%" -m gemmanima.core.dependencies
exit /b %errorlevel%

:end
if errorlevel 1 goto failed
exit /b 0

:failed
echo.
echo [GemmAnima] Command failed.
pause
exit /b 1
