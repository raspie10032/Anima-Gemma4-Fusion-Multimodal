@echo off
setlocal
cd /d "%~dp0\..\.."

echo [RTD] Repo: %CD%
echo [RTD] Python:
python --version
if errorlevel 1 goto failed

echo.
echo [RTD] Core tests:
python -m pytest tests/test_tipo_runtime.py tests/test_api.py tests/test_server_gui.py -q
if errorlevel 1 goto failed

echo.
echo [RTD] Renderer backends:
python -m gemmanima.cli renderer-backends --json
if errorlevel 1 goto failed

echo.
echo [RTD] Real render health:
python -m gemmanima.cli real-render-health --json
if errorlevel 1 goto failed

echo.
echo [RTD] GUI command:
python -m gemmanima.cli gui-command --json
if errorlevel 1 goto failed

exit /b 0

:failed
echo.
echo [GemmAnima] Health check failed.
pause
exit /b 1
