@echo off
setlocal
cd /d "%~dp0"
echo HAWIYA - Preparing the model-verified sample gallery
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: Python environment not found. Run START_WINDOWS.bat once.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -u "PREPARE_SAMPLES.py"
if errorlevel 1 (
  echo Preparation failed. Copy the ERROR text from this window.
  pause
  exit /b 1
)
echo Done. Restart the website to see the prepared results.
pause
