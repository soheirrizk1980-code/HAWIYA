@echo off
setlocal
cd /d "%~dp0"
set "HAWIYA_AI_ROOT=%~dp0HAWIYA_AI"
set "PYTHON_CMD=python"
py -3.11 -V >nul 2>&1
if not errorlevel 1 (set "PYTHON_CMD=py -3.11")
if not exist ".venv\Scripts\python.exe" (
  echo Creating separate Windows environment...
  %PYTHON_CMD% -m venv .venv
  if errorlevel 1 (echo Could not create environment. Install Python 3.11 and try again.& pause & exit /b 1)
)
if not exist ".venv\.hawiya_installed" (
  echo Installing required packages for the NEW project only...
  .venv\Scripts\python.exe -m pip install -U pip
  .venv\Scripts\python.exe -m pip install -r requirements.txt
  if errorlevel 1 (echo Package install failed. Check the messages above.&pause&exit /b 1)
  echo installed>".venv\.hawiya_installed"
)
echo.
echo Checking the blue-eye reference...
.venv\Scripts\python.exe -u server.py --ensure-blue
if errorlevel 1 (
  echo.
  echo ERROR: Could not prepare the blue reference. The original model files and sample photos are required.
  echo Check the messages above, then retry.
  pause
  exit /b 1
)
echo.
echo HAWIYA REAL AI LAB starts at http://127.0.0.1:8800
echo Keep this window open. First model inference can be slower on CPU.
start "" "http://127.0.0.1:8800"
.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8800
pause

