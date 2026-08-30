@echo off
setlocal

set "ROOT=%~dp0"
set "PYTHON=%ROOT%venv\Scripts\python.exe"
set "FRONTEND=%ROOT%frontend"
set "WECHAT_URL=https://cloud1-d2gl1lav2eb6e440e-1477389215.ap-shanghai.app.tcloudbase.com/fallAlarmPush"

if not exist "%PYTHON%" (
  echo [ERROR] Python virtual environment not found: %PYTHON%
  pause
  exit /b 1
)

if not exist "%FRONTEND%\package.json" (
  echo [ERROR] Frontend directory not found: %FRONTEND%
  pause
  exit /b 1
)

if not exist "%FRONTEND%\node_modules" (
  echo [INFO] Installing frontend dependencies...
  pushd "%FRONTEND%"
  call npm.cmd install
  if errorlevel 1 (
    popd
    echo [ERROR] npm install failed.
    pause
    exit /b 1
  )
  popd
)

echo Checking service ports...
netstat -ano | findstr /R /C:":8000 .*LISTENING" >nul
if errorlevel 1 (
echo Starting backend...
  start "Fall Risk Backend" /D "%ROOT%" cmd /k "set WECHAT_FALL_ALARM_PUSH_ENABLED=1&& set WECHAT_FALL_ALARM_PUSH_URL=%WECHAT_URL%&& set WECHAT_FALL_ALARM_PUSH_PAYLOAD_MODE=hybrid&& venv\Scripts\python.exe -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000"
) else (
  echo Backend port 8000 is already in use; keeping the existing service.
)

netstat -ano | findstr /R /C:":5173 .*LISTENING" >nul
if errorlevel 1 (
  echo Starting frontend...
  start "Fall Risk Frontend" /D "%FRONTEND%" cmd /k "npm.cmd run dev -- --host 0.0.0.0 --port 5173"
) else (
  echo Frontend port 5173 is already in use; keeping the existing service.
)

echo.
echo Backend:  http://127.0.0.1:8000
echo Frontend: http://127.0.0.1:5173
echo WeChat alerts: HTTPS CloudBase push endpoint is enabled.
echo WeChat endpoint: %WECHAT_URL%
echo Close this window; the two service windows will remain open.
endlocal
