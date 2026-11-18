@echo off
title Zambia Agro-Satellite Yield Intelligence
color 0A
echo ======================================================================
echo    ZAMBIA AGRO-SATELLITE INTELLIGENCE
echo    SAR-Augmented Attention CNN-LSTM with GDD/KDD
echo    District & National Crop Yield Forecasting Platform
echo ======================================================================
echo.

:: Detect Python executable (checking local virtual environments first)
set PYTHON_CMD=python
if exist "myenv\Scripts\python.exe" (
    set PYTHON_CMD="myenv\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
    set PYTHON_CMD=".venv\Scripts\python.exe"
) else (
    where python >nul 2>nul
    if %errorlevel% neq 0 (
        if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python313\python.exe" (
            set PYTHON_CMD="C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python313\python.exe"
        ) else if exist "C:\Program Files\Python313\python.exe" (
            set PYTHON_CMD="C:\Program Files\Python313\python.exe"
        ) else (
            set PYTHON_CMD=py
        )
    )
)

echo Starting application server with %PYTHON_CMD%...
echo Web UI will be available at: http://127.0.0.1:8000
echo.

%PYTHON_CMD% -m uvicorn app:app --host 127.0.0.1 --port 8000
pause

