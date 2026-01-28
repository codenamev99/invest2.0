@echo off
setlocal

rem Run from this script's folder
set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

rem Pick Python command: prefer py -3 on Windows
where py >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON_CMD=py -3"
) else (
    set "PYTHON_CMD=python"
)

rem Optional: activate venv if present
if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

rem -------- Configuration --------
rem Set this to your Stooq download path if you want auto refresh
set "STOOQ_SRC=/Users/v/Downloads/data"
set "STOOQ_MODE=move"
set "DATA_DEST=%PROJECT_DIR%data 2"
set "TICKERS_FILE=%PROJECT_DIR%nyse_tickers.csv"
set "RESULTS_FILE=%PROJECT_DIR%results.xlsx"
set "ROOT_DATA=%PROJECT_DIR%data 2\daily\us"
set "BENCHMARK=SPY.US"

rem -------- Daily Steps --------
if not "%STOOQ_SRC%"=="" (
    %PYTHON_CMD% refresh_stooq_dump.py --src "%STOOQ_SRC%" --dest "%DATA_DEST%" --mode "%STOOQ_MODE%"
) else (
    echo Skipping data refresh. Set STOOQ_SRC in run_daily.bat to enable.
)

if exist "requirements.txt" (
    %PYTHON_CMD% -m pip install -r "requirements.txt"
)

%PYTHON_CMD% generate_tickers.py --dir "%ROOT_DATA%\nyse stocks" --out "%TICKERS_FILE%"

%PYTHON_CMD% screen_stooq.py --tickers "%TICKERS_FILE%" --root "%ROOT_DATA%" --benchmark "%BENCHMARK%" --out "%RESULTS_FILE%"

echo Done.
