@echo off
setlocal

rem Run from this script's folder
set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

rem Pick Python command: prefer py -3 on Windows
rem Use "if not errorlevel 1" — NOT "if %%errorlevel%%==0" (can parse as "if ==0" and break)
where py >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=py -3"
) else (
    set "PYTHON_CMD=python"
)

rem Optional: activate venv if present
if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

rem -------- Configuration --------
rem Stooq US daily (txt) on Windows is usually named d_us_txt (.zip) or extracted folder d_us_txt with daily\ inside.
rem Set STOOQ_SRC to that folder or zip, e.g. C:\Users\YourName\Downloads\d_us_txt  or  ...\d_us_txt.zip
rem Use a real Windows path — NOT /Users/... (Python maps that to C:\Users\... and breaks if that user does not exist).
rem Leave empty to skip refresh (you must already have data under DATA_DEST, see below).
set "STOOQ_SRC="
set "STOOQ_MODE=copy"
set "DATA_DEST=%PROJECT_DIR%data 2"
set "TICKERS_FILE=%PROJECT_DIR%nyse_tickers.csv"
set "RESULTS_FILE=%PROJECT_DIR%results.xlsx"
set "ROOT_DATA=%PROJECT_DIR%data 2\daily\us"
set "BENCHMARK=SPY.US"

rem -------- Daily Steps --------
if not "%STOOQ_SRC%"=="" (
    %PYTHON_CMD% refresh_stooq_dump.py --src "%STOOQ_SRC%" --dest "%DATA_DEST%" --mode "%STOOQ_MODE%"
    if errorlevel 1 (
        echo ERROR: refresh_stooq_dump.py failed. Fix STOOQ_SRC or your data path, then retry.
        exit /b 1
    )
) else (
    echo Skipping data refresh. Set STOOQ_SRC in run_daily.bat to enable, or copy data into: "%DATA_DEST%"
)

if exist "requirements.txt" (
    %PYTHON_CMD% -m pip install -r "requirements.txt"
    if errorlevel 1 exit /b 1
)

if not exist "%ROOT_DATA%\nyse stocks\" (
    echo ERROR: NYSE data folder not found:
    echo   "%ROOT_DATA%\nyse stocks"
    echo Download/extract Stooq US daily data and set STOOQ_SRC, or place files under the path above.
    exit /b 1
)

%PYTHON_CMD% generate_tickers.py --dir "%ROOT_DATA%\nyse stocks" --out "%TICKERS_FILE%"
if errorlevel 1 exit /b 1

%PYTHON_CMD% screen_stooq.py --tickers "%TICKERS_FILE%" --root "%ROOT_DATA%" --benchmark "%BENCHMARK%" --out "%RESULTS_FILE%"
if errorlevel 1 exit /b 1

echo Done.
