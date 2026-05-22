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
rem Preferred: set POLYGON_API_KEY in your user environment or Task Scheduler to auto-refresh from Polygon.
rem Optional fallback: Stooq US daily (txt) is usually named d_us_txt (.zip) or extracted folder d_us_txt with daily\ inside.
rem Set STOOQ_SRC to that folder or zip, e.g. C:\Users\YourName\Downloads\d_us_txt  or  ...\d_us_txt.zip
rem Use a real Windows path — NOT /Users/... (Python maps that to C:\Users\... and breaks if that user does not exist).
rem Leave empty to skip refresh (you must already have data under DATA_DEST, see below).
rem Example: set "STOOQ_SRC=C:\Users\YourName\Downloads\d_us_txt"
set "STOOQ_SRC="
set "STOOQ_MODE=copy"
set "DATA_DEST=%PROJECT_DIR%data 2"
set "TICKERS_FILE=%PROJECT_DIR%nyse_tickers.csv"
set "RESULTS_FILE=%PROJECT_DIR%results.xlsx"
set "ROOT_DATA=%PROJECT_DIR%data 2\daily\us"
set "BENCHMARK=SPY.US"
if "%POLYGON_BOOTSTRAP_YEARS%"=="" set "POLYGON_BOOTSTRAP_YEARS=2"

rem -------- Daily Steps --------
if not "%POLYGON_API_KEY%"=="" (
    if not exist "%ROOT_DATA%\nyse stocks" (
        echo NYSE data folder missing; bootstrapping %POLYGON_BOOTSTRAP_YEARS% years from Polygon.
        %PYTHON_CMD% refresh_polygon_daily.py --bootstrap --root "%ROOT_DATA%" --bootstrap-years "%POLYGON_BOOTSTRAP_YEARS%"
        if errorlevel 1 (
            echo ERROR: Polygon bootstrap failed. Check POLYGON_API_KEY, plan history, and rate limits.
            exit /b 1
        )
    )
    %PYTHON_CMD% refresh_polygon_daily.py --root "%ROOT_DATA%"
    if errorlevel 1 (
        echo ERROR: refresh_polygon_daily.py failed. Check POLYGON_API_KEY and your data path, then retry.
        exit /b 1
    )
) else if not "%STOOQ_SRC%"=="" (
    %PYTHON_CMD% refresh_stooq_dump.py --src "%STOOQ_SRC%" --dest "%DATA_DEST%" --mode "%STOOQ_MODE%"
    if errorlevel 1 (
        echo ERROR: refresh_stooq_dump.py failed. Fix STOOQ_SRC or your data path, then retry.
        exit /b 1
    )
) else (
    echo Skipping data refresh. Set POLYGON_API_KEY or STOOQ_SRC in run_daily.bat to enable, or copy data into: "%DATA_DEST%"
)

if exist "requirements.txt" (
    %PYTHON_CMD% -m pip install -r "requirements.txt"
    if errorlevel 1 exit /b 1
)

rem No trailing \ before " — that breaks CMD parsing (escaped quote).
if not exist "%ROOT_DATA%\nyse stocks" (
    echo ERROR: NYSE data folder not found:
    echo   "%ROOT_DATA%\nyse stocks"
    echo Download/extract Stooq US daily data and set STOOQ_SRC, or place files under the path above.
    exit /b 1
)

%PYTHON_CMD% generate_tickers.py --dir "%ROOT_DATA%\nyse stocks" --out "%TICKERS_FILE%"
if errorlevel 1 exit /b 1

%PYTHON_CMD% screen_stooq.py --run_mode all --tickers "%TICKERS_FILE%" --root "%ROOT_DATA%" --benchmark "%BENCHMARK%" --out "%RESULTS_FILE%"
if errorlevel 1 exit /b 1

echo Done.
