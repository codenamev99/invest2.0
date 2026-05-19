# Stooq Screener (Windows + PyCharm quick start)

These steps are for someone who already **cloned the repo** on Windows and wants to run it in PyCharm.

## 1) Open the project in PyCharm
- **File > Open** and select the cloned folder.
- If prompted, trust the project.

## 2) Create/select a Python interpreter
- **File > Settings > Project > Python Interpreter**
- Add a new **Virtualenv** (recommended) located at `./.venv`

## 3) Install dependencies
Open the PyCharm Terminal and run:
```
python -m pip install -r requirements.txt
```

Optional (earnings columns): the screener fetches earnings dates from the public Nasdaq earnings calendar API (no API key required).

## 4) Set up daily data refresh
Preferred: use Polygon for automated daily refresh. Set your API key as an environment variable named `POLYGON_API_KEY`; do not paste the key into the repo.

PowerShell example:
```
[Environment]::SetEnvironmentVariable("POLYGON_API_KEY", "your_key_here", "User")
```

The Polygon updater appends or updates the latest daily bar for symbols that already exist under your data folder. Keep an initial Stooq-style data folder in place, then Polygon can maintain it day to day.

Optional fallback: download Stooq manually.

## 5) Download Stooq data
- On **Windows**, the US daily text bundle from Stooq is typically **`d_us_txt.zip`** (folder after extract: **`d_us_txt`**, containing `daily/`).
- You can point `STOOQ_SRC` at either the **zip** or the **extracted folder**.
  - Example folder: `C:\Users\you\Downloads\d_us_txt\daily\...`
  - Example zip: `C:\Users\you\Downloads\d_us_txt.zip`

## 6) Set the Stooq download location (optional fallback)
Open `run_daily.bat` and update the configuration block near the top if you want to use Stooq instead of Polygon. By default `STOOQ_SRC` is **empty** (Stooq refresh skipped) until you set it.

**Do not use Mac paths like `/Users/yourname/...` on Windows** — Python may turn that into `C:\Users\yourname\...`, which is wrong if that user folder does not exist on this PC.

Use a real path to **`d_us_txt`** (extracted) or **`d_us_txt.zip`**:
```
set "STOOQ_SRC=C:\Users\you\Downloads\d_us_txt"
```
or:
```
set "STOOQ_SRC=C:\Users\you\Downloads\d_us_txt.zip"
```

Other variables:
```
set "STOOQ_MODE=copy"
set "DATA_DEST=%PROJECT_DIR%data 2"
set "ROOT_DATA=%PROJECT_DIR%data 2\daily\us"
```

`STOOQ_MODE` controls cleanup behavior:
- `move` moves the Stooq data into the project (removes it from the source location).
- `copy` keeps the source zip/folder intact (recommended).

Optional: if you want the project to store the data somewhere else, update both:
- `DATA_DEST` (where the project will keep the data copy), and
- `ROOT_DATA` (must match `DATA_DEST` + `\daily\us`)

Example:
```
set "DATA_DEST=C:\stooq\data"
set "ROOT_DATA=C:\stooq\data\daily\us"
```

To skip the auto-refresh step entirely, set:
```
set "STOOQ_SRC="
```

## 7) Run the daily job
From the PyCharm Terminal (project root). In **PowerShell** use:
```
.\run_daily.bat
```
In **cmd.exe**, `run_daily.bat` also works.

This will:
- Refresh from Polygon if `POLYGON_API_KEY` is set, otherwise copy/refresh Stooq data if `STOOQ_SRC` is set
- Install requirements (if needed)
- Generate `nyse_tickers.csv`
- Run the screener

## 8) Find the results
The output file is:
```
results.xlsx
```

