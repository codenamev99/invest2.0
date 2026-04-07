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

## 4) Download Stooq data
- On **Windows**, the US daily text bundle from Stooq is typically **`d_us_txt.zip`** (folder after extract: **`d_us_txt`**, containing `daily/`).
- You can point `STOOQ_SRC` at either the **zip** or the **extracted folder**.
  - Example folder: `C:\Users\you\Downloads\d_us_txt\daily\...`
  - Example zip: `C:\Users\you\Downloads\d_us_txt.zip`

## 5) Set the Stooq download location (important)
Open `run_daily.bat` and update the configuration block near the top. By default `STOOQ_SRC` is **empty** (refresh skipped) until you set it.

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

## 6) Run the daily job
From the PyCharm Terminal (project root). In **PowerShell** use:
```
.\run_daily.bat
```
In **cmd.exe**, `run_daily.bat` also works.

This will:
- Copy/refresh the Stooq data into the project
- Install requirements (if needed)
- Generate `nyse_tickers.csv`
- Run the screener

## 7) Find the results
The output file is:
```
results.xlsx
```

