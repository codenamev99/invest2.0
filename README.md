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

## 4) Download Stooq data
- Download the daily **Stooq data zip** (the file is usually named `data.zip`).
- Extract it so you have a folder named `data` that contains `daily/`.
  - Example: `C:\Users\you\Downloads\data\daily\...`

## 5) Set the Stooq download location (important)
Open `run_daily.bat` and update the configuration block near the top:
```
set "STOOQ_SRC=/Users/v/Downloads/data"
set "STOOQ_MODE=move"
set "DATA_DEST=%PROJECT_DIR%data 2"
set "ROOT_DATA=%PROJECT_DIR%data 2\daily\us"
```

Change `STOOQ_SRC` to **where your Stooq download folder lives** on Windows:
```
set "STOOQ_SRC=C:\Users\you\Downloads\data"
```

`STOOQ_MODE` controls cleanup behavior:
- `move` will move the `data` folder into the project (removes it from Downloads).
- `copy` keeps the source intact.

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
From the PyCharm Terminal:
```
run_daily.bat
```

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

