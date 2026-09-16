{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Refresh Stooq Data (copy from Downloads)",
      "type": "shell",
      "command": "${workspaceFolder}/.venv/bin/python",
      "args": [
        "${workspaceFolder}/refresh_stooq_dump.py",
        "--src",
        "~/Downloads/d_us_txt.zip",
        "--dest",
        "${workspaceFolder}/data 2",
        "--mode",
        "copy"
      ],
      "options": {
        "cwd": "${workspaceFolder}"
      },
      "problemMatcher": [],
      "presentation": {
        "reveal": "always",
        "panel": "shared",
        "clear": false
      }
    },
    {
      "label": "Run Screener (Daily beta, 252d)",
      "type": "shell",
      "command": "${workspaceFolder}/.venv/bin/python",
      "args": [
        "${workspaceFolder}/screen_stooq.py",
        "--tickers",
        "${workspaceFolder}/us_tickers.csv",
        "--root",
        "${workspaceFolder}/data 2/daily/us",
        "--benchmark",
        "SPY.US",
        "--beta_freq",
        "daily",
        "--beta_lookback",
        "252",
        "--out",
        "${workspaceFolder}/results.xlsx"
      ],
      "options": {
        "cwd": "${workspaceFolder}"
      },
      "problemMatcher": [],
      "presentation": {
        "reveal": "always",
        "panel": "shared",
        "clear": false
      }
    },
    {
      "label": "Run Screener (Monthly beta, 60mo)",
      "type": "shell",
      "command": "${workspaceFolder}/.venv/bin/python",
      "args": [
        "${workspaceFolder}/screen_stooq.py",
        "--tickers",
        "${workspaceFolder}/us_tickers.csv",
        "--root",
        "${workspaceFolder}/data 2/daily/us",
        "--benchmark",
        "SPY.US",
        "--beta_freq",
        "monthly",
        "--beta_months",
        "60",
        "--out",
        "${workspaceFolder}/results.xlsx"
      ],
      "options": {
        "cwd": "${workspaceFolder}"
      },
      "problemMatcher": [],
      "presentation": {
        "reveal": "always",
        "panel": "shared",
        "clear": false
      }
    },
    {
      "label": "Refresh + Screen (Daily)",
      "type": "shell",
      "dependsOn": [
        "Refresh Stooq Data (copy from Downloads)",
        "Run Screener (Daily beta, 252d)"
      ],
      "dependsOrder": "sequence",
      "problemMatcher": []
    },
    {
      "label": "Refresh + Screen (Monthly)",
      "type": "shell",
      "dependsOn": [
        "Refresh Stooq Data (copy from Downloads)",
        "Run Screener (Monthly beta, 60mo)"
      ],
      "dependsOrder": "sequence",
      "problemMatcher": []
    }
  ]
}
