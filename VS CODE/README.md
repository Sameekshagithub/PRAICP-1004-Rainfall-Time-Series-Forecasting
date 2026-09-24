# 🌧️ PRAICP-1004 — Rainfall Time Series Forecasting (Web App)

An **Artificial Intelligence Capstone Project** that forecasts monthly rainfall totals and the
highest single-day rainfall per month, using historical data from the Changi Climate Station
(1982–2020). This folder is a complete, self-contained **VS Code project**: a Python/Flask
backend serving a light-blue themed interactive dashboard frontend.

---

## 📁 Project Structure

```
rainfall-forecast-app/
├── app.py                     # Flask backend (routes + JSON API)
├── requirements.txt           # Python dependencies
├── data/
│   └── Data/                  # Source rainfall CSVs (Changi Climate Station)
├── model/
│   ├── train_model.py         # Trains SARIMA + Random Forest models, writes forecasts.json
│   └── forecasts.json         # Precomputed forecasts & metrics consumed by the frontend
├── templates/
│   └── index.html             # Dashboard page
├── static/
│   ├── css/style.css          # Light-blue UI theme
│   └── js/app.js              # Dashboard logic (fetches API, renders charts)
└── .vscode/
    ├── launch.json            # One-click "Run & Debug" configs
    ├── settings.json
    └── extensions.json
```

---

## 🚀 How to Run (in VS Code)

1. **Open the folder in VS Code**: `File → Open Folder…` → select `rainfall-forecast-app`.
2. **Create a virtual environment** (recommended) and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   source .venv/bin/activate     # macOS / Linux

   pip install -r requirements.txt
   ```
3. **(Optional) Regenerate the forecasts** from the raw data — a precomputed
   `model/forecasts.json` is already included, so this step is optional unless you
   change the dataset or modelling code:
   ```bash
   python model/train_model.py
   ```
4. **Run the app**:
   - Press **F5** in VS Code (uses the included `launch.json`), **or**
   - Run from the terminal:
     ```bash
     python app.py
     ```
5. Open your browser at **http://127.0.0.1:5000**

---

## 🖥️ What the Dashboard Shows

- **Overview stat cards** — data coverage, best-performing model, its RMSE, and the ADF
  stationarity test result.
- **Total Rainfall Forecast panel** — pick any of the next 12 months from a dropdown and
  instantly see the SARIMA forecast (with 95% confidence interval), the tuned Random Forest
  forecast, and the historical average for that calendar month, plus a bar chart.
- **Highest Daily Rainfall panel** — pick a calendar month (Jan–Dec) and see the historical
  average, historical record, and the forecasted single-day maximum.
- **Historical Trend chart** — full 1982–present monthly rainfall history with the next
  12-month forecast overlaid.
- **Seasonal chart** — average rainfall by calendar month (monsoon pattern).
- **Model Comparison table** — MAE / RMSE / R² for SARIMA, Linear Regression, Random Forest,
  and tuned Random Forest, with the best model highlighted.

---

## 🧠 Modelling Approach (`model/train_model.py`)

- **Total rainfall**: lag features (1, 2, 3, 6, 12 months) + rolling means + calendar features,
  compared across Linear Regression, Random Forest, tuned Random Forest (`GridSearchCV` +
  `TimeSeriesSplit`), and SARIMA (order `(1,0,1)`, seasonal order `(1,1,1,12)`).
- **Highest daily rainfall**: the same lag-feature approach applied to the monthly
  `max_rainfall_in_a_day` series using a Random Forest Regressor.
- Both series are refit on the **full** history and forecast **12 months ahead** using a
  recursive forecasting loop.
- Train/test split is **chronological** (no shuffling) to avoid leaking future data into
  the past, as is required for time-series evaluation.

To retrain with different parameters, edit `model/train_model.py` and re-run it — it will
overwrite `model/forecasts.json`, which the Flask app reads on each request.

---

## 📊 Dataset

Source: National Environment Agency (Singapore), Changi Climate Station, via data.gov.sg.
Three monthly series (Jan 1982 – Jun 2020), merged on `month`:
- `rainfall-monthly-total.csv` — total rainfall (mm)
- `rainfall-monthly-highest-daily-total.csv` — highest single-day rainfall in the month (mm)
- `rainfall-monthly-number-of-rain-days.csv` — number of rain days in the month

---

## 🔧 Tech Stack

| Layer     | Technology                                   |
|-----------|-----------------------------------------------|
| Backend   | Python, Flask                                 |
| Modelling | scikit-learn (Random Forest, GridSearchCV), statsmodels (SARIMA) |
| Frontend  | HTML5, CSS3 (light-blue theme), vanilla JS    |
| Charts    | Chart.js (via CDN)                            |

---

## 📝 Notes

- `forecasts.json` is precomputed so the app starts instantly — no model training happens
  on page load. Re-run `train_model.py` whenever the source data changes.
- All chart colors and UI accents use a consistent **light-blue palette** (`#e3f2fd` →
  `#0d47a1`) defined as CSS variables at the top of `static/css/style.css`, so the whole
  theme can be re-colored from one place.
