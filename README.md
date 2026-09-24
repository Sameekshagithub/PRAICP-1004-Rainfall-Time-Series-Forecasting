# 🌧️ PRAICP-1004 — Rainfall Time Series Forecasting

**Artificial Intelligence Capstone Project** | DataMites™ Project Mentoring PR-1004

Forecasts monthly total rainfall for the next 12 months and the highest single-day rainfall
per calendar month, using 38+ years of historical data (1982–2020) from the Changi Climate
Station, Singapore. Includes a full EDA + modelling pipeline **and** an in-notebook
interactive dashboard (`ipywidgets`) for exploring the forecasts without reading code.

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Dataset](#-dataset)
3. [Repository Contents](#-repository-contents)
4. [Setup & Installation](#-setup--installation)
5. [How to Run](#-how-to-run)
6. [Notebook Structure](#-notebook-structure)
7. [Methodology](#-methodology)
8. [Results Summary](#-results-summary)
9. [Interactive Dashboard](#-interactive-dashboard)
10. [Known Limitations](#-known-limitations)
11. [Future Enhancements](#-future-enhancements)
12. [Tech Stack](#-tech-stack)
13. [Credits & License](#-credits--license)

---

## 🎯 Project Overview

**Business case:** Develop a Machine Learning / Time-Series model to predict rainfall for the
next one year using previous data, and separately forecast the highest rainfall that occurs
in a single day for a given month.

**Goals:**
- Understand and preprocess historical rainfall data
- Build and compare multiple forecasting models
- Forecast total monthly rainfall 12 months ahead
- Forecast the highest daily rainfall expected per calendar month
- Present the results through both static analysis and an interactive dashboard

---

## 📊 Dataset

| | |
|---|---|
| **Source** | National Environment Agency (Singapore), via data.gov.sg |
| **Station** | Changi Climate Station |
| **Coverage** | January 1982 – June 2020 (462 months) |
| **Format** | 3 CSV files, merged on the `month` column |
| **Missing values** | None |
| **Duplicates** | None |

| File | Column | Description |
|---|---|---|
| `rainfall-monthly-total.csv` | `total_rainfall` | Total rainfall for the month (mm) |
| `rainfall-monthly-highest-daily-total.csv` | `maximum_rainfall_in_a_day` | Highest single-day rainfall in the month (mm) |
| `rainfall-monthly-number-of-rain-days.csv` | `no_of_rainy_days` | Number of rain days (≥0.2 mm) in the month |

> **Note on granularity:** the source data is aggregated **monthly**, not daily. "Highest
> rainfall in a day for a month" is therefore forecast as the monthly `maximum_rainfall_in_a_day`
> statistic, not from a true daily-resolution series. See [Known Limitations](#-known-limitations).

---

## 📁 Repository Contents

```
PRAICP-1004-Rainfall-TS-Forecasting/
├── PRAICP-1004-Rainfall-TS-Forecasting.ipynb   # Main notebook (already executed, outputs saved)
├── PRAICP-1004-RainfallTS.zip                  # Raw dataset (3 CSVs + metadata)
└── README.md                                    # This file
```

The notebook auto-extracts `PRAICP-1004-RainfallTS.zip` into a local `Data/` folder the first
time it runs, so no manual unzip step is required as long as both files sit in the same
directory.

---

## ⚙️ Setup & Installation

### Option A — Google Colab (zero setup)
1. Upload `PRAICP-1004-Rainfall-TS-Forecasting.ipynb` and `PRAICP-1004-RainfallTS.zip` to the
   same Colab session (`Files` panel → drag & drop, or `Files → Upload`).
2. Open the notebook and run all cells (`Runtime → Run all`). Colab has `ipywidgets` enabled
   by default, so the interactive dashboard renders immediately.

### Option B — Local Jupyter Notebook / JupyterLab
```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels xgboost ipywidgets jupyterlab

# 3. Enable widgets (usually automatic on modern JupyterLab/Notebook 7+)
jupyter nbextension enable --py widgetsnbextension   # only needed on classic Notebook <7

# 4. Launch
jupyter lab      # or: jupyter notebook
```
Place `PRAICP-1004-RainfallTS.zip` in the same folder as the `.ipynb` file, then open it and
run all cells.

### Option C — VS Code
1. Install the **Python** and **Jupyter** extensions.
2. Open the project folder in VS Code.
3. Create/select a Python interpreter with the packages listed above installed.
4. Open the `.ipynb` file — VS Code's built-in Jupyter support renders `ipywidgets` natively.
5. Click **Run All**.

---

## ▶️ How to Run

1. Make sure `PRAICP-1004-RainfallTS.zip` is in the same directory as the notebook (or an
   already-extracted `Data/` folder — the notebook checks for this automatically).
2. Run all cells top to bottom.
3. Scroll to **Section 13 — Interactive Frontend** to use the live dashboard:
   - Pick a month from the dropdown → click **"Get Rainfall Forecast"**
   - Pick a calendar month → click **"Get Highest-Daily Forecast"**
4. All plots, tables, and forecasts regenerate automatically; no external services or API
   keys are required.

**Total runtime:** ~30–60 seconds on a typical laptop (SARIMA fitting and `GridSearchCV`
hyperparameter search are the slowest steps).

---

## 🗂️ Notebook Structure

| # | Section |
|---|---|
| 1 | Problem Statement & Project Objective |
| 2 | Import Python Libraries |
| 3 | Upload the Dataset and Domain Analysis |
| 4 | Basic Checks |
| 5 | Data Preprocessing |
| 6 | Exploratory Data Analysis (EDA) |
| 7 | Data Preparation for Modelling (lag features, train/test split) |
| 8 | Model Building & Training (SARIMA, Linear Regression, Random Forest, Gradient Boosting, XGBoost) |
| 9 | Hyperparameter Tuning (`GridSearchCV` + `TimeSeriesSplit`) |
| 10 | Model Comparison Report |
| 11 | One-Year Rainfall Forecast |
| 12 | Highest Daily Rainfall Forecast |
| 13 | **Interactive Frontend — Rainfall Forecast Dashboard** (`ipywidgets`) |
| 14 | Business Insights |
| 15 | Challenges Faced |
| 16 | Final Conclusion |
| 17 | Project Links |

---

## 🧠 Methodology

**Time-series integrity:**
- Chronological (not random) train/test split — the most recent ~15% of months are held out.
- `TimeSeriesSplit` cross-validation used for hyperparameter tuning (never random k-fold).

**Total rainfall forecasting — models compared:**
| Model | Approach |
|---|---|
| SARIMA `(1,0,1)(1,1,1,12)` | Classical statistical model, captures trend + seasonality directly |
| Linear Regression | Baseline on lag + calendar features |
| Random Forest | Ensemble on lag + calendar features |
| Random Forest (Tuned) | `GridSearchCV` over `n_estimators`, `max_depth`, `min_samples_split` |
| Gradient Boosting | Sequential boosting ensemble |
| XGBoost | Optimized gradient boosting |

**Feature engineering:** lag features (`t-1, t-2, t-3, t-6, t-12`), rolling means (3- and
12-month), calendar month/year.

**Evaluation metrics:** MAE, RMSE, R² on the held-out chronological test period.

**Highest-daily-rainfall forecasting:** the same lag-feature approach is applied independently
to the `maximum_rainfall_in_a_day` series using a Random Forest Regressor.

**Multi-step forecasting:** both target series use a **recursive** forecasting loop —
each month's prediction is fed back in as a lag feature to predict the next month, producing
a full 12-month-ahead forecast from single-step models.

---

## 📈 Results Summary

On the held-out chronological test period:

| Model | MAE (mm) | RMSE (mm) | R² |
|---|---|---|---|
| **Linear Regression** | ~68.9 | ~82.8 | ~0.10 |
| SARIMA | ~69.6 | ~84.5 | ~0.06 |
| Gradient Boosting | ~69.1 | ~85.4 | ~0.04 |
| Random Forest (Tuned) | ~76.2 | ~95.2 | ~-0.20 |
| XGBoost | ~77.2 | ~100.4 | ~-0.33 |
| Random Forest (default) | ~82.7 | ~103.4 | ~-0.41 |

*(Exact values depend on the current dataset version and random seeds; see the notebook's
own Section 10 output for the authoritative numbers.)*

**Key takeaway:** rainfall is a genuinely noisy climate signal — even the best model only
explains a modest share of month-to-month variance (R² ≈ 0.1). The forecasts are most useful
for capturing the **seasonal (monsoon) pattern** rather than predicting exact totals. This is
stated plainly in the notebook rather than glossed over.

---

## 🖥️ Interactive Dashboard

Section 13 embeds a small `ipywidgets`-based dashboard directly in the notebook:

1. **Rainfall Forecast Dashboard** — dropdown of the next 12 forecast months + button →
   shows SARIMA forecast (with 95% CI), tuned Random Forest forecast, historical average,
   and a bar chart comparison.
2. **Highest Daily Rainfall Dashboard** — dropdown of calendar months (Jan–Dec) + button →
   shows historical average, historical record, and the forecasted single-day maximum.

> ⚠️ The dashboard is interactive **only with a live kernel** (Jupyter, JupyterLab, Colab, or
> VS Code's notebook interface). It will render as static images/text in a PDF or HTML export.

---

## ⚠️ Known Limitations

- **Monthly granularity only** — no true daily rainfall series is available in the source
  data, so single-day extremes are approximated via the monthly maximum-daily-total column.
- **High natural variability** — rainfall has low autocorrelation month-to-month beyond
  seasonality, which caps achievable model accuracy (see R² values above).
- **No exogenous variables** — the models use only the rainfall series itself; ENSO
  (El Niño/La Niña) indices or other climate drivers are not incorporated.
- **Recursive forecasting compounds error** — each ML forecast step feeds its own prediction
  back in as a lag feature, so uncertainty grows over the 12-month horizon (SARIMA's
  confidence interval reflects this; the ML recursive forecast does not carry an explicit CI).

---

## 🚀 Future Enhancements

- Source **daily-resolution** rainfall data for true single-day forecasting.
- Incorporate exogenous climate indices (ENSO/ONI, monsoon indices).
- Deploy as a scheduled pipeline/API that auto-refits as new monthly data arrives.
- Extend the in-notebook `ipywidgets` dashboard into a standalone **Flask/Streamlit web app**
  (a companion Flask app + light-blue frontend is available separately — see
  `PRAICP-1004-Rainfall-Forecast-App.zip`).
- Add prediction intervals to the ML recursive forecasts (e.g., via quantile regression or
  bootstrapped residuals).

---

## 🔧 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Data handling | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Statistical modelling | statsmodels (SARIMA, seasonal decomposition, ADF test) |
| Machine learning | scikit-learn (Random Forest, Linear Regression, GridSearchCV, TimeSeriesSplit), XGBoost |
| In-notebook UI | ipywidgets |
| Notebook environment | Jupyter Notebook / JupyterLab / Google Colab / VS Code |

---

## 🏷️ Credits & License

- **Dataset:** National Environment Agency, Singapore (via data.gov.sg), licensed under the
  [Singapore Open Data Licence](https://data.gov.sg/open-data-licence).
- **Project code:** PRAICP-1004, DataMites™ Project Mentoring PR-1004.
- This notebook is submitted as part of an Artificial Intelligence Capstone Project and is
  intended for educational/academic use.
