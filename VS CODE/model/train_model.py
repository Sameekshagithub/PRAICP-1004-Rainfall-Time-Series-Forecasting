"""
PRAICP-1004 :: Rainfall Time Series Forecasting
------------------------------------------------
Loads the Changi Climate Station rainfall data, trains the forecasting
models (SARIMA + Random Forest for total rainfall, Random Forest for
highest-daily rainfall), evaluates them, generates a 12-month-ahead
forecast for both targets, and writes everything the web frontend needs
into model/forecasts.json.

Run this once (or whenever the source data changes):
    python model/train_model.py
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "Data")
OUT_PATH = os.path.join(BASE_DIR, "model", "forecasts.json")


def load_data():
    total = pd.read_csv(os.path.join(DATA_DIR, "rainfall-monthly-total.csv"))
    maxday = pd.read_csv(os.path.join(DATA_DIR, "rainfall-monthly-highest-daily-total.csv"))
    raindays = pd.read_csv(os.path.join(DATA_DIR, "rainfall-monthly-number-of-rain-days.csv"))

    df = total.merge(maxday, on="month").merge(raindays, on="month")
    df.rename(columns={
        "maximum_rainfall_in_a_day": "max_rainfall_in_a_day",
        "no_of_rainy_days": "rainy_days"
    }, inplace=True)
    df["month"] = pd.to_datetime(df["month"], format="%Y-%m")
    df = df.sort_values("month").reset_index(drop=True)
    df.set_index("month", inplace=True)
    df = df.asfreq("MS")
    return df


def make_lag_features(series, lags, roll_windows):
    fe = pd.DataFrame(index=series.index)
    fe["value"] = series
    fe["month_num"] = series.index.month
    fe["year"] = series.index.year
    for lag in lags:
        fe[f"lag_{lag}"] = series.shift(lag)
    for w in roll_windows:
        fe[f"rolling_mean_{w}"] = series.shift(1).rolling(w).mean()
    return fe.dropna()


def evaluate(y_true, y_pred):
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "R2": round(float(r2_score(y_true, y_pred)), 3),
    }


def main():
    df = load_data()

    # ---------- Stationarity check ----------
    adf_stat, adf_p = adfuller(df["total_rainfall"])[:2]

    # ================================================================
    # 1) TOTAL RAINFALL — feature engineering + train/test split
    # ================================================================
    FEATURES = ["month_num", "year", "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
                "rolling_mean_3", "rolling_mean_12"]
    fe = make_lag_features(df["total_rainfall"], lags=[1, 2, 3, 6, 12], roll_windows=[3, 12])
    fe = fe.rename(columns={"value": "total_rainfall"})

    train_size = int(len(fe) * 0.85)
    train, test = fe.iloc[:train_size], fe.iloc[train_size:]
    X_train, y_train = train[FEATURES], train["total_rainfall"]
    X_test, y_test = test[FEATURES], test["total_rainfall"]

    results = {}

    # --- SARIMA ---
    sarima_model = sm.tsa.statespace.SARIMAX(
        df["total_rainfall"].loc[train.index.min():train.index.max()],
        order=(1, 0, 1), seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False, enforce_invertibility=False
    ).fit(disp=False)
    sarima_test_preds = sarima_model.get_forecast(steps=len(test)).predicted_mean
    results["SARIMA"] = evaluate(y_test, sarima_test_preds.values)

    # --- Linear Regression ---
    lin_reg = LinearRegression().fit(X_train, y_train)
    results["Linear Regression"] = evaluate(y_test, lin_reg.predict(X_test))

    # --- Random Forest (default) ---
    rf = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train, y_train)
    results["Random Forest"] = evaluate(y_test, rf.predict(X_test))

    # --- Random Forest (tuned) ---
    tscv = TimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(
        RandomForestRegressor(random_state=42),
        {"n_estimators": [100, 200, 300], "max_depth": [None, 5, 10], "min_samples_split": [2, 5]},
        cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=-1
    ).fit(X_train, y_train)
    best_rf = grid.best_estimator_
    results["Random Forest (Tuned)"] = evaluate(y_test, best_rf.predict(X_test))

    best_model_name = min(results, key=lambda k: results[k]["RMSE"])

    # ---------- Refit SARIMA + Random Forest on FULL history, forecast next 12 months ----------
    final_sarima = sm.tsa.statespace.SARIMAX(
        df["total_rainfall"], order=(1, 0, 1), seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False, enforce_invertibility=False
    ).fit(disp=False)
    fc = final_sarima.get_forecast(steps=12)
    sarima_future = fc.predicted_mean
    sarima_ci = fc.conf_int()

    final_rf = RandomForestRegressor(n_estimators=grid.best_params_["n_estimators"],
                                      max_depth=grid.best_params_["max_depth"],
                                      min_samples_split=grid.best_params_["min_samples_split"],
                                      random_state=42)
    final_rf.fit(fe[FEATURES], fe["total_rainfall"])

    future_dates = pd.date_range(df.index.max() + pd.offsets.MonthBegin(1), periods=12, freq="MS")
    history_ext = fe["total_rainfall"].copy()
    rf_future = []
    for d in future_dates:
        row = {
            "month_num": d.month, "year": d.year,
            "lag_1": history_ext.iloc[-1], "lag_2": history_ext.iloc[-2], "lag_3": history_ext.iloc[-3],
            "lag_6": history_ext.iloc[-6], "lag_12": history_ext.iloc[-12],
            "rolling_mean_3": history_ext.iloc[-3:].mean(), "rolling_mean_12": history_ext.iloc[-12:].mean(),
        }
        pred = float(final_rf.predict(pd.DataFrame([row])[FEATURES])[0])
        rf_future.append(pred)
        history_ext.loc[d] = pred

    rainfall_forecast = []
    for i, d in enumerate(future_dates):
        rainfall_forecast.append({
            "date": d.strftime("%Y-%m-%d"),
            "label": d.strftime("%B %Y"),
            "month_name": d.strftime("%B"),
            "sarima_mm": round(float(sarima_future.iloc[i]), 1),
            "sarima_lower_ci": round(float(sarima_ci.iloc[i, 0]), 1),
            "sarima_upper_ci": round(float(sarima_ci.iloc[i, 1]), 1),
            "rf_mm": round(rf_future[i], 1),
            "historical_avg_mm": round(float(df[df.index.month == d.month]["total_rainfall"].mean()), 1),
        })

    # ================================================================
    # 2) HIGHEST DAILY RAINFALL — feature engineering + forecast
    # ================================================================
    MAXDAY_FEATURES = ["month_num", "year", "lag_1", "lag_2", "lag_3", "lag_12", "rolling_mean_12"]
    fe2 = make_lag_features(df["max_rainfall_in_a_day"], lags=[1, 2, 3, 12], roll_windows=[12])
    fe2 = fe2.rename(columns={"value": "max_rainfall_in_a_day"})

    train_size2 = int(len(fe2) * 0.85)
    train2, test2 = fe2.iloc[:train_size2], fe2.iloc[train_size2:]
    maxday_rf = RandomForestRegressor(n_estimators=200, random_state=42)
    maxday_rf.fit(train2[MAXDAY_FEATURES], train2["max_rainfall_in_a_day"])
    maxday_metrics = evaluate(test2["max_rainfall_in_a_day"], maxday_rf.predict(test2[MAXDAY_FEATURES]))

    maxday_rf_full = RandomForestRegressor(n_estimators=200, random_state=42)
    maxday_rf_full.fit(fe2[MAXDAY_FEATURES], fe2["max_rainfall_in_a_day"])

    history_ext2 = fe2["max_rainfall_in_a_day"].copy()
    maxday_future = []
    for d in future_dates:
        row = {
            "month_num": d.month, "year": d.year,
            "lag_1": history_ext2.iloc[-1], "lag_2": history_ext2.iloc[-2], "lag_3": history_ext2.iloc[-3],
            "lag_12": history_ext2.iloc[-12], "rolling_mean_12": history_ext2.iloc[-12:].mean(),
        }
        pred = float(maxday_rf_full.predict(pd.DataFrame([row])[MAXDAY_FEATURES])[0])
        maxday_future.append(pred)
        history_ext2.loc[d] = pred

    # Per-calendar-month historical stats + forecast (Jan..Dec)
    calendar_months = ["January", "February", "March", "April", "May", "June",
                        "July", "August", "September", "October", "November", "December"]
    maxday_by_month = {}
    for m_num, m_name in enumerate(calendar_months, start=1):
        hist = df[df.index.month == m_num]["max_rainfall_in_a_day"]
        fut_rows = [(d, v) for d, v in zip(future_dates, maxday_future) if d.month == m_num]
        maxday_by_month[m_name] = {
            "historical_avg_mm": round(float(hist.mean()), 1),
            "historical_max_mm": round(float(hist.max()), 1),
            "historical_min_mm": round(float(hist.min()), 1),
            "forecast_mm": round(fut_rows[0][1], 1) if fut_rows else None,
            "forecast_label": fut_rows[0][0].strftime("%B %Y") if fut_rows else None,
        }

    # ================================================================
    # 3) EDA / chart data
    # ================================================================
    history_series = [
        {"date": idx.strftime("%Y-%m-%d"), "total_rainfall": float(row["total_rainfall"]),
         "max_rainfall_in_a_day": float(row["max_rainfall_in_a_day"]), "rainy_days": int(row["rainy_days"])}
        for idx, row in df.iterrows()
    ]
    seasonal_avg = df.groupby(df.index.month)["total_rainfall"].mean().round(1)
    seasonal_avg_by_month = {calendar_months[i - 1]: float(v) for i, v in seasonal_avg.items()}

    yearly = df["total_rainfall"].resample("YE").sum()
    yearly_totals = [{"year": int(idx.year), "total_rainfall": round(float(v), 1)} for idx, v in yearly.items()]

    output = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "data_range": {
            "start": df.index.min().strftime("%B %Y"),
            "end": df.index.max().strftime("%B %Y"),
            "n_months": int(len(df)),
        },
        "adf_test": {
            "statistic": round(float(adf_stat), 3),
            "p_value": round(float(adf_p), 5),
            "is_stationary": bool(adf_p < 0.05),
        },
        "model_comparison": results,
        "best_model": best_model_name,
        "rainfall_forecast": rainfall_forecast,
        "maxday_forecast_metrics": maxday_metrics,
        "maxday_by_month": maxday_by_month,
        "seasonal_avg_by_month": seasonal_avg_by_month,
        "yearly_totals": yearly_totals,
        "history": history_series,
        "calendar_months": calendar_months,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved forecasts to {OUT_PATH}")
    print(f"Best model: {best_model_name}  ->  {results[best_model_name]}")


if __name__ == "__main__":
    main()
