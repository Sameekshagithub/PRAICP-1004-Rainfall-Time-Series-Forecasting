"""
PRAICP-1004 :: Rainfall Time Series Forecasting — Web App
-----------------------------------------------------------
Flask backend that serves the frontend dashboard and exposes the
precomputed forecast data (model/forecasts.json) as a small JSON API.

Run:
    pip install -r requirements.txt
    python app.py
Then open:
    http://127.0.0.1:5000
"""

import os
import json

from flask import Flask, jsonify, render_template

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FORECASTS_PATH = os.path.join(BASE_DIR, "model", "forecasts.json")

app = Flask(__name__)


def load_forecasts():
    if not os.path.exists(FORECASTS_PATH):
        raise FileNotFoundError(
            "forecasts.json not found. Run 'python model/train_model.py' first "
            "to generate it from the rainfall dataset."
        )
    with open(FORECASTS_PATH, "r") as f:
        return json.load(f)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/forecasts")
def api_forecasts():
    """Full precomputed dataset: model comparison, 12-month forecasts,
    per-calendar-month highest-daily-rainfall stats, and history for charts."""
    return jsonify(load_forecasts())


@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
