// PRAICP-1004 Rainfall Forecast Dashboard — frontend logic
// Fetches the precomputed forecast data from /api/forecasts and renders
// the stat cards, two interactive forecast panels, and three Chart.js charts.

let DATA = null;
let rainfallBarChart, maxdayBarChart, trendChart, seasonalChart;

const BLUE = {
  line: "#1e88e5",
  fill: "rgba(30, 136, 229, 0.12)",
  forecast: "#0d47a1",
  forecastFill: "rgba(13, 71, 161, 0.15)",
  bar1: "#42a5f5",
  bar2: "#1565c0",
  bar3: "#90caf9",
  grid: "rgba(21, 101, 192, 0.08)",
};

async function init() {
  try {
    const res = await fetch("/api/forecasts");
    if (!res.ok) throw new Error("Failed to load forecast data");
    DATA = await res.json();
  } catch (err) {
    document.getElementById("dataRangeBadge").textContent = "⚠️ Could not load data";
    console.error(err);
    return;
  }

  renderStats();
  populateMonthDropdowns();
  renderModelTable();
  renderTrendChart();
  renderSeasonalChart();

  document.getElementById("forecastBtn").addEventListener("click", showRainfallForecast);
  document.getElementById("maxdayBtn").addEventListener("click", showMaxdayForecast);

  // Show an initial result for the first available option
  showRainfallForecast();
  showMaxdayForecast();
}

function renderStats() {
  const d = DATA;
  document.getElementById("dataRangeBadge").textContent =
    `${d.data_range.start} – ${d.data_range.end}`;
  document.getElementById("statCoverage").textContent =
    `${d.data_range.n_months} months`;
  document.getElementById("statBestModel").textContent = d.best_model;
  document.getElementById("statBestRmse").textContent =
    `${d.model_comparison[d.best_model].RMSE} mm`;
  document.getElementById("statAdf").textContent =
    d.adf_test.is_stationary ? "Stationary ✅" : "Non-stationary ⚠️";
}

function populateMonthDropdowns() {
  const monthSelect = document.getElementById("monthSelect");
  monthSelect.innerHTML = DATA.rainfall_forecast
    .map((f, i) => `<option value="${i}">${f.label}</option>`)
    .join("");

  const calSelect = document.getElementById("calMonthSelect");
  calSelect.innerHTML = DATA.calendar_months
    .map((m) => `<option value="${m}">${m}</option>`)
    .join("");
  calSelect.value = "July";
}

function fmt(n) {
  return Number(n).toFixed(1);
}

function showRainfallForecast() {
  const idx = Number(document.getElementById("monthSelect").value || 0);
  const f = DATA.rainfall_forecast[idx];
  if (!f) return;

  const box = document.getElementById("rainfallResult");
  box.innerHTML = `
    <div class="result-title">📅 ${f.label}</div>
    <div class="result-row"><span class="r-label">SARIMA forecast</span>
      <span class="r-value">${fmt(f.sarima_mm)} mm</span></div>
    <div class="result-row"><span class="r-label">95% Confidence interval</span>
      <span class="r-value">${fmt(f.sarima_lower_ci)} – ${fmt(f.sarima_upper_ci)} mm</span></div>
    <div class="result-row"><span class="r-label">Random Forest (tuned)</span>
      <span class="r-value">${fmt(f.rf_mm)} mm</span></div>
    <div class="result-row"><span class="r-label">Historical average (${f.month_name})</span>
      <span class="r-value">${fmt(f.historical_avg_mm)} mm</span></div>
  `;

  const ctx = document.getElementById("rainfallBarChart").getContext("2d");
  const chartData = {
    labels: ["SARIMA", "Random Forest", "Historical Avg"],
    datasets: [{
      data: [f.sarima_mm, f.rf_mm, f.historical_avg_mm],
      backgroundColor: [BLUE.bar2, BLUE.bar1, BLUE.bar3],
      borderRadius: 8,
      maxBarThickness: 60,
    }],
  };

  if (rainfallBarChart) {
    rainfallBarChart.data = chartData;
    rainfallBarChart.options.plugins.title.text = `Rainfall Outlook — ${f.label}`;
    rainfallBarChart.update();
  } else {
    rainfallBarChart = new Chart(ctx, {
      type: "bar",
      data: chartData,
      options: baseBarOptions(`Rainfall Outlook — ${f.label}`, "mm"),
    });
  }
}

function showMaxdayForecast() {
  const monthName = document.getElementById("calMonthSelect").value;
  const m = DATA.maxday_by_month[monthName];
  if (!m) return;

  const box = document.getElementById("maxdayResult");
  box.innerHTML = `
    <div class="result-title">💧 ${monthName}</div>
    <div class="result-row"><span class="r-label">Historical average single-day max</span>
      <span class="r-value">${fmt(m.historical_avg_mm)} mm</span></div>
    <div class="result-row"><span class="r-label">Historical record (highest ever)</span>
      <span class="r-value">${fmt(m.historical_max_mm)} mm</span></div>
    <div class="result-row"><span class="r-label">Forecast (${m.forecast_label || "n/a"})</span>
      <span class="r-value">${m.forecast_mm !== null ? fmt(m.forecast_mm) + " mm" : "outside forecast window"}</span></div>
  `;

  const ctx = document.getElementById("maxdayBarChart").getContext("2d");
  const chartData = {
    labels: ["Historical Avg", "Historical Record", "Forecast"],
    datasets: [{
      data: [m.historical_avg_mm, m.historical_max_mm, m.forecast_mm ?? 0],
      backgroundColor: [BLUE.bar3, BLUE.bar2, BLUE.bar1],
      borderRadius: 8,
      maxBarThickness: 60,
    }],
  };

  if (maxdayBarChart) {
    maxdayBarChart.data = chartData;
    maxdayBarChart.options.plugins.title.text = `Highest Daily Rainfall — ${monthName}`;
    maxdayBarChart.update();
  } else {
    maxdayBarChart = new Chart(ctx, {
      type: "bar",
      data: chartData,
      options: baseBarOptions(`Highest Daily Rainfall — ${monthName}`, "mm"),
    });
  }
}

function baseBarOptions(title, unit) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      title: { display: true, text: title, color: "#0d47a1", font: { size: 13, weight: "700" } },
      tooltip: { callbacks: { label: (c) => `${c.formattedValue} ${unit}` } },
    },
    scales: {
      y: { beginAtZero: true, grid: { color: BLUE.grid }, ticks: { color: "#4a6178" } },
      x: { grid: { display: false }, ticks: { color: "#4a6178" } },
    },
  };
}

function renderTrendChart() {
  const hist = DATA.history;
  const fc = DATA.rainfall_forecast;

  const histLabels = hist.map((h) => h.date.slice(0, 7));
  const histValues = hist.map((h) => h.total_rainfall);
  const fcLabels = fc.map((f) => f.date.slice(0, 7));
  const fcValues = fc.map((f) => f.sarima_mm);

  // Build a combined label axis: history followed by forecast months
  const labels = [...histLabels, ...fcLabels];
  const historySeries = [...histValues, ...Array(fcValues.length).fill(null)];
  // connect the forecast line to the last historical point
  const forecastSeries = Array(histValues.length - 1).fill(null)
    .concat([histValues[histValues.length - 1]], fcValues);

  const ctx = document.getElementById("trendChart").getContext("2d");
  trendChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Historical Total Rainfall",
          data: historySeries,
          borderColor: BLUE.line,
          backgroundColor: BLUE.fill,
          pointRadius: 0,
          borderWidth: 1.5,
          tension: 0.15,
          fill: true,
        },
        {
          label: "1-Year Forecast (SARIMA)",
          data: forecastSeries,
          borderColor: BLUE.forecast,
          backgroundColor: BLUE.forecastFill,
          borderWidth: 2.5,
          borderDash: [6, 4],
          pointRadius: 2,
          tension: 0.15,
          fill: true,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { position: "top", labels: { color: "#0d47a1", font: { weight: "600" } } },
      },
      scales: {
        x: {
          ticks: { color: "#4a6178", maxTicksLimit: 14, autoSkip: true },
          grid: { display: false },
        },
        y: {
          title: { display: true, text: "Total Rainfall (mm)", color: "#4a6178" },
          grid: { color: BLUE.grid },
          ticks: { color: "#4a6178" },
        },
      },
    },
  });
}

function renderSeasonalChart() {
  const entries = Object.entries(DATA.seasonal_avg_by_month);
  const ctx = document.getElementById("seasonalChart").getContext("2d");
  seasonalChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: entries.map(([m]) => m.slice(0, 3)),
      datasets: [{
        label: "Average Rainfall (mm)",
        data: entries.map(([, v]) => v),
        backgroundColor: BLUE.bar1,
        borderRadius: 6,
        maxBarThickness: 34,
      }],
    },
    options: baseBarOptions("Average Monthly Rainfall (1982–present)", "mm"),
  });
}

function renderModelTable() {
  const tbody = document.querySelector("#modelTable tbody");
  const rows = Object.entries(DATA.model_comparison)
    .sort((a, b) => a[1].RMSE - b[1].RMSE);

  tbody.innerHTML = rows.map(([name, m]) => `
    <tr class="${name === DATA.best_model ? "best-row" : ""}">
      <td>${name}${name === DATA.best_model ? " 🏆" : ""}</td>
      <td>${m.MAE}</td>
      <td>${m.RMSE}</td>
      <td>${m.R2}</td>
    </tr>
  `).join("");
}

document.addEventListener("DOMContentLoaded", init);
