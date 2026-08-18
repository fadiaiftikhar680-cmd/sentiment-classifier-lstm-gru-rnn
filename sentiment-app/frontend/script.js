// ============================================================================
// CineSense frontend logic
// Talks to the FastAPI backend (main.py) running on http://localhost:8000
// ============================================================================

const API_BASE = "https://sentiment-classifier-lstm-gru-rnn-production.up.railway.app";
const els = {
  input: document.getElementById("review-input"),
  modelSelect: document.getElementById("model-select"),
  analyzeBtn: document.getElementById("analyze-btn"),
  btnLabel: document.querySelector(".btn-label"),
  btnSpinner: document.querySelector(".btn-spinner"),
  errorMsg: document.getElementById("error-msg"),
  placeholder: document.getElementById("result-placeholder"),
  resultContent: document.getElementById("result-content"),
  verdictLabel: document.getElementById("verdict-label"),
  verdictModel: document.getElementById("verdict-model"),
  gaugeArc: document.getElementById("gauge-arc"),
  gaugeNumber: document.getElementById("gauge-number"),
  barPositive: document.getElementById("bar-positive"),
  barNegative: document.getElementById("bar-negative"),
  pctPositive: document.getElementById("pct-positive"),
  pctNegative: document.getElementById("pct-negative"),
  inferenceNote: document.getElementById("inference-note"),
  modelBadges: document.getElementById("model-badges"),
  statStrip: document.getElementById("stat-strip"),
};

const GAUGE_CIRCUMFERENCE = 283; // matches the arc path length approximation

function setLoading(isLoading) {
  els.analyzeBtn.disabled = isLoading;
  els.btnLabel.textContent = isLoading ? "Reading the room…" : "Analyze sentiment";
  els.btnSpinner.hidden = !isLoading;
}

function showError(message) {
  els.errorMsg.textContent = message;
  els.errorMsg.hidden = false;
}

function clearError() {
  els.errorMsg.hidden = true;
}

function animateGauge(confidencePct, isPositive) {
  const offset = GAUGE_CIRCUMFERENCE - (GAUGE_CIRCUMFERENCE * confidencePct) / 100;
  els.gaugeArc.style.stroke = isPositive ? "var(--gold)" : "var(--crimson)";
  requestAnimationFrame(() => {
    els.gaugeArc.style.strokeDashoffset = offset;
  });

  let current = 0;
  const target = Math.round(confidencePct);
  const step = () => {
    current += Math.max(1, Math.round((target - current) / 6));
    if (current >= target) {
      els.gaugeNumber.textContent = `${target}%`;
      return;
    }
    els.gaugeNumber.textContent = `${current}%`;
    requestAnimationFrame(step);
  };
  step();
}

async function analyze() {
  const text = els.input.value.trim();
  clearError();

  if (!text) {
    showError("Enter a review first.");
    return;
  }

  setLoading(true);

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        model: els.modelSelect.value || undefined,
      }),
    });

    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `Request failed (${res.status})`);
    }

    const data = await res.json();
    renderResult(data);
  } catch (err) {
    showError(err.message || "Could not reach the backend. Is it running on :8000?");
  } finally {
    setLoading(false);
  }
}

function renderResult(data) {
  els.placeholder.hidden = true;
  els.resultContent.hidden = false;

  const isPositive = data.sentiment.toLowerCase() === "positive";
  els.verdictLabel.textContent = data.sentiment.toUpperCase();
  els.verdictLabel.className = `verdict-label ${isPositive ? "positive" : "negative"}`;
  els.verdictModel.textContent = data.model_used;

  const confidencePct = data.confidence * 100;
  animateGauge(confidencePct, isPositive);

  const posPct = Math.round(data.probability_positive * 100);
  const negPct = Math.round(data.probability_negative * 100);
  els.barPositive.style.width = `${posPct}%`;
  els.barNegative.style.width = `${negPct}%`;
  els.pctPositive.textContent = `${posPct}%`;
  els.pctNegative.textContent = `${negPct}%`;

  els.inferenceNote.textContent = `Inference took ${data.inference_time_ms} ms`;
}

els.analyzeBtn.addEventListener("click", analyze);
els.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) analyze();
});

// ----------------------------------------------------------------------
// Backend status badges
// ----------------------------------------------------------------------
async function loadBadges() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    els.modelBadges.innerHTML = "";

    if (!data.models_loaded || data.models_loaded.length === 0) {
      els.modelBadges.innerHTML = `<span class="badge-loading">backend online, no models found</span>`;
      return;
    }

    data.models_loaded.forEach((name) => {
      const badge = document.createElement("span");
      badge.className = "badge-model online";
      badge.innerHTML = `<span class="dot"></span>${name}`;
      els.modelBadges.appendChild(badge);
    });
  } catch (err) {
    els.modelBadges.innerHTML = `<span class="badge-loading">backend offline — start uvicorn on :8000</span>`;
  }
}

// ----------------------------------------------------------------------
// Model comparison charts
// ----------------------------------------------------------------------
async function loadComparison() {
  try {
    const res = await fetch(`${API_BASE}/model-comparison`);
    if (!res.ok) throw new Error("no comparison data");
    const rows = await res.json();
    renderCharts(rows);
    renderStatStrip(rows);
  } catch (err) {
    document.getElementById("comparison-section").innerHTML +=
      `<p style="color: var(--text-muted); font-family: var(--font-mono); font-size: 12px;">
        Comparison data unavailable — copy results/model_comparison.csv from the notebook into backend/results/.
      </p>`;
  }
}

function renderStatStrip(rows) {
  const best = rows.slice().sort((a, b) => b["F1 Score"] - a["F1 Score"])[0];
  els.statStrip.innerHTML = "";
  rows.forEach((row) => {
    const card = document.createElement("div");
    card.className = "stat-card" + (row.Model === best.Model ? " best" : "");
    card.innerHTML = `
      <span class="stat-label">${row.Model}${row.Model === best.Model ? " · best" : ""}</span>
      <span class="stat-value">${(row.Accuracy * 100).toFixed(1)}%</span>
    `;
    els.statStrip.appendChild(card);
  });
}

function renderCharts(rows) {
  const labels = rows.map((r) => r.Model);
  const gold = "#e3b23c";
  const crimson = "#d5384a";
  const emerald = "#3fae74";

  const commonOpts = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: "#b3a99a", font: { family: "Space Grotesk", size: 11 } } },
    },
    scales: {
      x: { ticks: { color: "#b3a99a" }, grid: { color: "rgba(227,178,60,0.08)" } },
      y: { ticks: { color: "#b3a99a" }, grid: { color: "rgba(227,178,60,0.08)" }, beginAtZero: true },
    },
  };

  new Chart(document.getElementById("metricsChart"), {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Accuracy",
          data: rows.map((r) => +(r.Accuracy * 100).toFixed(1)),
          backgroundColor: gold,
          borderRadius: 6,
        },
        {
          label: "F1 score",
          data: rows.map((r) => +(r["F1 Score"] * 100).toFixed(1)),
          backgroundColor: emerald,
          borderRadius: 6,
        },
      ],
    },
    options: {
      ...commonOpts,
      scales: {
        ...commonOpts.scales,
        y: { ...commonOpts.scales.y, max: 100, title: { display: true, text: "%", color: "#766c60" } },
      },
    },
  });

  new Chart(document.getElementById("timeChart"), {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Training time (s)",
          data: rows.map((r) => +r["Training Time (s)"].toFixed(1)),
          backgroundColor: crimson,
          borderRadius: 6,
        },
      ],
    },
    options: commonOpts,
  });
}

loadBadges();
loadComparison();
