"""
CineSense — AI Review Sentiment Analyzer
Streamlit frontend, styled to match the cinema-marquee theme.
Talks to the existing FastAPI backend (RNN / LSTM / GRU models).
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE = "https://sentiment-classifier-lstm-gru-rnn-production.up.railway.app"

st.set_page_config(
    page_title="CineSense — AI Review Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Theme (colors match the original CSS variables)
# ---------------------------------------------------------------------------
GOLD = "#e3b23c"
GOLD_DIM = "#a9832c"
CRIMSON = "#d5384a"
EMERALD = "#3fae74"
VOID = "#0c0a11"
SURFACE_1 = "#1c1824"
SURFACE_0 = "#14111a"
TEXT_PRIMARY = "#f3efe6"
TEXT_SECONDARY = "#b3a99a"
TEXT_MUTED = "#766c60"
BORDER = "rgba(227, 178, 60, 0.14)"

st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
.stApp {{
    background: {VOID};
    color: {TEXT_PRIMARY};
    font-family: 'Space Grotesk', sans-serif;
}}
#MainMenu, header, footer {{visibility: hidden;}}

.cs-brand {{
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 6px;
}}
.cs-brand-mark {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 22px;
    width: 46px; height: 46px;
    border-radius: 50%;
    border: 2px solid {GOLD};
    color: {GOLD};
    display: flex; align-items: center; justify-content: center;
}}
.cs-brand h1 {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 34px;
    letter-spacing: 1px;
    margin: 0;
}}
.cs-brand .accent {{ color: {GOLD}; }}
.cs-tagline {{
    font-size: 13px;
    color: {TEXT_SECONDARY};
    margin: 2px 0 0;
}}
.cs-eyebrow {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: {GOLD};
}}
.cs-panel {{
    background: {SURFACE_1};
    border: 1px solid {BORDER};
    border-radius: 20px;
    padding: 26px;
}}
.cs-heading {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 28px;
    font-weight: 400;
    letter-spacing: 0.5px;
    margin: 6px 0 18px;
}}
.stTextArea textarea {{
    background: {SURFACE_0} !important;
    color: {TEXT_PRIMARY} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 12px !important;
    font-family: 'Space Grotesk', sans-serif !important;
}}
.stSelectbox div[data-baseweb="select"] > div {{
    background: {SURFACE_0} !important;
    color: {TEXT_PRIMARY} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 12px !important;
}}
.stButton > button {{
    background: linear-gradient(135deg, {GOLD}, {GOLD_DIM});
    color: #221a06;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 17px;
    letter-spacing: 1px;
    border: none;
    border-radius: 12px;
    padding: 10px 26px;
    box-shadow: 0 8px 20px -8px rgba(227, 178, 60, 0.55);
}}
.stButton > button:hover {{
    transform: translateY(-1px);
}}
.cs-verdict {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 32px;
    letter-spacing: 1px;
}}
.cs-verdict.positive {{ color: {GOLD}; }}
.cs-verdict.negative {{ color: {CRIMSON}; }}
.cs-model-tag {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: {TEXT_SECONDARY};
    border: 1px solid {BORDER};
    padding: 4px 10px;
    border-radius: 100px;
}}
.cs-inference {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: {TEXT_MUTED};
    text-align: center;
    margin-top: 12px;
}}
.cs-badge {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    padding: 6px 12px;
    border-radius: 100px;
    border: 1px solid {BORDER};
    color: {TEXT_SECONDARY};
    background: {SURFACE_1};
    display: inline-block;
    margin-right: 6px;
}}
.cs-badge.online {{ color: {EMERALD}; border-color: rgba(63,174,116,0.35); }}
.cs-stat-card {{
    background: {SURFACE_1};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 16px 18px;
    text-align: center;
}}
.cs-stat-card .label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: {TEXT_MUTED};
    display: block;
    margin-bottom: 6px;
}}
.cs-stat-card .value {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 26px;
}}
.cs-stat-card.best {{ border-color: rgba(227,178,60,0.32); }}
.cs-stat-card.best .value {{ color: {GOLD}; }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="cs-brand">
    <div class="cs-brand-mark">CS</div>
    <div>
        <h1>Cine<span class="accent">Sense</span></h1>
        <p class="cs-tagline">Deep learning sentiment analysis · RNN &middot; LSTM &middot; GRU</p>
    </div>
</div>
""", unsafe_allow_html=True)

# --- backend health badges ---
badge_placeholder = st.empty()

def load_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=8)
        r.raise_for_status()
        data = r.json()
        models = data.get("models_loaded", [])
        if not models:
            return "<span class='cs-badge'>backend online, no models found</span>"
        return "".join(
            f"<span class='cs-badge online'>&#9679; {m}</span>" for m in models
        )
    except Exception:
        return "<span class='cs-badge'>backend offline — check the API</span>"

badge_placeholder.markdown(load_health(), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Stage: input + result
# ---------------------------------------------------------------------------
col_input, col_result = st.columns([1.3, 1], gap="large")

with col_input:
    st.markdown("<div class='cs-panel'>", unsafe_allow_html=True)
    st.markdown("""
        <span class="cs-eyebrow">Now screening</span>
        <div class="cs-heading">Drop in a review. We'll read the room.</div>
    """, unsafe_allow_html=True)

    review_text = st.text_area(
        "Review",
        placeholder="e.g. The cinematography was breathtaking and the lead "
                    "performance carried every scene, even when the script "
                    "stumbled in the third act...",
        height=180,
        label_visibility="collapsed",
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        model_choice = st.selectbox(
            "Model",
            options=["Best model (auto)", "GRU", "LSTM", "RNN"],
        )
    with c2:
        st.write("")  # vertical spacer to align button
        analyze_clicked = st.button("Analyze sentiment", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

with col_result:
    st.markdown("<div class='cs-panel'>", unsafe_allow_html=True)

    if analyze_clicked:
        if not review_text.strip():
            st.error("Enter a review first.")
        else:
            with st.spinner("Reading the room…"):
                try:
                    payload = {"text": review_text.strip()}
                    if model_choice != "Best model (auto)":
                        payload["model"] = model_choice

                    resp = requests.post(f"{API_BASE}/predict", json=payload, timeout=20)
                    resp.raise_for_status()
                    data = resp.json()

                    is_positive = data["sentiment"].lower() == "positive"
                    verdict_class = "positive" if is_positive else "negative"
                    confidence_pct = round(data["confidence"] * 100)
                    pos_pct = round(data["probability_positive"] * 100)
                    neg_pct = round(data["probability_negative"] * 100)

                    st.markdown(f"""
                        <div class="cs-verdict {verdict_class}">{data['sentiment'].upper()}</div>
                        <span class="cs-model-tag">{data['model_used']}</span>
                    """, unsafe_allow_html=True)

                    # Gauge
                    gauge_color = GOLD if is_positive else CRIMSON
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=confidence_pct,
                        number={"suffix": "%", "font": {"color": TEXT_PRIMARY, "size": 34}},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": TEXT_SECONDARY},
                            "bar": {"color": gauge_color},
                            "bgcolor": SURFACE_0,
                            "borderwidth": 0,
                        },
                    ))
                    fig_gauge.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        font={"color": TEXT_SECONDARY, "family": "JetBrains Mono"},
                        height=220,
                        margin=dict(l=20, r=20, t=10, b=10),
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)

                    # Probability bars
                    fig_bars = go.Figure()
                    fig_bars.add_trace(go.Bar(
                        y=["Positive"], x=[pos_pct], orientation="h",
                        marker_color=GOLD, text=[f"{pos_pct}%"], textposition="outside",
                    ))
                    fig_bars.add_trace(go.Bar(
                        y=["Negative"], x=[neg_pct], orientation="h",
                        marker_color=CRIMSON, text=[f"{neg_pct}%"], textposition="outside",
                    ))
                    fig_bars.update_layout(
                        showlegend=False,
                        barmode="stack",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font={"color": TEXT_SECONDARY, "family": "JetBrains Mono"},
                        xaxis=dict(visible=False, range=[0, 100]),
                        height=140,
                        margin=dict(l=10, r=40, t=10, b=10),
                    )
                    st.plotly_chart(fig_bars, use_container_width=True)

                    st.markdown(
                        f"<div class='cs-inference'>Inference took {data.get('inference_time_ms', '—')} ms</div>",
                        unsafe_allow_html=True,
                    )

                except requests.exceptions.RequestException as e:
                    st.error(f"Could not reach the backend: {e}")
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
    else:
        st.markdown(f"""
            <div style="text-align:center; color:{TEXT_MUTED}; padding: 60px 0;">
                🎞️<br><br>
                <p>Your verdict will roll here.</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Model comparison section
# ---------------------------------------------------------------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <span class="cs-eyebrow">Behind the scenes</span>
    <div class="cs-heading" style="font-size:30px;">Three models, one dataset, one winner</div>
    <p style="color:#b3a99a; font-size:14px;">
        Trained on 50,000 IMDb reviews. Here's how Simple RNN, LSTM, and GRU stacked up on the held-out test set.
    </p>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)
def load_comparison():
    r = requests.get(f"{API_BASE}/model-comparison", timeout=10)
    r.raise_for_status()
    return r.json()

try:
    rows = load_comparison()

    models = [row["Model"] for row in rows]
    accuracy = [round(row["Accuracy"] * 100, 1) for row in rows]
    f1 = [round(row["F1 Score"] * 100, 1) for row in rows]
    train_time = [round(row["Training Time (s)"], 1) for row in rows]

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("<div class='cs-panel'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#b3a99a; font-size:14px; text-transform:uppercase;'>Accuracy &amp; F1 by model</h4>", unsafe_allow_html=True)
        fig_metrics = go.Figure()
        fig_metrics.add_trace(go.Bar(name="Accuracy", x=models, y=accuracy, marker_color=GOLD))
        fig_metrics.add_trace(go.Bar(name="F1 score", x=models, y=f1, marker_color=EMERALD))
        fig_metrics.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": TEXT_SECONDARY, "family": "Space Grotesk"},
            yaxis=dict(range=[0, 100], title="%"),
            legend=dict(orientation="h", y=1.15),
            margin=dict(l=10, r=10, t=10, b=10),
            height=280,
        )
        st.plotly_chart(fig_metrics, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with chart_col2:
        st.markdown("<div class='cs-panel'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#b3a99a; font-size:14px; text-transform:uppercase;'>Training time (seconds)</h4>", unsafe_allow_html=True)
        fig_time = go.Figure(go.Bar(x=models, y=train_time, marker_color=CRIMSON))
        fig_time.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": TEXT_SECONDARY, "family": "Space Grotesk"},
            margin=dict(l=10, r=10, t=10, b=10),
            height=280,
        )
        st.plotly_chart(fig_time, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Stat strip
    best_model = max(rows, key=lambda r: r["F1 Score"])["Model"]
    stat_cols = st.columns(len(rows))
    for col, row in zip(stat_cols, rows):
        is_best = row["Model"] == best_model
        best_class = "best" if is_best else ""
        label = f"{row['Model']} · best" if is_best else row["Model"]
        with col:
            st.markdown(f"""
                <div class="cs-stat-card {best_class}">
                    <span class="label">{label}</span>
                    <span class="value">{row['Accuracy']*100:.1f}%</span>
                </div>
            """, unsafe_allow_html=True)

except Exception:
    st.markdown(
        f"<p style='color:{TEXT_MUTED}; font-family:JetBrains Mono; font-size:12px;'>"
        "Comparison data unavailable — copy results/model_comparison.csv from the "
        "notebook into backend/results/.</p>",
        unsafe_allow_html=True,
    )

st.markdown(f"""
<div style="text-align:center; color:{TEXT_MUTED}; font-family:'JetBrains Mono'; font-size:11px; padding:20px 0 0;">
    CineSense · Sentiment analysis backend powered by FastAPI + TensorFlow/Keras
</div>
<div style="text-align:center; color:{GOLD}; font-family:'JetBrains Mono'; font-size:11px; padding:6px 0 20px;">
    Built by Fadia Iftikhar
</div>
""", unsafe_allow_html=True)