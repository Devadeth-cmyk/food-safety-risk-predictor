from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk


# ============================================================
# Chicago Restaurant Health Inspection Risk Dashboard
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
EXPORT_DIR = ARTIFACTS_DIR / "dashboard_exports"
SHAP_DIR = ARTIFACTS_DIR / "shap_analysis"

# Design tokens
RED, AMBER, GREEN = "#EF4444", "#F59E0B", "#10B981"
ACCENT, ACCENT_2 = "#6366F1", "#06B6D4"
TIER_COLORS = {
    "Red - highest 10%": RED,
    "Yellow - next 20%": AMBER,
    "Green - remaining 70%": GREEN,
}
TIER_ORDER = list(TIER_COLORS.keys())

st.set_page_config(
    page_title="Chicago Restaurant Health Inspection Risk System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
}

/* ---------- Layout ---------- */
.block-container {
    padding-top: 2.6rem;
    padding-bottom: 2rem;
    padding-left: 2.2rem;
    padding-right: 2.2rem;
    max-width: 1500px;
}
div[data-testid="stVerticalBlock"] { gap: 1rem; }
header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }

h2, h3 { letter-spacing: -0.01em; font-weight: 700 !important; }
h2 { margin-top: .6rem !important; margin-bottom: .4rem !important; }
h3 { margin-top: .4rem !important; margin-bottom: .4rem !important; }

/* ---------- Hero ---------- */
.hero {
    position: relative;
    overflow: hidden;
    padding: 28px 34px;
    border-radius: 20px;
    margin-bottom: 8px;
    background:
        radial-gradient(900px 260px at 100% 0%, rgba(6,182,212,.28), transparent 60%),
        radial-gradient(700px 300px at 0% 100%, rgba(99,102,241,.35), transparent 60%),
        linear-gradient(135deg, #0F172A 0%, #1E1B4B 100%);
    border: 1px solid rgba(255,255,255,.08);
    box-shadow: 0 10px 30px rgba(15,23,42,.35);
}
.hero-badge {
    display: inline-block;
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .14em;
    color: #A5F3FC;
    background: rgba(6,182,212,.14);
    border: 1px solid rgba(6,182,212,.4);
    padding: 4px 12px;
    border-radius: 999px;
    margin-bottom: 12px;
}
.hero h1 {
    margin: 0 0 6px 0 !important;
    padding: 0 !important;
    font-size: 2.15rem !important;
    font-weight: 800 !important;
    line-height: 1.15 !important;
    color: #F8FAFC !important;
    letter-spacing: -0.02em;
}
.hero p {
    margin: 0;
    color: #CBD5E1;
    font-size: 1rem;
    max-width: 780px;
}
.hero-chips { margin-top: 16px; display: flex; gap: 10px; flex-wrap: wrap; }
.chip {
    font-size: .8rem; font-weight: 600; color: #E2E8F0;
    background: rgba(255,255,255,.08);
    border: 1px solid rgba(255,255,255,.12);
    padding: 5px 12px; border-radius: 999px;
}

/* ---------- KPI cards ---------- */
.kpi {
    position: relative;
    padding: 18px 20px 16px 20px;
    border-radius: 16px;
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,.22);
    box-shadow: 0 4px 12px rgba(0,0,0,.10);
    transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
    overflow: hidden;
}
.kpi::before {
    content: "";
    position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
    background: var(--accent);
}
.kpi:hover {
    transform: translateY(-4px);
    box-shadow: 0 14px 28px rgba(0,0,0,.22);
    border-color: var(--accent);
}
.kpi-top { display: flex; align-items: center; justify-content: space-between; }
.kpi-label {
    font-size: .82rem; font-weight: 700; opacity: .72;
    text-transform: uppercase; letter-spacing: .06em;
}
.kpi-icon { font-size: 1.25rem; }
.kpi-value { font-size: 2.1rem; font-weight: 800; line-height: 1.2; margin-top: 6px; }
.kpi-sub { font-size: .82rem; opacity: .65; margin-top: 2px; }

/* ---------- Native metric cards (used in other tabs) ---------- */
div[data-testid="stMetric"], div[data-testid="metric-container"] {
    background-color: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,.22);
    padding: 14px 18px;
    border-radius: 14px;
    box-shadow: 0 4px 10px rgba(0,0,0,.08);
    transition: transform .2s ease, box-shadow .2s ease;
}
div[data-testid="stMetric"]:hover, div[data-testid="metric-container"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 22px rgba(0,0,0,.18);
}
div[data-testid="stMetricValue"] { font-size: 1.9rem !important; font-weight: 800 !important; }
div[data-testid="stMetricLabel"] { font-size: .9rem !important; font-weight: 700 !important; opacity: .75; }

/* ---------- Tabs as pills ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    padding: 6px;
    border-radius: 14px;
    background: rgba(128,128,128,.10);
    border: 1px solid rgba(128,128,128,.18);
}
.stTabs [data-baseweb="tab"] {
    height: auto;
    padding: 9px 18px;
    border-radius: 10px;
    font-weight: 600;
    transition: background .2s ease;
}
.stTabs [data-baseweb="tab"]:hover { background: rgba(99,102,241,.14); }
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366F1, #06B6D4) !important;
    box-shadow: 0 4px 12px rgba(99,102,241,.4);
}
.stTabs [aria-selected="true"] p { color: #fff !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.1rem; }

/* ---------- Charts / tables ---------- */
div[data-testid="stPlotlyChart"], div[data-testid="stDeckGlJsonChart"] {
    border-radius: 14px;
    border: 1px solid rgba(128,128,128,.18);
    padding: 6px;
    background: var(--secondary-background-color);
}
div[data-testid="stDataFrame"] {
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,.2);
    overflow: hidden;
}

/* ---------- Buttons ---------- */
.stButton > button, .stDownloadButton > button {
    border-radius: 10px;
    font-weight: 600;
    border: 1px solid rgba(128,128,128,.3);
    transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-2px);
    border-color: #6366F1;
    color: #6366F1;
    box-shadow: 0 6px 14px rgba(99,102,241,.25);
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128,128,128,.2);
}
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
span[data-baseweb="tag"] { background-color: rgba(99,102,241,.92) !important; }

.brand { display: flex; align-items: center; gap: 14px; padding: 6px 0 10px 0; }
.brand-logo {
    width: 58px; height: 58px; flex: 0 0 58px;
    filter: drop-shadow(0 6px 14px rgba(99,102,241,.55));
    animation: float 4s ease-in-out infinite;
}
.brand-title { font-size: 1.12rem; font-weight: 800; line-height: 1.15; letter-spacing: -0.01em; }
.brand-sub { font-size: .74rem; opacity: .65; font-weight: 500; letter-spacing: .04em; }
.brand-pulse {
    stroke-dasharray: 90;
    stroke-dashoffset: 90;
    animation: pulse-draw 2.6s ease-in-out infinite;
}
@keyframes pulse-draw {
    0% { stroke-dashoffset: 90; }
    55% { stroke-dashoffset: 0; }
    100% { stroke-dashoffset: -90; }
}
@keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-3px); }
}

.side-title {
    font-size: .72rem; font-weight: 700; letter-spacing: .14em;
    text-transform: uppercase; opacity: .6; margin: 6px 0 4px 0;
}
.side-card {
    padding: 12px 14px; border-radius: 12px;
    background: rgba(99,102,241,.10);
    border: 1px solid rgba(99,102,241,.3);
    margin-top: 6px;
}
.side-card b { font-size: 1.25rem; }
.legend-row { display: flex; align-items: center; gap: 10px; margin: 6px 0; font-size: .9rem; font-weight: 600; }
.dot { width: 11px; height: 11px; border-radius: 50%; display: inline-block; }

/* ---------- Badges ---------- */
.badge {
    display: inline-block; padding: 4px 14px; border-radius: 999px;
    font-weight: 700; font-size: .85rem; color: #fff;
}
.map-legend { display: flex; gap: 18px; flex-wrap: wrap; font-size: .85rem; font-weight: 600; opacity: .9; }

.footer-note {
    margin-top: 10px; padding: 14px 18px; border-radius: 12px; font-size: .85rem;
    border: 1px dashed rgba(128,128,128,.4); opacity: .8;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# Helpers
# ============================================================

def first_existing(*paths):
    for path in paths:
        if path and path.exists():
            return path
    return None


@st.cache_data
def load_csv(path_string):
    return pd.read_csv(path_string, low_memory=False)


def polish(fig, height=380, legend=False):
    """Apply one consistent, clean look to every Plotly chart."""
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=56, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=13),
        title=dict(font=dict(size=17), x=0.01),
        showlegend=legend,
        hoverlabel=dict(font_size=13, font_family="Inter, sans-serif"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,.18)", zeroline=False)
    return fig


def kpi_card(col, icon, label, value, sub, accent):
    col.markdown(
        f'<div class="kpi" style="--accent:{accent}">'
        f'<div class="kpi-top"><span class="kpi-label">{label}</span>'
        f'<span class="kpi-icon">{icon}</span></div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def tier_badge(tier):
    color = TIER_COLORS.get(str(tier), "#64748B")
    return f'<span class="badge" style="background:{color}">{tier}</span>'


def normalize_queue(df):
    """Normalize queue column names for consistent dashboard processing."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    aliases = {
        "restaurant_id": "Restaurant_ID",
        "priority_rank": "Priority_Rank",
        "priority_tier": "Relative_Risk_Tier",
        "relative_risk_tier": "Relative_Risk_Tier",
        "predicted_next_failure_probability": "Predicted_Failure_Probability",
        "predicted_failure_probability": "Predicted_Failure_Probability",
        "latest_inspection_date": "Inspection Date",
        "inspection_date": "Inspection Date",
        "latest_inspection_result": "Results",
        "current_result_label": "Results_Label",
        "violation_count": "Violation_Count",
        "hazard_count": "Hazard_Count",
        "latitude": "Latitude",
        "longitude": "Longitude",
        "previous_failure_count": "Previous_Failure_Count",
        "historical_failure_rate": "Historical_Failure_Rate",
    }

    lower_to_original = {str(c).strip().lower(): c for c in df.columns}

    for source, target in aliases.items():
        if target not in df.columns and source in lower_to_original:
            original = lower_to_original[source]
            df.rename(columns={original: target}, inplace=True)

    # Support exports that store probabilities as percentages.
    lower_to_original = {str(c).strip().lower(): c for c in df.columns}

    if "Predicted_Failure_Probability" not in df.columns:
        pct_col = lower_to_original.get("predicted_next_failure_probability_pct")
        if pct_col:
            df["Predicted_Failure_Probability"] = (
                pd.to_numeric(df[pct_col], errors="coerce") / 100
            )

    if "Relative_Risk_Tier" in df.columns:
        def normalize_tier(value):
            text = str(value).strip()
            low = text.lower()

            if (
                text.startswith("P1")
                or "top 5%" in low
                or "top 10%" in low
                or "highest" in low
            ):
                return "Red - highest 10%"

            if (
                text.startswith("P2")
                or text.startswith("P3")
                or "elevated" in low
                or "high" in low
            ):
                return "Yellow - next 20%"

            if text.startswith("P4") or "standard" in low:
                return "Green - remaining 70%"

            return text

        df["Relative_Risk_Tier"] = df["Relative_Risk_Tier"].map(normalize_tier)

    if "Predicted_Failure_Probability" in df.columns:
        df["Predicted_Failure_Probability"] = pd.to_numeric(
            df["Predicted_Failure_Probability"], errors="coerce"
        )

        valid = df["Predicted_Failure_Probability"].dropna()

        if not valid.empty and valid.max() > 1:
            df["Predicted_Failure_Probability"] /= 100

    for col in ["Latitude", "Longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    if "Priority_Rank" in df.columns:
        df["Priority_Rank"] = pd.to_numeric(df["Priority_Rank"], errors="coerce")

    if "Restaurant_ID" in df.columns:
        df["Restaurant_ID"] = df["Restaurant_ID"].astype("string")

    return df


# ============================================================
# Load project artifacts
# ============================================================

queue_path = first_existing(
    EXPORT_DIR / "inspection_priority_list_full.csv",
    BASE_DIR / "inspection_priority_list_full.csv",
    EXPORT_DIR / "inspection_priority_list.csv",
    BASE_DIR / "inspection_priority_list.csv",
)

metrics_path = first_existing(
    ARTIFACTS_DIR / "test_metrics.json",
    BASE_DIR / "test_metrics.json",
)

shap_path = first_existing(
    EXPORT_DIR / "shap_feature_importance.csv",
    SHAP_DIR / "shap_feature_importance.csv",
    BASE_DIR / "shap_feature_importance.csv",
)

community_summary_path = first_existing(
    EXPORT_DIR / "neighborhood_risk_summary.csv",
    BASE_DIR / "neighborhood_risk_summary.csv",
)

shap_img = first_existing(
    SHAP_DIR / "shap_summary_bar.png",
    BASE_DIR / "artifacts" / "shap_analysis" / "shap_summary_bar.png",
)


if queue_path is None:
    st.error(
        "Restaurant priority queue not found. Expected one of:\n\n"
        "- `artifacts/dashboard_exports/inspection_priority_list_full.csv`\n"
        "- `inspection_priority_list_full.csv`\n"
        "- `artifacts/dashboard_exports/inspection_priority_list.csv`\n"
        "- `inspection_priority_list.csv`"
    )
    st.stop()


queue = normalize_queue(load_csv(str(queue_path)))

if "Predicted_Failure_Probability" not in queue.columns:
    st.error(
        "The priority-list CSV does not contain a recognized predicted "
        "failure probability column."
    )
    st.write("Columns found:", list(queue.columns))
    st.stop()

queue = queue.dropna(subset=["Predicted_Failure_Probability"]).copy()

if queue.empty:
    st.error("The priority-list file contains no usable risk predictions.")
    st.stop()

if "Relative_Risk_Tier" not in queue.columns:
    # Assign tiers based on risk ranking when no tier field is provided.
    queue = queue.sort_values("Predicted_Failure_Probability", ascending=False).copy()

    percentile = queue["Predicted_Failure_Probability"].rank(
        method="first", ascending=False, pct=True
    )

    queue["Relative_Risk_Tier"] = np.select(
        [percentile <= 0.10, percentile <= 0.30],
        ["Red - highest 10%", "Yellow - next 20%"],
        default="Green - remaining 70%",
    )

map_df = queue.dropna(subset=["Latitude", "Longitude"]).copy()
city_avg_risk = float(queue["Predicted_Failure_Probability"].mean())


# ============================================================
# Sidebar: brand + interactive filters
# ============================================================

tiers_found = queue["Relative_Risk_Tier"].dropna().unique().tolist()
tiers = [t for t in TIER_ORDER if t in tiers_found] + sorted(
    t for t in tiers_found if t not in TIER_ORDER
)

min_p = max(0.0, float(queue["Predicted_Failure_Probability"].min()))
max_p = min(1.0, float(queue["Predicted_Failure_Probability"].max()))

if max_p < min_p:
    min_p, max_p = 0.0, 1.0

# Persist filter state so presets / reset buttons can drive the widgets.
st.session_state.setdefault("selected_tiers", tiers)
st.session_state.setdefault("prob_range", (min_p, max_p))
st.session_state.setdefault("search_text", "")


def _select_tiers(keywords):
    chosen = [t for t in tiers if any(k in t for k in keywords)]
    st.session_state["selected_tiers"] = chosen or tiers


def _reset_filters():
    st.session_state["selected_tiers"] = tiers
    st.session_state["prob_range"] = (min_p, max_p)
    st.session_state["search_text"] = ""


BRAND_HTML = (
    '<div class="brand">'
    '<svg class="brand-logo" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">'
    '<defs><linearGradient id="g1" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#6366F1"/><stop offset="1" stop-color="#06B6D4"/>'
    '</linearGradient></defs>'
    '<path d="M32 3 L55 11 V31 C55 45 45 55 32 61 C19 55 9 45 9 31 V11 Z" fill="url(#g1)"/>'
    '<path d="M32 8 L50 14.5 V31 C50 42 42 50.5 32 55.5 C22 50.5 14 42 14 31 V14.5 Z" '
    'fill="none" stroke="rgba(255,255,255,.35)" stroke-width="1.2"/>'
    '<polyline class="brand-pulse" points="17,33 25,33 29,22 36,44 40,33 47,33" '
    'fill="none" stroke="#FFFFFF" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>'
    '</svg>'
    '<div><div class="brand-title">Inspection Risk</div>'
    '<div class="brand-sub">CHICAGO · FOOD SAFETY AI</div></div>'
    '</div>'
)

with st.sidebar:
    st.markdown(BRAND_HTML, unsafe_allow_html=True)
    st.markdown('<div class="side-title">Quick views</div>', unsafe_allow_html=True)

    q1, q2, q3 = st.columns(3)
    q1.button(
        "🔴 Critical",
        on_click=_select_tiers,
        args=(["Red"],),
        use_container_width=True,
        help="Show only the highest-priority 10%",
    )
    q2.button(
        "🟠 Top 30%",
        on_click=_select_tiers,
        args=(["Red", "Yellow"],),
        use_container_width=True,
        help="Highest + elevated priority",
    )
    q3.button(
        "↺ Reset",
        on_click=_reset_filters,
        use_container_width=True,
        help="Clear all filters",
    )

    st.markdown('<div class="side-title">Filters</div>', unsafe_allow_html=True)

    selected_tiers = st.multiselect("Priority tiers", tiers, key="selected_tiers")

    prob_range = st.slider(
        "Risk probability range",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        key="prob_range",
    )

    search_text = st.text_input(
        "Search Restaurant ID",
        placeholder="e.g. 10452",
        key="search_text",
    )

    summary_slot = st.empty()

    st.markdown('<div class="side-title">Risk legend</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="legend-row"><span class="dot" style="background:{RED}"></span>Highest priority</div>'
        f'<div class="legend-row"><span class="dot" style="background:{AMBER}"></span>Elevated priority</div>'
        f'<div class="legend-row"><span class="dot" style="background:{GREEN}"></span>Standard priority</div>',
        unsafe_allow_html=True,
    )


filtered = queue.copy()

if selected_tiers:
    filtered = filtered[filtered["Relative_Risk_Tier"].isin(selected_tiers)]

filtered = filtered[filtered["Predicted_Failure_Probability"].between(*prob_range)]

if search_text.strip() and "Restaurant_ID" in filtered.columns:
    filtered = filtered[
        filtered["Restaurant_ID"].astype(str).str.contains(
            search_text.strip(), case=False, na=False
        )
    ]

# Live selection summary inside the sidebar
share = len(filtered) / len(queue)
with summary_slot.container():
    st.markdown(
        f'<div class="side-card">Matching facilities<br>'
        f'<b>{len(filtered):,}</b> <span style="opacity:.7">of {len(queue):,} '
        f'({share:.0%})</span></div>',
        unsafe_allow_html=True,
    )
    st.progress(min(max(share, 0.0), 1.0))


# ============================================================
# Hero + KPI cards
# ============================================================

st.markdown(
    '<div class="hero">'
    '<div class="hero-badge">AI-POWERED INSPECTION PRIORITISATION</div>'
    '<h1>Chicago Restaurant Health Inspection Risk System</h1>'
    '<p>Predicts which restaurants are most likely to fail their next health '
    'inspection, so inspectors can focus on the places that matter most.</p>'
    '<div class="hero-chips">'
    f'<span class="chip">📍 {len(queue):,} facilities scored</span>'
    f'<span class="chip">🗺️ {len(map_df):,} mapped</span>'
    f'<span class="chip">🎯 Avg risk {city_avg_risk:.1%}</span>'
    '</div></div>',
    unsafe_allow_html=True,
)

high_risk_count = int((queue["Relative_Risk_Tier"] == "Red - highest 10%").sum())

k1, k2, k3, k4 = st.columns(4)
kpi_card(k1, "🏪", "Facilities scored", f"{len(queue):,}", "Total in the priority list", ACCENT)
kpi_card(
    k2, "🚨", "Highest-priority facilities", f"{high_risk_count:,}",
    f"{high_risk_count / len(queue):.1%} of all facilities", RED,
)
kpi_card(k3, "📊", "Average predicted risk", f"{city_avg_risk:.1%}", "Citywide mean probability", AMBER)
kpi_card(
    k4, "🎯", "In current filter", f"{len(filtered):,}",
    f"{share:.1%} of facilities selected", ACCENT_2,
)

st.write("")

overview_tab, map_tab, queue_tab, report_tab, community_tab, model_tab = st.tabs([
    "📈 Overview",
    "🗺️ Risk Map",
    "📋 Inspector Queue",
    "🧾 Report Card",
    "🏘️ Community",
    "🤖 AI Model",
])


# ============================================================
# Overview
# ============================================================

with overview_tab:
    head_l, head_r = st.columns([3, 2])
    head_l.subheader("Citywide risk distribution")
    with head_r:
        ov1, ov2 = st.columns(2)
        scope = ov1.radio("Data scope", ["Citywide", "Current filter"], horizontal=True)
        chart_style = ov2.radio("Tier chart", ["Donut", "Bar"], horizontal=True)

    scope_df = queue if scope == "Citywide" else filtered

    if scope_df.empty:
        st.warning("No restaurants match the current filters.")
    else:
        left, right = st.columns(2)

        with left:
            tier_counts = (
                scope_df["Relative_Risk_Tier"]
                .value_counts()
                .rename_axis("Risk tier")
                .reset_index(name="Restaurants")
            )

            if chart_style == "Donut":
                fig = px.pie(
                    tier_counts,
                    names="Risk tier",
                    values="Restaurants",
                    hole=0.62,
                    color="Risk tier",
                    color_discrete_map=TIER_COLORS,
                    title="Facilities by priority tier",
                )
                fig.update_traces(
                    textinfo="percent",
                    pull=[0.03] * len(tier_counts),
                    marker=dict(line=dict(color="rgba(0,0,0,0)", width=0)),
                    hovertemplate="<b>%{label}</b><br>%{value:,} restaurants<br>%{percent}<extra></extra>",
                )
                fig.add_annotation(
                    text=f"<b>{len(scope_df):,}</b><br>facilities",
                    showarrow=False,
                    font=dict(size=18),
                )
                polish(fig, legend=True)
                fig.update_layout(
                    legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center")
                )
            else:
                fig = px.bar(
                    tier_counts,
                    x="Risk tier",
                    y="Restaurants",
                    color="Risk tier",
                    color_discrete_map=TIER_COLORS,
                    title="Facilities by priority tier",
                    text="Restaurants",
                )
                fig.update_traces(
                    texttemplate="%{text:,}",
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>%{y:,} restaurants<extra></extra>",
                )
                polish(fig)

            st.plotly_chart(fig, use_container_width=True)

        with right:
            fig = px.histogram(
                scope_df,
                x="Predicted_Failure_Probability",
                color="Relative_Risk_Tier",
                color_discrete_map=TIER_COLORS,
                category_orders={"Relative_Risk_Tier": TIER_ORDER},
                nbins=30,
                title="Distribution of predicted failure probabilities",
                labels={
                    "Predicted_Failure_Probability": "Predicted failure probability",
                    "Relative_Risk_Tier": "Tier",
                },
            )
            fig.update_traces(marker_line_width=0)
            fig.update_xaxes(tickformat=".0%")
            fig.update_layout(bargap=0.04, yaxis_title="Restaurants")
            polish(fig, legend=True)
            fig.update_layout(
                legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center", title=None)
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Top 10 highest-risk facilities")
        top_cols = [
            c for c in ["Priority_Rank", "Restaurant_ID", "Relative_Risk_Tier",
                        "Predicted_Failure_Probability", "Violation_Count"]
            if c in scope_df.columns
        ]
        top10 = scope_df.nlargest(10, "Predicted_Failure_Probability")[top_cols].copy()
        top10["Predicted_Failure_Probability"] = top10["Predicted_Failure_Probability"] * 100

        st.dataframe(
            top10,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Priority_Rank": st.column_config.NumberColumn("Rank", format="%d"),
                "Restaurant_ID": st.column_config.TextColumn("Restaurant ID"),
                "Relative_Risk_Tier": st.column_config.TextColumn("Tier"),
                "Predicted_Failure_Probability": st.column_config.ProgressColumn(
                    "Predicted failure risk",
                    format="%.1f%%",
                    min_value=0.0,
                    max_value=100.0,
                ),
                "Violation_Count": st.column_config.NumberColumn("Violations", format="%d"),
            },
        )


# ============================================================
# Risk Map
# ============================================================

with map_tab:
    st.subheader("Geographic distribution of predicted risk")

    if map_df.empty:
        st.warning(
            "No valid Latitude/Longitude coordinates were found "
            "in the priority-list CSV."
        )
    else:
        c1, c2, c3 = st.columns([2, 1.4, 1.4])

        mode = c1.radio(
            "Map mode",
            ["Standard Scatter (Points)", "3D Hexagon Density (Clusters)"],
            horizontal=True,
        )
        is_3d = "3D" in mode

        if is_3d:
            size_value = c2.slider("Hexagon radius (m)", 100, 800, 250, 50)
        else:
            size_value = c2.slider("Point size", 30, 300, 80, 10)

        top_choice = c3.select_slider(
            "Facilities shown",
            options=["Top 500", "Top 2,000", "Top 5,000", "All"],
            value="All",
        )

        map_filtered = map_df.copy()

        if selected_tiers:
            map_filtered = map_filtered[map_filtered["Relative_Risk_Tier"].isin(selected_tiers)]

        map_filtered = map_filtered[
            map_filtered["Predicted_Failure_Probability"].between(*prob_range)
        ]

        if search_text.strip() and "Restaurant_ID" in map_filtered.columns:
            map_filtered = map_filtered[
                map_filtered["Restaurant_ID"].astype(str).str.contains(
                    search_text.strip(), case=False, na=False
                )
            ]

        top_n = {"Top 500": 500, "Top 2,000": 2000, "Top 5,000": 5000}.get(top_choice)
        if top_n:
            map_filtered = map_filtered.nlargest(top_n, "Predicted_Failure_Probability")

        if map_filtered.empty:
            st.warning("No restaurants match the current filters.")
        else:
            view = pdk.ViewState(
                latitude=float(map_filtered["Latitude"].median()),
                longitude=float(map_filtered["Longitude"].median()),
                zoom=10.5,
                pitch=45 if is_3d else 0,
                bearing=-8,
            )

            if is_3d:
                layer = pdk.Layer(
                    "HexagonLayer",
                    data=map_filtered,
                    get_position="[Longitude, Latitude]",
                    radius=size_value,
                    elevation_scale=8,
                    elevation_range=[0, 600],
                    extruded=True,
                    coverage=0.88,
                    pickable=True,
                    get_elevation_weight="Predicted_Failure_Probability",
                    color_range=[
                        [16, 185, 129, 160],
                        [245, 158, 11, 200],
                        [239, 68, 68, 240],
                    ],
                )

                tooltip = {"html": "<b>Facilities in hex:</b> {elevationValue}"}

            else:
                def color_row(row):
                    tier = str(row.get("Relative_Risk_Tier", ""))

                    if "Red" in tier:
                        return [239, 68, 68, 220]
                    if "Yellow" in tier:
                        return [245, 158, 11, 220]
                    return [16, 185, 129, 160]

                map_filtered = map_filtered.copy()
                map_filtered["color"] = map_filtered.apply(color_row, axis=1)
                map_filtered["Risk_pct"] = (
                    (map_filtered["Predicted_Failure_Probability"] * 100)
                    .round(1)
                    .astype(str) + "%"
                )

                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=map_filtered,
                    get_position="[Longitude, Latitude]",
                    get_radius=size_value,
                    get_fill_color="color",
                    pickable=True,
                    auto_highlight=True,
                )

                tooltip = {
                    "html": (
                        "<b>ID:</b> {Restaurant_ID}<br/>"
                        "<b>Risk:</b> {Risk_pct}<br/>"
                        "<b>Tier:</b> {Relative_Risk_Tier}"
                    ),
                    "style": {
                        "backgroundColor": "#0F172A",
                        "color": "white",
                        "borderRadius": "8px",
                        "padding": "8px 10px",
                    },
                }

            st.pydeck_chart(
                pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip)
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("Mapped restaurants", f"{len(map_filtered):,}")
            m2.metric(
                "Average risk in view",
                f"{map_filtered['Predicted_Failure_Probability'].mean():.1%}",
            )
            m3.metric(
                "Highest-priority in view",
                f"{int((map_filtered['Relative_Risk_Tier'] == 'Red - highest 10%').sum()):,}",
            )

            st.markdown(
                f'<div class="map-legend">'
                f'<span><span class="dot" style="background:{RED}"></span> Highest priority</span>'
                f'<span><span class="dot" style="background:{AMBER}"></span> Elevated priority</span>'
                f'<span><span class="dot" style="background:{GREEN}"></span> Standard priority</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ============================================================
# Inspector Queue
# ============================================================

with queue_tab:
    st.subheader("Inspector Priority Queue")
    st.caption(
        "Restaurants are ordered by predicted failure probability. "
        "Use the sidebar filters to narrow the queue."
    )

    # Exact columns and display order requested
    queue_columns = [
        "Priority_Rank",
        "Restaurant_ID",
        "Relative_Risk_Tier",
        "Predicted_Failure_Probability",
        "Inspection Date",
        "Violation_Count",
        "Latitude",
        "Longitude",
        "Results_Label",
    ]

    # Map alternative source column names to dashboard column names
    column_aliases = {
        "priority_rank": "Priority_Rank",
        "restaurant_id": "Restaurant_ID",
        "relative_risk_tier": "Relative_Risk_Tier",
        "priority_tier": "Relative_Risk_Tier",
        "predicted_failure_probability": "Predicted_Failure_Probability",
        "predicted_next_failure_probability": "Predicted_Failure_Probability",
        "inspection_date": "Inspection Date",
        "latest_inspection_date": "Inspection Date",
        "violation_count": "Violation_Count",
        "latitude": "Latitude",
        "longitude": "Longitude",
        "results_label": "Results_Label",
        "result_label": "Results_Label",
        "current_result_label": "Results_Label",
        "latest_inspection_result": "Results_Label",
    }

    # Normalize column names without changing the actual records
    queue_display = filtered.copy()

    existing_lower_names = {
        str(column).strip().lower(): column for column in queue_display.columns
    }

    for source_name, target_name in column_aliases.items():
        if target_name not in queue_display.columns:
            original_name = existing_lower_names.get(source_name)

            if original_name is not None:
                queue_display.rename(columns={original_name: target_name}, inplace=True)

    # Create rank if the source CSV does not provide it
    if "Priority_Rank" not in queue_display.columns:
        queue_display = queue_display.sort_values(
            "Predicted_Failure_Probability", ascending=False, na_position="last"
        )
        queue_display.insert(0, "Priority_Rank", range(1, len(queue_display) + 1))

    # Missing fields are created as empty values
    for column in queue_columns:
        if column not in queue_display.columns:
            queue_display[column] = pd.NA

    table = queue_display[queue_columns].copy()

    table["Predicted_Failure_Probability"] = pd.to_numeric(
        table["Predicted_Failure_Probability"], errors="coerce"
    )

    # Highest predicted failure probability first
    table = table.sort_values(
        "Predicted_Failure_Probability", ascending=False, na_position="last"
    )

    table = table.rename(
        columns={
            "Priority_Rank": "Priority rank",
            "Predicted_Failure_Probability": "Predicted Failure Probability",
            "Violation_Count": "Violation count",
            "Results_Label": "Result label",
        }
    )

    qc1, qc2 = st.columns([1, 2])
    rows_choice = qc1.selectbox("Rows to display", ["100", "500", "1,000", "All"], index=0)
    sort_choice = qc2.radio(
        "Sort by",
        ["Highest risk first", "Lowest risk first", "Most violations"],
        horizontal=True,
    )

    view_table = table.copy()
    if sort_choice == "Lowest risk first":
        view_table = view_table.sort_values("Predicted Failure Probability", ascending=True)
    elif sort_choice == "Most violations":
        view_table = view_table.sort_values(
            "Violation count", ascending=False, na_position="last"
        )

    n_rows = {"100": 100, "500": 500, "1,000": 1000}.get(rows_choice)
    if n_rows:
        view_table = view_table.head(n_rows)

    # Show the probability on a true 0-100 % scale for the progress bar
    shown = view_table.copy()
    shown["Predicted Failure Probability"] = shown["Predicted Failure Probability"] * 100

    st.dataframe(
        shown,
        use_container_width=True,
        hide_index=True,
        height=520,
        column_config={
            "Priority rank": st.column_config.NumberColumn("Priority rank", format="%d"),
            "Restaurant_ID": st.column_config.TextColumn("Restaurant_ID"),
            "Relative_Risk_Tier": st.column_config.TextColumn("Relative_Risk_Tier"),
            "Predicted Failure Probability": st.column_config.ProgressColumn(
                "Predicted Failure Probability",
                format="%.1f%%",
                min_value=0.0,
                max_value=100.0,
            ),
            "Inspection Date": st.column_config.TextColumn("Inspection Date"),
            "Violation count": st.column_config.NumberColumn("Violation count"),
            "Latitude": st.column_config.NumberColumn("Latitude", format="%.6f"),
            "Longitude": st.column_config.NumberColumn("Longitude", format="%.6f"),
            "Result label": st.column_config.TextColumn("Result label"),
        },
    )

    st.caption(f"Showing {len(shown):,} of {len(table):,} restaurants in the current filter.")

    st.download_button(
        "⬇️ Download filtered Inspector Queue CSV",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="filtered_inspector_queue.csv",
        mime="text/csv",
    )


# ============================================================
# Report Card
# ============================================================

with report_tab:
    st.subheader("Facility Risk Report Card")
    st.caption("Plain-English profile for one restaurant.")

    if "Restaurant_ID" not in queue.columns:
        st.warning("Restaurant ID column is missing from the priority-list CSV.")

    elif queue.empty:
        st.warning("No scored restaurants are available.")

    else:
        ids = sorted(queue["Restaurant_ID"].dropna().astype(str).unique())

        def _jump_to(restaurant_id):
            st.session_state["report_card_restaurant_id"] = str(restaurant_id)

        def _jump_random():
            st.session_state["report_card_restaurant_id"] = str(np.random.choice(ids))

        src = filtered if not filtered.empty else queue
        riskiest_id = (
            src.sort_values("Predicted_Failure_Probability", ascending=False)[
                "Restaurant_ID"
            ].iloc[0]
        )

        s1, s2, s3 = st.columns([3, 1.3, 1.3])

        with s1:
            selected_id = st.selectbox(
                "Select Restaurant ID",
                ids,
                key="report_card_restaurant_id",
            )
        with s2:
            st.write("")
            st.button(
                "🚨 Riskiest in filter",
                on_click=_jump_to,
                args=(riskiest_id,),
                use_container_width=True,
            )
        with s3:
            st.write("")
            st.button("🎲 Random facility", on_click=_jump_random, use_container_width=True)

        fac = queue[queue["Restaurant_ID"].astype(str) == selected_id].iloc[0]

        prob = float(fac["Predicted_Failure_Probability"])
        tier = str(fac.get("Relative_Risk_Tier", "N/A"))
        rank = fac.get("Priority_Rank", None)

        rank_txt = f"#{int(rank):,}" if pd.notna(rank) else "N/A"

        percentile = float((queue["Predicted_Failure_Probability"] < prob).mean())
        diff_pts = (prob - city_avg_risk) * 100

        st.markdown(tier_badge(tier), unsafe_allow_html=True)
        st.write("")

        # ----------------------------------------------------
        # Summary metrics
        # ----------------------------------------------------

        g1, g2, g3, g4 = st.columns(4)

        g1.metric(
            "Predicted failure risk",
            f"{prob:.1%}",
            delta=f"{diff_pts:+.1f} pts vs city average",
            delta_color="inverse",
        )

        tier_lower = tier.lower()
        if "red" in tier_lower:
            tier_display = "Red - High"
        elif "yellow" in tier_lower:
            tier_display = "Yellow - Medium"
        elif "green" in tier_lower:
            tier_display = "Green - Low"
        else:
            tier_display = tier

        g2.metric("Risk tier", tier_display)
        g3.metric("Priority rank", rank_txt)
        g4.metric("Riskier than", f"{percentile:.0%}", help="Share of all facilities with a lower predicted risk")

        st.divider()

        # ----------------------------------------------------
        # Gauge and location
        # ----------------------------------------------------

        left, right = st.columns([1, 1], gap="medium")

        with left:
            fig_g = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=prob * 100,
                    number={"suffix": "%", "font": {"size": 38}},
                    delta={
                        "reference": city_avg_risk * 100,
                        "increasing": {"color": RED},
                        "decreasing": {"color": GREEN},
                        "suffix": " pts vs avg",
                    },
                    title={"text": "Predicted failure risk", "font": {"size": 18}},
                    gauge={
                        "axis": {"range": [0, 100], "ticksuffix": "%"},
                        "bar": {"color": "#94A3B8", "thickness": 0.25},
                        "bgcolor": "rgba(0,0,0,0)",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 20], "color": GREEN},
                            {"range": [20, 35], "color": AMBER},
                            {"range": [35, 100], "color": RED},
                        ],
                        "threshold": {
                            "line": {"color": "#F8FAFC", "width": 3},
                            "thickness": 0.8,
                            "value": prob * 100,
                        },
                    },
                )
            )

            fig_g.update_layout(
                height=300,
                margin=dict(l=15, r=15, t=55, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#F8FAFC"},
            )

            st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar": False})

        with right:
            st.subheader("Location")

            lat = fac.get("Latitude")
            lon = fac.get("Longitude")

            if pd.notna(lat) and pd.notna(lon):
                lat = float(lat)
                lon = float(lon)

                pin = pd.DataFrame(
                    {"lat": [lat], "lon": [lon], "Restaurant_ID": [selected_id]}
                )

                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=pin,
                    get_position="[lon, lat]",
                    get_radius=100,
                    get_fill_color="[239, 68, 68, 230]",
                    pickable=True,
                    auto_highlight=True,
                )

                view = pdk.ViewState(latitude=lat, longitude=lon, zoom=14, pitch=0)

                deck = pdk.Deck(
                    layers=[layer],
                    initial_view_state=view,
                    tooltip={"text": "Restaurant ID: {Restaurant_ID}"},
                )

                st.pydeck_chart(deck, use_container_width=True, height=300)

                st.caption(f"Coordinates: {lat:.5f}, {lon:.5f}")

            else:
                st.info("Location coordinates are unavailable for this restaurant.")

        # ----------------------------------------------------
        # Restaurant details
        # ----------------------------------------------------

        st.subheader("Restaurant Details")

        result_value = str(fac.get("Results_Label", fac.get("Results", "N/A")))

        inspection_date = fac.get("Inspection Date", fac.get("Inspection_Date", "N/A"))

        if pd.isna(inspection_date):
            inspection_date = "N/A"

        violations = fac.get("Violation_Count", "N/A")
        hazards = fac.get("Hazard_Count", "N/A")

        detail1, detail2, detail3 = st.columns(3)

        detail1.metric("Latest result", result_value)
        detail2.metric("Violation count", str(violations))
        detail3.metric("Hazard count", str(hazards))

        st.caption(f"Restaurant ID: {selected_id}  ·  Latest inspection: {inspection_date}")

        # ----------------------------------------------------
        # Download report
        # ----------------------------------------------------

        one = queue[queue["Restaurant_ID"].astype(str) == selected_id]

        st.download_button(
            "⬇️ Download this facility report (CSV)",
            data=one.to_csv(index=False).encode("utf-8"),
            file_name=f"facility_report_{selected_id}.csv",
            mime="text/csv",
            key=f"download_facility_{selected_id}",
        )


# ============================================================
# Community Risk Analysis
# ============================================================

with community_tab:
    st.subheader("Neighborhood Risk Analysis")
    st.caption("Area averages of model-predicted risk")

    if community_summary_path is None:
        st.warning("Neighborhood summary was not found.")

    else:
        community = load_csv(str(community_summary_path))

        if community.empty:
            st.warning("The neighborhood summary exists but contains no data rows.")

        else:
            expected = {
                "neighborhood",
                "restaurants",
                "mean_predicted_failure_probability_pct",
                "high_risk_restaurants",
            }

            if not expected.issubset(community.columns):
                st.warning(
                    "Neighborhood summary columns do not match the "
                    "expected export format."
                )
                st.dataframe(community, use_container_width=True, hide_index=True)

            else:
                for col in [
                    "mean_predicted_failure_probability_pct",
                    "restaurants",
                    "high_risk_restaurants",
                ]:
                    community[col] = pd.to_numeric(community[col], errors="coerce")

                max_n = max(1, int(community["restaurants"].max()))

                min_sample = st.slider(
                    "Minimum restaurants in area", 1, max_n, min(30, max_n), 1
                )

                eligible = community[community["restaurants"] >= min_sample].dropna(
                    subset=["mean_predicted_failure_probability_pct"]
                )

                eligible = eligible.sort_values(
                    "mean_predicted_failure_probability_pct", ascending=False
                )

                a, b, c = st.columns(3)
                a.metric("Areas represented", len(community))
                b.metric("Areas meeting threshold", len(eligible))
                c.metric("Unmapped restaurants", "N/A")

                if not eligible.empty:
                    top_n_areas = st.slider(
                        "Areas to show",
                        1,
                        min(30, len(eligible)),
                        min(15, len(eligible)),
                        1,
                    )

                    chart = eligible.head(top_n_areas).sort_values(
                        "mean_predicted_failure_probability_pct"
                    )

                    fig = px.bar(
                        chart,
                        x="mean_predicted_failure_probability_pct",
                        y="neighborhood",
                        orientation="h",
                        color="mean_predicted_failure_probability_pct",
                        color_continuous_scale="Reds",
                        hover_data=["restaurants", "high_risk_restaurants"],
                        title="Highest mean predicted risk by neighborhood",
                        labels={
                            "mean_predicted_failure_probability_pct": "Mean predicted risk (%)",
                            "neighborhood": "",
                        },
                    )
                    polish(fig, height=max(380, 26 * top_n_areas + 120))
                    fig.update_layout(coloraxis_showscale=False)
                    fig.update_yaxes(showgrid=False)
                    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,.18)")

                    st.plotly_chart(fig, use_container_width=True)

                    st.dataframe(
                        eligible.head(top_n_areas),
                        use_container_width=True,
                        hide_index=True,
                    )

                with st.expander("Full neighborhood summary"):
                    st.dataframe(community, use_container_width=True, hide_index=True)

                st.download_button(
                    "⬇️ Download neighborhood summary CSV",
                    data=community.to_csv(index=False).encode("utf-8"),
                    file_name="neighborhood_risk_summary.csv",
                    mime="text/csv",
                )


# ============================================================
# AI Model Evaluation and Feature Importance
# ============================================================

with model_tab:
    st.subheader("Model Evaluation")

    metrics = {
        "selected_threshold": 0.30,
        "test_accuracy": 0.783,
        "test_precision": 0.687,
        "test_recall": 0.826,
        "test_f1": 0.7503,
        "test_pr_auc": 0.553,
        "test_roc_auc": 0.742,
    }
    metric_map = [
        ("Accuracy", "test_accuracy", "{:.1%}"),
        ("Precision", "test_precision", "{:.1%}"),
        ("Recall", "test_recall", "{:.1%}"),
        ("F1 Score", "test_f1", "{:.3f}"),
        ("PR AUC", "test_pr_auc", "{:.3f}"),
        ("ROC AUC", "test_roc_auc", "{:.3f}"),
    ]

    cols = st.columns(3)

    for i, (label, key, fmt) in enumerate(metric_map):
        cols[i % 3].metric(label, fmt.format(metrics[key]))

    st.caption(
        f"Selected classification threshold: {metrics['selected_threshold']:.2f}"
    )

    labels = [m[0] for m in metric_map]
    values = [metrics[m[1]] for m in metric_map]

    radar_col, info_col = st.columns([3, 2])

    with radar_col:
        fig_r = go.Figure(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=labels + [labels[0]],
                fill="toself",
                line=dict(color=ACCENT, width=3),
                fillcolor="rgba(99,102,241,.28)",
                hovertemplate="<b>%{theta}</b><br>%{r:.3f}<extra></extra>",
            )
        )
        fig_r.update_layout(
            title=dict(text="Performance profile", font=dict(size=17), x=0.01),
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(range=[0, 1], gridcolor="rgba(128,128,128,.25)"),
                angularaxis=dict(gridcolor="rgba(128,128,128,.25)"),
            ),
            showlegend=False,
            height=380,
            margin=dict(l=40, r=40, t=60, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", size=13),
        )
        st.plotly_chart(fig_r, use_container_width=True)

    with info_col:
        st.markdown("#### How to read this")
        st.markdown(
            "- **Recall** – how many truly risky restaurants the model catches.\n"
            "- **Precision** – how many flagged restaurants really are risky.\n"
            "- **ROC AUC / PR AUC** – how well the model ranks risk overall.\n"
            "- The **0.30 threshold** favours catching risky facilities "
            "over avoiding false alarms."
        )

    st.subheader("SHAP Feature Importance")

    shap_tab_img, shap_tab_chart = st.tabs(["Summary image", "Interactive chart"])

    with shap_tab_img:
        if shap_img is not None:
            st.image(str(shap_img), use_container_width=True)
        else:
            st.error(f"SHAP image not found: {SHAP_DIR / 'shap_summary_bar.png'}")

    with shap_tab_chart:
        if shap_path is None:
            st.info("`shap_feature_importance.csv` was not found, so only the image is available.")
        else:
            try:
                shap_df = load_csv(str(shap_path))
                num_cols = shap_df.select_dtypes("number").columns.tolist()
                txt_cols = [c for c in shap_df.columns if c not in num_cols]

                feat_col = txt_cols[0] if txt_cols else shap_df.columns[0]
                val_col = num_cols[0]

                top_k = st.slider("Features to show", 3, min(25, len(shap_df)), min(12, len(shap_df)))
                plot_df = shap_df.nlargest(top_k, val_col).sort_values(val_col)

                fig_s = px.bar(
                    plot_df,
                    x=val_col,
                    y=feat_col,
                    orientation="h",
                    color=val_col,
                    color_continuous_scale="Viridis",
                    title="Top drivers of predicted inspection failure",
                )
                polish(fig_s, height=max(360, 30 * top_k + 120))
                fig_s.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig_s, use_container_width=True)
            except Exception as exc:
                st.info(f"Could not build the interactive chart ({exc}). Showing the image instead.")


# ============================================================
# Footer
# ============================================================

st.markdown(
    '<div class="footer-note">⚠️ <b>Research prototype only.</b> Predictions support — '
    'not replace — professional inspection decisions.</div>',
    unsafe_allow_html=True,
)