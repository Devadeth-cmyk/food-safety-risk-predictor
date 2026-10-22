"""
Chicago Restaurant Health Inspection Risk System  (Streamlit dashboard)

Run with:  streamlit run app.py

Required input (beside app.py or in ./outputs):
    restaurant_risk_priority_queue_corrected.csv   (or restaurant_risk_priority_queue.csv)

Optional inputs (each section degrades gracefully with a message if missing):
    model_test_metrics.csv, shap_feature_importance.csv, seasonal_failure_by_month.csv,
    community_area_risk_summary.csv, restaurant_community_area_risk.csv,
    inspection_history.csv / violation_details.csv      -> Violation Analytics, yearly trend
    local_shap_values.csv                               -> individual SHAP (precomputed)
    risk_model.joblib + model_features.csv              -> individual SHAP (computed live)
    pr_curve.csv, calibration_curve.csv, confusion_matrix.csv, model_comparison.csv
                                                        -> extra Model Performance charts
"""

import math
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pydeck as pdk
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

st.set_page_config(
    page_title="Chicago Food Safety Risk System",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Visual theme
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    :root { --ink:#18313f; --muted:#607481; --teal:#087f8c; --teal-dark:#075c69;
            --gold:#f2a900; --canvas:#f3f6fa; --line:#dce4ed; }
    .stApp { background:var(--canvas); color:var(--ink); }
    [data-testid="stHeader"] { background:rgba(243,246,250,.92); }
    [data-testid="stMainBlockContainer"] { padding-top:2rem; padding-bottom:3rem; }
    [data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"] { gap:1.05rem; }
    h1,h2,h3 { color:var(--ink); letter-spacing:-.025em; }
    h1 { font-weight:750; }
    [data-testid="stMainBlockContainer"] h2 { margin-top:.75rem; margin-bottom:.15rem; }
    [data-testid="stMainBlockContainer"] h3 { margin-top:.55rem; margin-bottom:.1rem; }
    [data-testid="stCaptionContainer"] { color:var(--muted); }
    .hero-panel { min-height:220px; box-sizing:border-box; padding:2rem 2.1rem;
        border-radius:18px; color:white; display:flex; flex-direction:column;
        justify-content:center; background:radial-gradient(ellipse at 88% 5%,rgba(242,169,0,.45),transparent 32%),
        linear-gradient(118deg,#12344d 0%,#075c69 54%,#078b86 100%);
        box-shadow:0 14px 34px rgba(7,76,94,.18); }
    .hero-eyebrow { color:#ffd36a; text-transform:uppercase; letter-spacing:.15em;
        font-size:.76rem; font-weight:800; margin:0 0 .7rem; }
    .hero-title { color:#fff; font-size:clamp(1.8rem,3vw,2.8rem); line-height:1.08;
        font-weight:800; letter-spacing:-.04em; margin:0 0 .75rem; }
    .hero-copy { color:#e2f3f4; font-size:1rem; line-height:1.55; max-width:760px; margin:0; }
    .hero-visual-label { color:#075c69; font-size:.78rem; font-weight:750;
        letter-spacing:.08em; text-transform:uppercase; margin:0 0 .55rem; }
    [data-testid="stImage"] img { border-radius:13px; border:1px solid #dce4ed; }
    [data-testid="stMetric"] { background:#fff; border:1px solid var(--line);
        border-top:3px solid var(--teal); border-radius:12px; padding:1rem 1.1rem;
        box-shadow:0 3px 12px rgba(23,52,58,.045); }
    [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2) [data-testid="stMetric"] {
        border-top-color:#d94355; }
    [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(3) [data-testid="stMetric"] {
        border-top-color:#e5a000; }
    [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(4) [data-testid="stMetric"] {
        border-top-color:#4c6fff; }
    [data-testid="stMetricLabel"] { color:var(--muted); font-weight:600; }
    [data-testid="stMetricValue"] { color:var(--ink); font-weight:750; }
    [data-testid="stTabs"] [role="tablist"] { gap:.8rem; border-bottom:1px solid var(--line); }
    [data-testid="stTabs"] button[role="tab"] { color:var(--muted);
        border-radius:9px 9px 0 0; font-weight:600; padding:.55rem .7rem; }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color:var(--teal-dark); border-bottom:3px solid var(--gold); }
    [data-testid="stSidebar"] { background:#e7eff6; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 { color:var(--teal-dark); }
    .stButton > button,.stDownloadButton > button { border-radius:9px;
        border-color:var(--teal); color:var(--teal-dark); font-weight:650; transition:all .15s ease; }
    .stButton > button:hover,.stDownloadButton > button:hover {
        color:#fff; background:var(--teal); border-color:var(--teal); }
    [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:10px; }
    [data-testid="stAlert"] { border-radius:10px; }
    hr { border-color:var(--line); }
</style>
""", unsafe_allow_html=True)

CHART_COLORS = ["#087f8c", "#f2a900", "#4c6fff", "#e05263", "#168a64", "#8657d8"]
pio.templates["food_safety"] = pio.templates["plotly_white"]
pio.templates["food_safety"].layout.update(
    font={"family":"Inter, Segoe UI, sans-serif", "color":"#29464b", "size":12},
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", colorway=CHART_COLORS,
    margin={"l":12, "r":12, "t":58, "b":12},
    title={"font":{"size":16, "color":"#17343a"}, "x":0.02, "xanchor":"left"},
    xaxis={"gridcolor":"#e7eeec", "zerolinecolor":"#dce7e4", "linecolor":"#dce7e4"},
    yaxis={"gridcolor":"#e7eeec", "zerolinecolor":"#dce7e4", "linecolor":"#dce7e4"},
    legend={"bgcolor":"rgba(255,255,255,.7)", "borderwidth":0},
)
pio.templates.default = "food_safety"
px.defaults.template = "food_safety"
px.defaults.color_discrete_sequence = CHART_COLORS

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
TIER_COLORS_RGB = {"red": [217, 67, 85], "yellow": [229, 160, 0], "green": [20, 134, 109]}
TIER_COLORS_HEX = {"red": "#D94355", "yellow": "#E5A000", "green": "#14866D"}
TIER_TEXT = {
    "red": "Red - higher predicted risk",
    "yellow": "Yellow - medium predicted risk",
    "green": "Green - lower predicted risk",
}
CHICAGO_CENTER = (41.8781, -87.6298)
# Loose bounding box around Chicago, used to flag obviously invalid coordinates.
LAT_RANGE = (41.60, 42.10)
LON_RANGE = (-87.95, -87.45)

# canonical column name -> accepted alternative names in the source file
CANONICAL_COLUMNS = {
    "Priority_Rank": ["Priority_Rank", "Priority Rank", "Rank"],
    "Restaurant_ID": ["Restaurant_ID", "Restaurant ID", "License #", "License_Number", "License"],
    "Restaurant_Name": ["Restaurant_Name", "Restaurant Name", "DBA Name", "DBA_Name", "AKA Name", "Name"],
    "Relative_Risk_Tier": ["Relative_Risk_Tier", "Risk_Tier", "Risk Tier"],
    "Predicted_Failure_Probability": ["Predicted_Failure_Probability", "Predicted Failure Probability"],
    "Inspection_Date": ["Inspection Date", "Inspection_Date", "Last_Inspection_Date", "Last Inspection Date"],
    "Previous_Result": ["Results", "Previous_Result", "Previous_Inspection_Result", "Last_Result"],
    "Violation_Count": ["Violation_Count", "Violation Count", "Violations_Count"],
    "Hazard_Count": ["Hazard_Count", "Hazard Count"],
    "Address": ["Address", "Street_Address"],
    "Community_Area": ["Community_Area", "Community Area", "Community_Area_Name"],
    "Latitude": ["Latitude", "Lat"],
    "Longitude": ["Longitude", "Lon", "Lng", "Long"],
}

EXPORT_LABELS = {
    "Priority_Rank": "Priority Rank",
    "Restaurant_ID": "Restaurant ID",
    "Restaurant_Name": "Restaurant Name",
    "Relative_Risk_Tier": "Relative Risk Tier",
    "Predicted_Failure_Probability": "Predicted Failure Probability",
    "Inspection_Date": "Most Recent Inspection Date",
    "Previous_Result": "Previous Inspection Result",
    "Violation_Count": "Violation Count",
    "Hazard_Count": "Hazard Count",
    "Address": "Address",
    "Community_Area": "Community Area",
}

PREVIOUSLY_REPORTED = {
    "PR-AUC": 0.375,
    "ROC-AUC": 0.641,
    "Precision @ top 5%": 0.470,
}


# ----------------------------------------------------------------------------
# Generic helpers
# ----------------------------------------------------------------------------
def norm(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def find_col(df, candidates):
    """Case/punctuation-insensitive column lookup. Returns the actual column name or None."""
    lookup = {norm(c): c for c in df.columns}
    for cand in candidates:
        if norm(cand) in lookup:
            return lookup[norm(cand)]
    return None


def first_existing(*names):
    for name in names:
        for path in (OUTPUT_DIR / name, BASE_DIR / name):
            if path.exists():
                return path
    return None


@st.cache_data(show_spinner=False)
def load_csv(path_string, mtime):
    """Cached CSV loader. `mtime` is part of the cache key so edited files are re-read."""
    return pd.read_csv(path_string, low_memory=False)


def read_csv_safe(path):
    """Load a CSV (cached). Returns None (and shows a warning) if it cannot be read."""
    if path is None:
        return None
    try:
        return load_csv(str(path), path.stat().st_mtime)
    except Exception as exc:  # noqa: BLE001 - surface any parsing problem to the user
        st.warning(f"Could not read {path.name}: {exc}")
        return None


def file_date(path):
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d") if path else None


def tier_key(label):
    s = str(label).lower()
    for key in ("red", "yellow", "green"):
        if key in s:
            return key
    return None


def tier_color_map(tiers):
    return {t: TIER_COLORS_HEX.get(tier_key(t), "#888888") for t in tiers}


def ordered_tiers(df, tier_col="Relative_Risk_Tier", prob_col="Predicted_Failure_Probability"):
    """Order tiers from highest to lowest mean predicted risk."""
    if tier_col not in df.columns:
        return []
    means = df.groupby(tier_col)[prob_col].mean().sort_values(ascending=False)
    return means.index.tolist()


def pct(x, digits=1):
    return f"{x:.{digits}%}" if pd.notna(x) else "N/A"


def int_slider(label, lo, hi, default, **kwargs):
    """Slider that never crashes when lo >= hi (Streamlit rejects equal bounds)."""
    lo, hi = int(lo), int(hi)
    if hi <= lo:
        st.caption(f"{label}: {lo} (only option available)")
        return lo
    return st.slider(label, min_value=lo, max_value=hi, value=int(min(max(default, lo), hi)), step=1, **kwargs)


def to_csv_bytes(df):
    """CSV export that also works for an empty frame (headers only)."""
    return df.to_csv(index=False).encode("utf-8")


def prettify_feature(name):
    s = str(name)
    s = re.sub(r"^(num|cat|remainder|numeric|categorical)__", "", s)
    s = s.replace("_", " ").replace("  ", " ").strip()
    return s[:1].upper() + s[1:] if s else s


def metric_from_dict(metrics, *token_sets):
    """Find a metric value by token matching on normalised metric names."""
    for tokens in token_sets:
        for key, value in metrics.items():
            k = norm(key)
            if all(t in k for t in tokens):
                return value
    return None


def fmt_metric(value, digits=3):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A" if value is None else str(value)


# ----------------------------------------------------------------------------
# File discovery
# ----------------------------------------------------------------------------
queue_path = first_existing(
    "restaurant_risk_priority_queue_corrected.csv",
    "restaurant_risk_priority_queue.csv",
)
metrics_path = first_existing("model_test_metrics.csv", "model_test_metrics(2).csv")
season_path = first_existing("seasonal_failure_by_month.csv")
shap_path = first_existing("shap_feature_importance.csv", "shap_feature_importance(1).csv")
shap_figure_path = first_existing("shap_summary.png")
community_summary_path = first_existing("community_area_risk_summary.csv")
community_detail_path = first_existing("restaurant_community_area_risk.csv")
history_path = first_existing(
    "inspection_history.csv", "violation_details.csv", "violations.csv", "inspections.csv"
)
local_shap_path = first_existing("local_shap_values.csv", "shap_local_values.csv")
model_path = first_existing(
    "risk_model.joblib", "model.joblib", "best_model.joblib", "final_model.joblib",
    "risk_model.pkl", "model.pkl",
)
model_features_path = first_existing(
    "model_features.csv", "restaurant_model_features.csv",
    "scoring_features.csv", "model_input_features.csv",
)
pr_curve_path = first_existing("pr_curve.csv", "precision_recall_curve.csv")
calibration_path = first_existing("calibration_curve.csv", "calibration.csv")
confusion_path = first_existing("confusion_matrix.csv")
comparison_path = first_existing("model_comparison.csv")

# ----------------------------------------------------------------------------
# Header + required data
# ----------------------------------------------------------------------------
hero_visual = first_existing("geographic_risk_heatmap.png", "geographic_grid_risk_heatmap.png")
hero_text, hero_image = st.columns([1.65, 1], gap="medium")
with hero_text:
    st.markdown(
        "<section class='hero-panel'>"
        "<p class='hero-eyebrow'>CHICAGO | FOOD SAFETY INTELLIGENCE</p>"
        "<h1 class='hero-title'>Restaurant Inspection<br>Risk Dashboard</h1>"
        "<p class='hero-copy'>A citywide view of predicted inspection risk, restaurant locations, "
        "and inspection history. Use the risk queue to help guide professional review.</p>"
        "</section>",
        unsafe_allow_html=True,
    )
with hero_image:
    if hero_visual is not None:
        st.markdown("<p class='hero-visual-label'>Chicago risk by location</p>", unsafe_allow_html=True)
        st.image(hero_visual, use_container_width=True)
    else:
        st.markdown(
            "<div class='hero-panel'><p class='hero-eyebrow'>Restaurant inspection insights</p>"
            "<p class='hero-title'>Safer decisions<br>through data</p></div>",
            unsafe_allow_html=True,
        )
st.caption(
    "Prototype decision-support dashboard. Predicted probabilities estimate the model's "
    "risk of failure at the next routine Canvass inspection; they are not official safety ratings."
)

if queue_path is None:
    st.error(
        "Risk queue CSV not found. Place restaurant_risk_priority_queue_corrected.csv "
        "beside app.py (or in an `outputs` folder)."
    )
    st.stop()

queue_raw = read_csv_safe(queue_path)
if queue_raw is None:
    st.error("The risk queue file could not be loaded.")
    st.stop()

queue = queue_raw.copy()
# Rename any recognised alternative column names to canonical names.
for canonical, candidates in CANONICAL_COLUMNS.items():
    if canonical in queue.columns:
        continue
    found = find_col(queue, candidates)
    if found is not None:
        queue = queue.rename(columns={found: canonical})

if "Predicted_Failure_Probability" not in queue.columns:
    st.error("Risk queue is missing required column: Predicted_Failure_Probability")
    st.stop()

queue["Predicted_Failure_Probability"] = pd.to_numeric(
    queue["Predicted_Failure_Probability"], errors="coerce"
)
queue = queue.dropna(subset=["Predicted_Failure_Probability"]).copy()
if queue.empty:
    st.error("The risk queue contains no rows with a valid predicted probability.")
    st.stop()

# Restaurant ID is needed for search/lookup; fall back to row number if absent.
if "Restaurant_ID" not in queue.columns:
    queue["Restaurant_ID"] = [f"ROW-{i + 1}" for i in range(len(queue))]
    ID_IS_SYNTHETIC = True
else:
    ID_IS_SYNTHETIC = False
queue["Restaurant_ID"] = queue["Restaurant_ID"].astype(str)

HAS_NAME = "Restaurant_Name" in queue.columns
if HAS_NAME:
    queue["Restaurant_Name"] = queue["Restaurant_Name"].astype("string")

# Check: the scoring output should contain one row per restaurant.
duplicate_ids = int(queue["Restaurant_ID"].duplicated().sum()) if not ID_IS_SYNTHETIC else 0

# Relative risk tier: use the file's values; derive ranking bands only if the column is missing.
TIERS_DERIVED = "Relative_Risk_Tier" not in queue.columns
if TIERS_DERIVED:
    rank_pct = queue["Predicted_Failure_Probability"].rank(ascending=False, pct=True, method="first")
    queue["Relative_Risk_Tier"] = pd.cut(
        rank_pct, bins=[0, 0.10, 0.30, 1.0],
        labels=["Red - highest 10%", "Yellow - next 20%", "Green - remaining 70%"],
        include_lowest=True,
    ).astype(str)

queue = queue.sort_values("Predicted_Failure_Probability", ascending=False).reset_index(drop=True)
if "Priority_Rank" not in queue.columns:
    queue["Priority_Rank"] = range(1, len(queue) + 1)

# Coordinates: validate and flag.
for col in ("Latitude", "Longitude"):
    if col in queue.columns:
        queue[col] = pd.to_numeric(queue[col], errors="coerce")
HAS_COORDS = {"Latitude", "Longitude"}.issubset(queue.columns)
if HAS_COORDS:
    queue["Valid_Coordinates"] = (
        queue["Latitude"].between(*LAT_RANGE) & queue["Longitude"].between(*LON_RANGE)
    )
else:
    queue["Valid_Coordinates"] = False

# Inspection date (most recent inspection) -> datetime + year.
if "Inspection_Date" in queue.columns:
    queue["_inspection_dt"] = pd.to_datetime(queue["Inspection_Date"], errors="coerce")
    queue["_inspection_year"] = queue["_inspection_dt"].dt.year
HAS_DATE = "_inspection_year" in queue.columns and queue["_inspection_year"].notna().any()

# Community area: use the queue's column, otherwise try merging from the community detail file.
AREA_FROM_DETAIL = False
if "Community_Area" not in queue.columns and community_detail_path is not None:
    detail_df = read_csv_safe(community_detail_path)
    if detail_df is not None:
        id_col = find_col(detail_df, CANONICAL_COLUMNS["Restaurant_ID"])
        area_col = find_col(detail_df, CANONICAL_COLUMNS["Community_Area"])
        if id_col and area_col:
            lookup = detail_df[[id_col, area_col]].copy()
            lookup[id_col] = lookup[id_col].astype(str)
            lookup = lookup.drop_duplicates(subset=[id_col]).rename(
                columns={id_col: "Restaurant_ID", area_col: "Community_Area"}
            )
            queue = queue.merge(lookup, on="Restaurant_ID", how="left")
            AREA_FROM_DETAIL = True
HAS_AREA = "Community_Area" in queue.columns

# Which tier is "high priority"?
TIER_ORDER = ordered_tiers(queue)
HIGH_TIER = next((t for t in TIER_ORDER if tier_key(t) == "red"), TIER_ORDER[0] if TIER_ORDER else None)
TIER_CMAP = tier_color_map(TIER_ORDER)

# Data snapshot label (never display predictions without one).
snapshot_col = find_col(queue, ["Prediction_Date", "Snapshot_Date", "Scoring_Date", "Scored_On", "Data_Snapshot"])
if snapshot_col is not None:
    snap_vals = pd.to_datetime(queue[snapshot_col], errors="coerce").dropna()
    SNAPSHOT_LABEL = (
        f"Predictions as of {snap_vals.max().date()}" if not snap_vals.empty
        else f"Predictions file last modified {file_date(queue_path)}"
    )
else:
    SNAPSHOT_LABEL = (
        f"Static snapshot - queue file last modified {file_date(queue_path)} "
        "(no prediction-date column in the file)"
    )

# ----------------------------------------------------------------------------
# Sidebar: global filters (with reset)
# ----------------------------------------------------------------------------
prob_lo = max(0.0, math.floor(queue["Predicted_Failure_Probability"].min() * 100) / 100)
prob_hi = min(1.0, math.ceil(queue["Predicted_Failure_Probability"].max() * 100) / 100)
if prob_hi <= prob_lo:
    prob_lo, prob_hi = max(0.0, prob_lo - 0.01), min(1.0, prob_hi + 0.01)

area_options = (
    sorted(queue["Community_Area"].dropna().astype(str).unique().tolist()) if HAS_AREA else []
)
year_options = (
    sorted(int(y) for y in queue["_inspection_year"].dropna().unique()) if HAS_DATE else []
)

FILTER_DEFAULTS = {
    "f_tiers": TIER_ORDER,
    "f_prob": (prob_lo, prob_hi),
    "f_search": "",
    "f_areas": [],
    "f_years": year_options,
}
for key, value in FILTER_DEFAULTS.items():
    st.session_state.setdefault(key, value)


def reset_filters():
    for k, v in FILTER_DEFAULTS.items():
        st.session_state[k] = v


with st.sidebar:
    st.header("Filters")
    st.button("Reset filters", on_click=reset_filters, use_container_width=True)

    st.subheader("Predicted-risk filters")
    st.caption("These apply to scored restaurants: summary cards, overview charts, map and priority queue.")
    st.multiselect("Risk tier", TIER_ORDER, key="f_tiers",
                   help="Relative ranking bands, not official grades. Leave empty to include all tiers.")
    st.slider("Predicted failure probability", min_value=0.0, max_value=1.0,
              step=0.01, key="f_prob")
    st.text_input(
        "Search restaurant" + (" ID or name" if HAS_NAME else " ID"),
        key="f_search",
        placeholder="Enter an ID" + (" or name" if HAS_NAME else ""),
    )
    if HAS_AREA:
        st.multiselect("Community area", area_options, key="f_areas",
                       help="Leave empty to include all areas.")
    else:
        st.caption("Community-area filter unavailable: no area column in the data.")
    if HAS_DATE:
        st.multiselect("Most recent inspection year", year_options, key="f_years",
                       help="Filters scored restaurants by the year of their latest inspection. "
                            "Leave empty to include all years.")
    else:
        st.caption("Inspection-year filter unavailable: no inspection-date column in the queue.")

    st.divider()
    st.subheader("Historical-record filters")
    st.caption(
        "Violation Analytics uses historical inspection records. Its filters are separate "
        "from the predicted-risk filters above."
    )
    st.caption(f"🗓️ {SNAPSHOT_LABEL}")


def apply_filters(df):
    """Apply all sidebar predicted-risk filters consistently."""
    out = df
    sel_tiers = st.session_state.get("f_tiers") or []
    if sel_tiers and "Relative_Risk_Tier" in out.columns:
        out = out[out["Relative_Risk_Tier"].isin(sel_tiers)]
    lo, hi = st.session_state.get("f_prob", (prob_lo, prob_hi))
    out = out[out["Predicted_Failure_Probability"].between(lo, hi)]
    term = (st.session_state.get("f_search") or "").strip()
    if term:
        mask = out["Restaurant_ID"].astype(str).str.contains(term, case=False, na=False, regex=False)
        if HAS_NAME:
            mask |= out["Restaurant_Name"].astype(str).str.contains(term, case=False, na=False, regex=False)
        out = out[mask]
    sel_areas = st.session_state.get("f_areas") or []
    if sel_areas and HAS_AREA:
        out = out[out["Community_Area"].astype(str).isin(sel_areas)]
    sel_years = st.session_state.get("f_years") or []
    if sel_years and HAS_DATE and set(sel_years) != set(year_options):
        out = out[out["_inspection_year"].isin(sel_years)]
    return out


filtered = apply_filters(queue)

# ----------------------------------------------------------------------------
# KPI cards
# ----------------------------------------------------------------------------
high_in_filter = int((filtered["Relative_Risk_Tier"] == HIGH_TIER).sum()) if HIGH_TIER else None
c1, c2, c3, c4 = st.columns(4)
c1.metric("Restaurants Scored", f"{len(queue):,}", help="All restaurants in the risk queue (ignores filters).")
c2.metric(
    "High-Priority Restaurants",
    f"{high_in_filter:,}" if high_in_filter is not None else "N/A",
    help=f"Restaurants in the highest tier ({HIGH_TIER}) within the current filter.",
)
c3.metric(
    "Average Predicted Risk",
    pct(filtered["Predicted_Failure_Probability"].mean()) if len(filtered) else "N/A",
    help="Mean predicted failure probability within the current filter.",
)
c4.metric("Restaurants in Current Filter", f"{len(filtered):,}",
          delta=f"{len(filtered) / len(queue):.0%} of scored" if len(queue) else None,
          delta_color="off")
if duplicate_ids:
    st.warning(
        f"Data check: {duplicate_ids:,} duplicate Restaurant_ID values found in the queue. "
        "The scoring output should contain one row per restaurant."
    )
if ID_IS_SYNTHETIC:
    st.warning("No Restaurant_ID column found; row numbers are being used as identifiers.")
if TIERS_DERIVED:
    st.info("No Relative_Risk_Tier column found; tiers were derived from probability rank "
            "(top 10% / next 20% / remaining 70%).")

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
(overview_tab, map_tab, queue_tab, lookup_tab, violation_tab,
 community_tab, model_tab) = st.tabs([
    "Overview", "Geographic Risk Map", "Inspector Priority Queue",
    "Restaurant Lookup & SHAP", "Violation Analytics",
    "Community Area Analysis", "Model Performance",
])

# ============================ 1. OVERVIEW ====================================
with overview_tab:
    st.subheader("Overview")
    st.markdown(
        "**What the model predicts:** the probability that a restaurant will fail its next routine "
        "Canvass inspection, based on its inspection history. "
        "**Risk tiers** are relative ranking bands (top 10%, next 20%, remaining 70% of predicted "
        "probability) - they are *not* official food-safety grades. "
        f"**Data:** this is a static snapshot ({SNAPSHOT_LABEL}). "
        "Predictions are meant to support, not replace, professional inspection decisions."
    )
    if filtered.empty:
        st.warning("No restaurants match the current filters.")
    else:
        left, right = st.columns(2)
        with left:
            tier_counts = (
                filtered["Relative_Risk_Tier"].value_counts()
                .reindex([t for t in TIER_ORDER if t in set(filtered["Relative_Risk_Tier"])])
                .rename_axis("Risk tier").reset_index(name="Restaurants")
            )
            fig = px.bar(tier_counts, x="Risk tier", y="Restaurants", color="Risk tier",
                         color_discrete_map=TIER_CMAP, title="Restaurants by relative risk tier (current filter)")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = px.histogram(
                filtered, x="Predicted_Failure_Probability", nbins=30,
                title="Distribution of predicted failure probabilities (current filter)",
                labels={"Predicted_Failure_Probability": "Predicted failure probability"},
            )
            fig.update_xaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("**Historical failure trend** (observed outcomes, not predictions)")
        hist_df = read_csv_safe(history_path)
        yearly_done = False
        if hist_df is not None:
            res_col = find_col(hist_df, ["Results", "Result", "Inspection_Result"])
            date_col = find_col(hist_df, ["Inspection Date", "Inspection_Date", "Date"])
            if res_col and date_col:
                tmp = hist_df[[date_col, res_col]].copy()
                tmp["_year"] = pd.to_datetime(tmp[date_col], errors="coerce").dt.year
                tmp = tmp.dropna(subset=["_year", res_col])
                tmp["_fail"] = tmp[res_col].astype(str).str.contains("fail", case=False)
                # Drop rows that are not pass/fail-type outcomes only if clearly irrelevant
                yearly = tmp.groupby("_year")["_fail"].agg(["mean", "size"]).reset_index()
                if not yearly.empty:
                    fig = px.line(yearly, x="_year", y="mean", markers=True,
                                  labels={"_year": "Inspection year", "mean": "Failure rate"},
                                  hover_data={"size": True})
                    fig.update_yaxes(tickformat=".0%")
                    fig.update_layout(title="Historical failure rate by year (share of inspections)")
                    st.plotly_chart(fig, use_container_width=True)
                    yearly_done = True
        if not yearly_done:
            season_df = read_csv_safe(season_path)
            if season_df is not None and {"Month", "Failure_Rate_Percent"}.issubset(season_df.columns):
                fig = px.line(season_df, x="Month", y="Failure_Rate_Percent", markers=True,
                              labels={"Failure_Rate_Percent": "Failure rate (%)"},
                              title="Historical failure rate by calendar month")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Monthly historical pattern from seasonal_failure_by_month.csv.")
            else:
                st.info("No historical inspection data available for a trend chart.")
    with right:
        st.markdown("**Mapped vs. unmapped restaurants**")
        valid_n = int(filtered["Valid_Coordinates"].sum())
        mapped_df = pd.DataFrame({
            "Status": ["Mapped (valid coordinates)", "Unmapped (missing/invalid coordinates)"],
            "Restaurants": [valid_n, len(filtered) - valid_n],
        })
        fig = px.bar(mapped_df, x="Status", y="Restaurants", title="Coordinate coverage (current filter)")
        st.plotly_chart(fig, use_container_width=True)
        if not HAS_COORDS:
            st.warning("Latitude/Longitude columns are not present in the queue.")

# ============================ 2. MAP =========================================
with map_tab:
    st.subheader("Geographic distribution of predicted risk")
    st.caption(
        "Each point is a restaurant location colored by its relative risk tier. "
        "This is a point map, not a community-area boundary map or heatmap. A dense cluster of "
        "markers reflects where restaurants are located and does not mean an area is unsafe."
    )
    if not HAS_COORDS:
        st.warning("The queue has no Latitude/Longitude columns, so the map is unavailable.")
    elif filtered.empty:
        st.warning("No restaurants match the current filters.")
    else:
        invalid_n = int((~filtered["Valid_Coordinates"]).sum())
        map_filtered = filtered[filtered["Valid_Coordinates"]].copy()
        if invalid_n:
            st.info(f"{invalid_n:,} restaurant(s) in the current filter were excluded from the map "
                    "because of missing or invalid coordinates.")
        if map_filtered.empty:
            st.warning("No restaurants with valid coordinates match the current filters.")
        else:
            ctl1, ctl2 = st.columns([1, 2])
            with ctl1:
                center_on_data = st.checkbox("Center map on filtered restaurants", value=True)
            with ctl2:
                radius = st.slider("Marker size (meters)", 20, 200, 65, step=5)

            # Build a lean, string-safe frame for pydeck.
            def fill_color(label):
                return TIER_COLORS_RGB.get(tier_key(label), [136, 136, 136])

            map_view = pd.DataFrame({
                "Latitude": map_filtered["Latitude"],
                "Longitude": map_filtered["Longitude"],
                "color": map_filtered["Relative_Risk_Tier"].map(fill_color),
                "Restaurant": (map_filtered["Restaurant_Name"].fillna(map_filtered["Restaurant_ID"])
                               .astype(str) if HAS_NAME else map_filtered["Restaurant_ID"]),
                "Restaurant_ID": map_filtered["Restaurant_ID"],
                "Probability": map_filtered["Predicted_Failure_Probability"].map(lambda v: f"{v:.1%}"),
                "Tier": map_filtered["Relative_Risk_Tier"].astype(str),
            })
            tooltip_lines = [
                "<b>{Restaurant}</b>",
                "ID: {Restaurant_ID}",
                "Predicted failure probability: <b>{Probability}</b>",
                "Risk tier: {Tier}",
            ]
            optional_tooltip = [
                ("Inspection_Date", "Inspection date", "Inspection_Date_Text"),
                ("Previous_Result", "Previous result", "Previous_Result_Text"),
                ("Violation_Count", "Violations", "Violation_Count_Text"),
                ("Hazard_Count", "Hazards", "Hazard_Count_Text"),
            ]
            for src, label, new in optional_tooltip:
                if src in map_filtered.columns:
                    map_view[new] = map_filtered[src].astype(str).replace({"nan": "N/A", "<NA>": "N/A"})
                    tooltip_lines.append(f"{label}: {{{new}}}")

            if center_on_data:
                lat_c = float(map_view["Latitude"].mean())
                lon_c = float(map_view["Longitude"].mean())
                spread = max(
                    float(map_view["Latitude"].max() - map_view["Latitude"].min()),
                    float(map_view["Longitude"].max() - map_view["Longitude"].min()) * 0.75,
                    0.005,
                )
                zoom = max(9.0, min(15.0, math.log2(360 / spread) - 1.2))
            else:
                (lat_c, lon_c), zoom = CHICAGO_CENTER, 10

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_view,
                get_position="[Longitude, Latitude]",
                get_radius=radius,
                get_fill_color="color",
                pickable=True,
                auto_highlight=True,
            )
            st.pydeck_chart(pdk.Deck(
                layers=[layer],
                initial_view_state=pdk.ViewState(latitude=lat_c, longitude=lon_c, zoom=zoom, pitch=0),
                tooltip={
                    "html": "<br/>".join(tooltip_lines),
                    "style": {"backgroundColor": "#1f2937", "color": "white", "fontSize": "13px"},
                },
            ))

            present_keys = [k for k in ("red", "yellow", "green")
                            if any(tier_key(t) == k for t in map_filtered["Relative_Risk_Tier"].unique())]
            legend = "".join(
                f"<span style='display:inline-block;width:14px;height:14px;border-radius:50%;"
                f"background:{TIER_COLORS_HEX[k]};margin-right:6px;vertical-align:middle'></span>"
                f"<span style='margin-right:18px'>{TIER_TEXT[k]}</span>"
                for k in present_keys
            )
            st.markdown(f"**Legend:** {legend}", unsafe_allow_html=True)
            st.caption(f"Showing {len(map_filtered):,} mapped restaurants "
                       f"({invalid_n:,} excluded for missing/invalid coordinates). "
                       "Probabilities are model predictions from the current queue.")

# ============================ 3. PRIORITY QUEUE ==============================
QUEUE_COLUMNS = [
    "Priority_Rank", "Restaurant_ID", "Restaurant_Name", "Relative_Risk_Tier",
    "Predicted_Failure_Probability", "Inspection_Date", "Previous_Result",
    "Violation_Count", "Hazard_Count", "Address", "Community_Area",
]

with queue_tab:
    st.subheader("Inspector priority queue")
    st.caption(f"Sorted by predicted failure probability (highest first). {SNAPSHOT_LABEL}. "
               "Use the sidebar to filter by tier, probability, search term and area.")

    display_cols = [c for c in QUEUE_COLUMNS if c in filtered.columns]
    table = filtered.sort_values("Predicted_Failure_Probability", ascending=False)[display_cols].reset_index(drop=True)
    table_labeled = table.rename(columns=EXPORT_LABELS)

    if table.empty:
        st.warning("No restaurants match the current filters.")
        st.dataframe(table_labeled, use_container_width=True, hide_index=True)
    else:
        st.caption(f"{len(table):,} restaurants in the queue view.")
        column_config = {
            "Predicted Failure Probability": st.column_config.NumberColumn(format="%.3f"),
        }
        selected_row = None
        try:
            event = st.dataframe(
                table_labeled, use_container_width=True, hide_index=True,
                on_select="rerun", selection_mode="single-row",
                column_config=column_config, key="queue_table",
            )
            rows = event.selection.rows if event is not None else []
            if rows:
                selected_row = table.iloc[rows[0]]
        except TypeError:
            # Older Streamlit without dataframe selection: plain table + selectbox below.
            st.dataframe(table_labeled, use_container_width=True, hide_index=True)

        options = table["Restaurant_ID"].head(1000).tolist()
        st.markdown("**Select a restaurant to inspect**")
        st.caption("Click a table row (checkbox at left), or choose from the list. "
                   "The selection carries over to the *Restaurant Lookup & SHAP* tab.")
        row_id = str(selected_row["Restaurant_ID"]) if selected_row is not None else None
        if row_id and row_id != st.session_state.get("last_row_sel"):
            st.session_state["selected_id"] = row_id  # a new table click wins over the dropdown
            st.session_state["last_row_sel"] = row_id
        current = st.session_state.get("selected_id")
        idx = options.index(current) if current in options else 0

        def label_for(rid):
            row = table.loc[table["Restaurant_ID"] == rid].iloc[0]
            nm = f" - {row['Restaurant_Name']}" if HAS_NAME and pd.notna(row.get("Restaurant_Name")) else ""
            return f"#{int(row['Priority_Rank'])}  {rid}{nm}  ({row['Predicted_Failure_Probability']:.1%})"

        choice = st.selectbox("Restaurant", options, index=idx, format_func=label_for)
        st.session_state["selected_id"] = str(choice)

        detail_row = table.loc[table["Restaurant_ID"] == str(choice)].iloc[0]
        with st.container(border=True):
            d1, d2, d3 = st.columns(3)
            d1.metric("Predicted failure probability", pct(detail_row["Predicted_Failure_Probability"]))
            d2.metric("Risk tier", str(detail_row["Relative_Risk_Tier"]))
            d3.metric("Priority rank", f"{int(detail_row['Priority_Rank']):,}")
            extras = {EXPORT_LABELS.get(c, c): detail_row[c] for c in display_cols
                      if c not in ("Priority_Rank", "Relative_Risk_Tier", "Predicted_Failure_Probability")}
            st.dataframe(pd.DataFrame({"Field": list(extras), "Value": [str(v) for v in extras.values()]}),
                         hide_index=True, use_container_width=True)
            st.caption("Open the *Restaurant Lookup & SHAP* tab for the restaurant summary and SHAP analysis figure.")

    st.download_button(
        "Download filtered priority queue CSV",
        data=to_csv_bytes(table_labeled),
        file_name="filtered_restaurant_priority_queue.csv",
        mime="text/csv",
    )
    st.caption("The CSV contains exactly the rows, columns and sort order shown above.")


# ============================ 4. LOOKUP & SHAP ===============================
@st.cache_resource(show_spinner=False)
def load_model(path_string, mtime):
    import joblib  # imported lazily so the dashboard still runs without it
    return joblib.load(path_string)


def compute_local_shap(rid):
    """Return (DataFrame[Feature, SHAP_Value, Feature_Value], message, note).

    Order of preference:
      1. A precomputed local SHAP file (local_shap_values.csv).
      2. Live computation from the saved model and its feature matrix.
    If neither is possible, returns (None, explanation, None) - never invented values.
    """
    # --- 1. precomputed local SHAP values -----------------------------------
    if local_shap_path is not None:
        local = read_csv_safe(local_shap_path)
        if local is not None:
            id_col = find_col(local, CANONICAL_COLUMNS["Restaurant_ID"])
            if id_col is not None:
                local = local.assign(**{id_col: local[id_col].astype(str)})
                feat_col = find_col(local, ["Feature"])
                val_col = find_col(local, ["SHAP_Value", "SHAP", "Value"])
                if feat_col and val_col:  # long format
                    sub = local[local[id_col] == rid]
                    if sub.empty:
                        return None, "This restaurant is not present in the local SHAP file.", None
                    fv_col = find_col(local, ["Feature_Value"])
                    out = pd.DataFrame({
                        "Feature": sub[feat_col].astype(str),
                        "SHAP_Value": pd.to_numeric(sub[val_col], errors="coerce"),
                        "Feature_Value": sub[fv_col].astype(str) if fv_col else "",
                    }).dropna(subset=["SHAP_Value"])
                    return out, None, f"Precomputed values from {local_shap_path.name}."
                sub = local[local[id_col] == rid]  # wide format
                if sub.empty:
                    return None, "This restaurant is not present in the local SHAP file.", None
                row = sub.iloc[0].drop(labels=[id_col])
                row = pd.to_numeric(row, errors="coerce").dropna()
                row = row[[c for c in row.index if norm(c) not in ("basevalue", "expectedvalue")]]
                out = pd.DataFrame({"Feature": row.index.astype(str), "SHAP_Value": row.values, "Feature_Value": ""})
                return out, None, f"Precomputed values from {local_shap_path.name}."

    # --- 2. live computation -------------------------------------------------
    if model_path is None or model_features_path is None:
        return None, (
            "Individual SHAP explanations are not available: no local SHAP file "
            "(local_shap_values.csv) or saved model + feature file "
            "(risk_model.joblib + model_features.csv) was found. The global SHAP file cannot "
            "explain a single restaurant, so no explanation is shown."
        ), None
    try:
        import numpy as np
        import shap
    except ImportError as exc:
        return None, f"Required package not installed for live SHAP: {exc.name}.", None

    try:
        feats = read_csv_safe(model_features_path)
        id_col = find_col(feats, CANONICAL_COLUMNS["Restaurant_ID"])
        if feats is None or id_col is None:
            return None, "The model feature file has no Restaurant_ID column.", None
        feats = feats.assign(**{id_col: feats[id_col].astype(str)})
        row = feats[feats[id_col] == rid]
        if row.empty:
            return None, "No model features were found for this restaurant.", None
        row = row.iloc[[0]].drop(columns=[id_col])
        # drop any non-feature columns that slipped in
        for extra in ("Predicted_Failure_Probability", "Relative_Risk_Tier", "Priority_Rank"):
            if extra in row.columns:
                row = row.drop(columns=[extra])

        model = load_model(str(model_path), model_path.stat().st_mtime)
        if hasattr(model, "named_steps") or hasattr(model, "steps"):
            steps = model.steps
            estimator = steps[-1][1]
            if len(steps) > 1:
                pre = model[:-1]
                X = pre.transform(row)
                try:
                    names = list(pre.get_feature_names_out())
                except Exception:  # noqa: BLE001
                    names = [f"feature_{i}" for i in range(X.shape[1])]
            else:
                X, names = row, list(row.columns)
        else:
            estimator, X, names = model, row, list(row.columns)
        if hasattr(X, "toarray"):
            X = X.toarray()
        X_arr = np.asarray(X, dtype=float)

        explainer = shap.TreeExplainer(estimator)
        values = explainer.shap_values(X_arr)
        if isinstance(values, list):  # older SHAP: one array per class
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:  # (samples, features, classes)
            values = values[..., -1]
        values = values[0]

        # Check the explanation matches this restaurant's stored prediction.
        note = "Computed live with SHAP TreeExplainer on the saved model; contributions are in model-score units (e.g. log-odds)."
        try:
            live_p = float(model.predict_proba(row)[:, 1][0])
            stored_p = float(queue.loc[queue["Restaurant_ID"] == rid, "Predicted_Failure_Probability"].iloc[0])
            if abs(live_p - stored_p) > 0.02:
                note += (f" WARNING: model re-scoring gives {live_p:.1%} but the queue shows {stored_p:.1%}; "
                         "the saved model or features may not match the queue snapshot.")
        except Exception:  # noqa: BLE001
            pass

        out = pd.DataFrame({
            "Feature": names,
            "SHAP_Value": values,
            "Feature_Value": [str(v) for v in X_arr[0]],
        })
        return out, None, note
    except Exception as exc:  # noqa: BLE001
        return None, f"Could not compute the individual SHAP explanation: {exc}", None


def build_report(row, shap_df, history_rows):
    lines = [
        "# Restaurant Risk Report (decision-support only)",
        "",
        f"- Restaurant: {row.get('Restaurant_Name', '') if HAS_NAME else ''} (ID {row['Restaurant_ID']})".replace("  ", " "),
        f"- Predicted failure probability: {row['Predicted_Failure_Probability']:.1%}",
        f"- Relative risk tier: {row['Relative_Risk_Tier']}",
        f"- Prediction context: probability of failing the next routine Canvass inspection. {SNAPSHOT_LABEL}.",
    ]
    for col in ("Inspection_Date", "Previous_Result", "Violation_Count", "Hazard_Count", "Address", "Community_Area"):
        if col in row.index and pd.notna(row[col]):
            lines.append(f"- {EXPORT_LABELS.get(col, col)}: {row[col]}")
    if history_rows is not None and not history_rows.empty:
        lines += ["", f"## Historical inspections on file: {len(history_rows)}"]
    if shap_df is not None and not shap_df.empty:
        top = shap_df.reindex(shap_df["SHAP_Value"].abs().sort_values(ascending=False).index).head(10)
        lines += ["", "## Top contributing features (SHAP)"]
        for _, r in top.iterrows():
            direction = "increases" if r["SHAP_Value"] > 0 else "decreases"
            lines.append(f"- {prettify_feature(r['Feature'])}: {direction} the risk score ({r['SHAP_Value']:+.3f})")
    lines += [
        "", "## Limitations",
        "- The tiers are relative ranking bands, not official food-safety grades.",
        "- Predictions support, but do not replace, professional inspection judgment.",
        "- This document is not an official inspection certificate.",
    ]
    return "\n".join(lines)


with lookup_tab:
    st.subheader("Restaurant lookup & SHAP analysis")
    st.caption("Searches the full scored queue (sidebar filters do not restrict this lookup).")

    search = st.text_input(
        "Search by restaurant " + ("ID or name" if HAS_NAME else "ID"),
        key="lookup_search",
        placeholder="Type to search; leave blank to browse the highest-ranked restaurants",
    ).strip()

    matches = queue
    if search:
        mask = queue["Restaurant_ID"].str.contains(search, case=False, na=False, regex=False)
        if HAS_NAME:
            mask |= queue["Restaurant_Name"].astype(str).str.contains(search, case=False, na=False, regex=False)
        matches = queue[mask]

    if matches.empty:
        st.warning(f"No restaurant matches '{search}'.")
    else:
        matches = matches.head(500)
        match_ids = matches["Restaurant_ID"].tolist()
        preselect = st.session_state.get("selected_id")
        start = match_ids.index(preselect) if preselect in match_ids else 0

        def match_label(rid):
            r = matches.loc[matches["Restaurant_ID"] == rid].iloc[0]
            nm = f" - {r['Restaurant_Name']}" if HAS_NAME and pd.notna(r.get("Restaurant_Name")) else ""
            return f"{rid}{nm}  ({r['Predicted_Failure_Probability']:.1%}, rank {int(r['Priority_Rank'])})"

        if len(queue[mask] if search else queue) > 500:
            st.caption("Showing the top 500 matches - refine the search to narrow further.")
        rid = st.selectbox("Matching restaurants", match_ids, index=start, format_func=match_label)
        st.session_state["selected_id"] = str(rid)
        row = queue.loc[queue["Restaurant_ID"] == rid].iloc[0]

        # --- individual risk summary
        st.markdown(f"### {row['Restaurant_Name'] if HAS_NAME and pd.notna(row.get('Restaurant_Name')) else 'Restaurant ' + str(rid)}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted failure probability", pct(row["Predicted_Failure_Probability"]))
        m2.metric("Risk tier", str(row["Relative_Risk_Tier"]))
        m3.metric("Priority rank", f"{int(row['Priority_Rank']):,}")
        m4.metric("Most recent inspection",
                  str(row["Inspection_Date"]) if "Inspection_Date" in row.index and pd.notna(row["Inspection_Date"]) else "N/A")
        st.info(
            "**Prediction context:** the model estimates the probability that this restaurant fails its "
            f"*next routine Canvass inspection*. {SNAPSHOT_LABEL}. The tier is a relative ranking band."
        )

        detail_fields = [c for c in ["Restaurant_ID", "Restaurant_Name", "Address", "Community_Area",
                                     "Previous_Result", "Violation_Count", "Hazard_Count"] if c in row.index]
        if detail_fields:
            st.dataframe(
                pd.DataFrame({"Field": [EXPORT_LABELS.get(c, c) for c in detail_fields],
                              "Value": [str(row[c]) for c in detail_fields]}),
                hide_index=True, use_container_width=True,
            )

        # --- historical results, if a history file exists
        history_rows = None
        hist_all = read_csv_safe(history_path)
        if hist_all is not None:
            h_id = find_col(hist_all, CANONICAL_COLUMNS["Restaurant_ID"])
            if h_id is not None:
                history_rows = hist_all[hist_all[h_id].astype(str) == str(rid)]
                with st.expander(f"Historical inspection records on file ({len(history_rows)})"):
                    if history_rows.empty:
                        st.write("No historical records found for this restaurant.")
                    else:
                        st.dataframe(history_rows, use_container_width=True, hide_index=True)

        # --- SHAP analysis figure
        st.markdown("### SHAP analysis figure")
        shap_df_local = None
        if shap_figure_path is not None:
            st.image(shap_figure_path, width=1000)
            st.caption(
                "This SHAP summary shows how features influence the model across the full dataset. "
                "It provides context for the selected restaurant, but is not a restaurant-specific explanation."
            )
        else:
            st.info("The SHAP analysis figure (shap_summary.png) was not found in the outputs folder.")

        # --- optional report
        report_text = build_report(row, shap_df_local, history_rows)
        with st.expander("Restaurant report (optional)"):
            st.markdown(report_text)
        st.download_button(
            "Download restaurant report (Markdown)",
            data=report_text.encode("utf-8"),
            file_name=f"restaurant_report_{re.sub(r'[^A-Za-z0-9_-]', '_', str(rid))}.md",
            mime="text/markdown",
        )

# ============================ 5. VIOLATION ANALYTICS =========================
VIOLATION_CODE_RE = re.compile(r"^\s*(\d+)\.\s*(.*?)(?:\s*-\s*Comments:.*)?$", re.DOTALL)


def build_violation_frame(df):
    """Return (inspections_df, violations_df, notes).

    inspections_df: one row per source row (inspection) with parsed fields.
    violations_df: one row per violation (empty if violation-level detail is unavailable).
    """
    notes = []
    work = df.copy()
    date_col = find_col(work, ["Inspection Date", "Inspection_Date", "Date"])
    res_col = find_col(work, ["Results", "Result", "Inspection_Result"])
    type_col = find_col(work, ["Inspection Type", "Inspection_Type"])
    area_col = find_col(work, CANONICAL_COLUMNS["Community_Area"])
    id_col = find_col(work, CANONICAL_COLUMNS["Restaurant_ID"])
    code_col = find_col(work, ["Violation_Code", "Violation Code", "Code"])
    cat_col = find_col(work, ["Violation_Category", "Violation Category", "Violation_Description",
                              "Violation Description", "Description"])
    sev_col = find_col(work, ["Severity", "Violation_Severity", "Violation Severity", "Risk_Level"])
    raw_col = find_col(work, ["Violations", "Violation_Text", "Violation Text"])

    work["_year"] = pd.to_datetime(work[date_col], errors="coerce").dt.year if date_col else pd.NA
    work["_type"] = work[type_col].astype(str) if type_col else "Unknown"
    work["_area"] = work[area_col].astype(str) if area_col else "Unknown"
    work["_result"] = work[res_col].astype(str) if res_col else "Unknown"
    work["_fail"] = work["_result"].str.contains("fail", case=False, na=False) if res_col else pd.NA
    work["_id"] = work[id_col].astype(str) if id_col else ""
    work["_inspection_key"] = work.index.astype(str)

    viol = pd.DataFrame()
    if raw_col is not None:
        # Chicago-style raw text: violations separated by "|", each "NN. DESCRIPTION - Comments: ..."
        work["_raw"] = work[raw_col].astype("string")
        has_text = work["_raw"].fillna("").str.strip() != ""
        work["_n_viol"] = work["_raw"].fillna("").map(
            lambda s: len([p for p in s.split("|") if p.strip()]))
        exploded = work.loc[has_text, ["_inspection_key", "_year", "_type", "_area", "_fail", "_raw"]].copy()
        exploded["_piece"] = exploded["_raw"].str.split("|")
        exploded = exploded.explode("_piece")
        exploded = exploded[exploded["_piece"].fillna("").str.strip() != ""]
        parsed = exploded["_piece"].map(lambda s: VIOLATION_CODE_RE.match(s))
        exploded["Violation_Code"] = parsed.map(lambda m: m.group(1) if m else "Unparsed")
        exploded["Violation_Category"] = parsed.map(
            lambda m: re.sub(r"\s+", " ", m.group(2)).strip()[:90] if m else "Unparsed")
        viol = exploded
        notes.append("Violation codes/descriptions were parsed from the raw violation text field.")
    elif code_col is not None or cat_col is not None:
        # Already one row per violation.
        work["Violation_Code"] = work[code_col].astype(str) if code_col else work[cat_col].astype(str)
        work["Violation_Category"] = work[cat_col].astype(str) if cat_col else work["Violation_Code"]
        keep = work[(work["Violation_Code"].str.strip() != "") & work["Violation_Code"].notna()
                    & (work["Violation_Code"] != "nan")]
        viol = keep
        if id_col and date_col:
            work["_inspection_key"] = work["_id"] + "|" + work[date_col].astype(str)
            viol = viol.assign(_inspection_key=viol["_id"] + "|" + viol[date_col].astype(str))
        work["_n_viol"] = 0
        counts = viol.groupby("_inspection_key").size()
        work["_n_viol"] = work["_inspection_key"].map(counts).fillna(0).astype(int)
        notes.append("Data appears to be violation-level (one row per violation); inspections are "
                     "identified by restaurant ID + inspection date.")
    else:
        work["_n_viol"] = pd.NA

    viol = viol.reset_index(drop=True)
    if sev_col is not None and not viol.empty and sev_col in viol.columns:
        viol = viol.assign(Severity=viol[sev_col].astype(str))
    return work, viol, notes


with violation_tab:
    st.subheader("Violation analytics")
    st.caption("Historical inspection records - separate from predicted probabilities and from the sidebar filters.")

    hist = read_csv_safe(history_path)
    if hist is None:
        if {"Violation_Count"}.issubset(queue.columns) or {"Hazard_Count"}.issubset(queue.columns):
            st.markdown("**Basic count analytics from the priority queue** "
                        "(each bar counts *restaurants*, using their latest recorded counts; "
                        "the sidebar filters apply).")
            cols = st.columns(2)
            for col_widget, count_col, title in zip(
                cols, ["Violation_Count", "Hazard_Count"],
                ["Violation count per restaurant", "Hazard count per restaurant"],
            ):
                if count_col in filtered.columns and not filtered.empty:
                    vals = pd.to_numeric(filtered[count_col], errors="coerce").dropna()
                    if not vals.empty:
                        fig = px.histogram(vals, nbins=min(30, int(vals.max()) + 1 if vals.max() > 0 else 5),
                                           title=title, labels={"value": count_col.replace("_", " "),
                                                                "count": "Restaurants"})
                        fig.update_layout(showlegend=False, yaxis_title="Restaurants")
                        col_widget.plotly_chart(fig, use_container_width=True)
            if "Violation_Count" in filtered.columns and not filtered.empty:
                tmp = filtered.assign(_vc=pd.to_numeric(filtered["Violation_Count"], errors="coerce")).dropna(subset=["_vc"])
                if not tmp.empty:
                    tmp["_bin"] = tmp["_vc"].clip(upper=10).astype(int).astype(str).replace({"10": "10+"})
                    grp = tmp.groupby("_bin")["Predicted_Failure_Probability"].agg(["mean", "size"]).reset_index()
                    order = [str(i) for i in range(10)] + ["10+"]
                    grp["_bin"] = pd.Categorical(grp["_bin"], categories=order, ordered=True)
                    grp = grp.sort_values("_bin")
                    fig = px.bar(grp, x="_bin", y="mean", hover_data={"size": True},
                                 labels={"_bin": "Violation count", "mean": "Mean PREDICTED failure probability"},
                                 title="Mean predicted failure probability by violation count (model output)")
                    fig.update_yaxes(tickformat=".0%")
                    st.plotly_chart(fig, use_container_width=True)
    else:
        inspections, violations, parse_notes = build_violation_frame(hist)
        for n in parse_notes:
            st.caption(n)

        has_violation_detail = not violations.empty
        if not has_violation_detail:
            st.warning("The history file has no usable violation text, code or description columns.")

        # --- local filters (historical records)
        st.markdown("**Historical-record filters**")
        f1, f2, f3, f4 = st.columns(4)
        years = sorted(int(y) for y in inspections["_year"].dropna().unique())
        if len(years) >= 2:
            with f1:
                yr_lo, yr_hi = st.select_slider("Inspection year range", options=years, value=(years[0], years[-1]))
        elif years:
            yr_lo = yr_hi = years[0]
            f1.caption(f"Year: {years[0]}")
        else:
            yr_lo = yr_hi = None
            f1.caption("No inspection dates available.")
        types = sorted(inspections["_type"].dropna().unique().tolist())
        with f2:
            sel_types = st.multiselect("Inspection type", types, default=[], help="Empty = all types.") if len(types) > 1 else []
        cat_options = sorted(violations["Violation_Category"].dropna().unique().tolist()) if has_violation_detail else []
        with f3:
            sel_cats = st.multiselect("Violation category", cat_options, default=[], help="Empty = all.") if cat_options else []
        areas_h = sorted(a for a in inspections["_area"].dropna().unique().tolist() if a not in ("Unknown", "nan"))
        with f4:
            sel_areas_h = st.multiselect("Community area", areas_h, default=[], help="Empty = all.") if areas_h else []

        def keep_inspection(df):
            m = pd.Series(True, index=df.index)
            if yr_lo is not None:
                m &= df["_year"].between(yr_lo, yr_hi)
            if sel_types:
                m &= df["_type"].isin(sel_types)
            if sel_areas_h:
                m &= df["_area"].isin(sel_areas_h)
            return m

        insp_f = inspections[keep_inspection(inspections)]
        viol_f = violations[keep_inspection(violations)] if has_violation_detail else violations
        if sel_cats and has_violation_detail:
            keys = set(viol_f.loc[viol_f["Violation_Category"].isin(sel_cats), "_inspection_key"])
            viol_f = viol_f[viol_f["Violation_Category"].isin(sel_cats)]
            insp_f = insp_f[insp_f["_inspection_key"].isin(keys)]

        if insp_f.empty:
            st.warning("No historical records match the selected filters.")
        else:
            k1, k2, k3 = st.columns(3)
            k1.metric("Inspections (records)", f"{len(insp_f):,}")
            k2.metric("Individual violations recorded", f"{len(viol_f):,}" if has_violation_detail else "N/A")
            if "_fail" in insp_f.columns and insp_f["_fail"].notna().any():
                k3.metric("Historical failure rate", pct(insp_f["_fail"].astype(float).mean()),
                          help="Observed share of inspections with a Fail result. Not a predicted probability.")
            if has_violation_detail:
                blank = int((insp_f["_n_viol"].fillna(0) == 0).sum())
                st.caption(
                    f"{blank:,} inspection record(s) have no recorded violation text. A blank field is not "
                    "proof that the inspection passed (many inspection types record no violations or are not pass/fail)."
                )

            if has_violation_detail and not viol_f.empty:
                top_n = int_slider("Violations to show", 5, min(30, max(5, viol_f["Violation_Category"].nunique())), 15)
                freq = (viol_f.groupby(["Violation_Code", "Violation_Category"]).size()
                        .reset_index(name="Violations").sort_values("Violations", ascending=False))
                freq["Label"] = freq["Violation_Code"].astype(str) + ". " + freq["Violation_Category"].str.slice(0, 60)
                top = freq.head(top_n).copy()

                left, right = st.columns(2)
                with left:
                    fig = px.bar(top.sort_values("Violations"), x="Violations", y="Label", orientation="h",
                                 title="Most frequent violations (count of violations)",
                                 labels={"Label": "Violation"})
                    st.plotly_chart(fig, use_container_width=True)
                with right:
                    pareto = top.copy()
                    pareto["Cumulative share"] = pareto["Violations"].cumsum() / freq["Violations"].sum()
                    fig = go.Figure()
                    fig.add_bar(x=pareto["Violation_Code"].astype(str), y=pareto["Violations"], name="Violations")
                    fig.add_scatter(x=pareto["Violation_Code"].astype(str), y=pareto["Cumulative share"],
                                    name="Cumulative share of all violations", yaxis="y2", mode="lines+markers")
                    fig.update_layout(title="Pareto chart of violation codes",
                                      xaxis_title="Violation code",
                                      yaxis=dict(title="Violations"),
                                      yaxis2=dict(title="Cumulative share", overlaying="y", side="right",
                                                  tickformat=".0%", range=[0, 1.05]),
                                      legend=dict(orientation="h", y=-0.25))
                    st.plotly_chart(fig, use_container_width=True)

            left, right = st.columns(2)
            with left:
                if has_violation_detail:
                    counts = insp_f["_n_viol"].fillna(0).astype(int).clip(upper=20)
                    fig = px.histogram(counts, nbins=21,
                                       title="Distribution of violations per inspection (capped at 20)",
                                       labels={"value": "Violations in the inspection"})
                    fig.update_layout(showlegend=False, yaxis_title="Inspections")
                    st.plotly_chart(fig, use_container_width=True)
            with right:
                if has_violation_detail and "Severity" in viol_f.columns:
                    sev = viol_f["Severity"].value_counts().rename_axis("Severity").reset_index(name="Violations")
                    fig = px.bar(sev, x="Severity", y="Violations", title="Violations by severity (count of violations)")
                    st.plotly_chart(fig, use_container_width=True)
                elif has_violation_detail:
                    st.info("Critical / serious / minor classification is not available: the history file has "
                            "no severity column.")

            # Historical failure rate by violation category (inspection-level, association only)
            if has_violation_detail and insp_f["_fail"].notna().any():
                st.markdown("**Historical failure rate by violation category** (observed, inspection-level)")
                min_insp = int_slider("Minimum inspections per category", 1, 200, 30)
                pairs = viol_f[["_inspection_key", "Violation_Category"]].drop_duplicates()
                fail_map = insp_f.set_index("_inspection_key")["_fail"]
                fail_map = fail_map[~fail_map.index.duplicated(keep="first")]
                pairs = pairs.assign(_fail=pairs["_inspection_key"].map(fail_map)).dropna(subset=["_fail"])
                rate = (pairs.groupby("Violation_Category")["_fail"]
                        .agg(Failure_Rate="mean", Inspections="size").reset_index())
                rate = rate[rate["Inspections"] >= min_insp].sort_values("Failure_Rate", ascending=False).head(15)
                if rate.empty:
                    st.info("No violation category meets the minimum inspection count.")
                else:
                    rate["Label"] = rate["Violation_Category"].str.slice(0, 60)
                    fig = px.bar(rate.sort_values("Failure_Rate"), x="Failure_Rate", y="Label", orientation="h",
                                 hover_data={"Inspections": True, "Label": False},
                                 title="Share of inspections containing the violation that had a Fail result",
                                 labels={"Failure_Rate": "Historical failure rate", "Label": "Violation"})
                    fig.update_xaxes(tickformat=".0%")
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Counts are inspections. This shows an observed association in historical data "
                               "only - it does not show that a violation causes a failure, and it is separate "
                               "from the model's predicted probabilities.")

            # Trend by year
            yearly_v = insp_f.dropna(subset=["_year"])
            if yearly_v["_year"].nunique() >= 2:
                agg = yearly_v.groupby("_year").agg(Inspections=("_year", "size"),
                                                    Avg_violations=("_n_viol", lambda s: pd.to_numeric(s, errors="coerce").mean()))
                agg = agg.reset_index()
                fig = px.line(agg, x="_year", y="Avg_violations", markers=True, hover_data={"Inspections": True},
                              labels={"_year": "Inspection year", "Avg_violations": "Average violations per inspection"},
                              title="Average violations per inspection by year")
                st.plotly_chart(fig, use_container_width=True)

# ============================ 7. COMMUNITY AREAS =============================
with community_tab:
    st.subheader("Community-area risk analysis")
    st.caption(
        "Area-level figures summarize model-predicted failure probabilities, not observed "
        "failure rates or official safety grades. Areas with small samples can have unstable averages. "
        "This tab uses the full scored queue and ignores the sidebar filters."
    )

    community = read_csv_safe(community_summary_path)
    if community is None and HAS_AREA:
        # Fallback: build the summary directly from the queue.
        grp = queue.dropna(subset=["Community_Area"]).groupby("Community_Area")
        community = grp.agg(
            Restaurant_Count=("Restaurant_ID", "size"),
            Mean_Predicted_Failure_Probability=("Predicted_Failure_Probability", "mean"),
            High_Risk_Restaurant_Count=("Relative_Risk_Tier", lambda s: int((s == HIGH_TIER).sum())),
        ).reset_index()
        st.info("community_area_risk_summary.csv was not found; the summary below was computed "
                "from the queue's community-area column.")
    elif community is None:
        st.warning(
            "community_area_risk_summary.csv was not found and the queue has no community-area column. "
            "Run community_area_analysis.py and copy its output into this folder."
        )

    if community is not None:
        required_area_cols = {"Community_Area", "Restaurant_Count",
                              "Mean_Predicted_Failure_Probability", "High_Risk_Restaurant_Count"}
        missing_area_cols = required_area_cols - set(community.columns)
        if missing_area_cols:
            st.error("Community-area summary is missing columns: " + ", ".join(sorted(missing_area_cols)))
        else:
            community = community.copy()
            community["Restaurant_Count"] = pd.to_numeric(community["Restaurant_Count"], errors="coerce")
            community["Mean_Predicted_Failure_Probability"] = pd.to_numeric(
                community["Mean_Predicted_Failure_Probability"], errors="coerce")
            community["High_Risk_Restaurant_Count"] = pd.to_numeric(
                community["High_Risk_Restaurant_Count"], errors="coerce").fillna(0)
            community["Community_Area"] = community["Community_Area"].astype(str)

            max_count = int(community["Restaurant_Count"].max()) if community["Restaurant_Count"].notna().any() else 1
            min_sample = int_slider(
                "Minimum restaurants in area", 1, max(1, max_count), min(30, max(1, max_count)),
                help="Areas below this sample size are excluded from the ranking chart.",
            )
            community["Meets_Sample_Threshold"] = community["Restaurant_Count"] >= min_sample

            area_pick = st.multiselect("Limit to specific community areas (optional)",
                                       sorted(community["Community_Area"].unique().tolist()), default=[])
            scope = community[community["Community_Area"].isin(area_pick)] if area_pick else community

            eligible = (scope[scope["Meets_Sample_Threshold"]]
                        .dropna(subset=["Mean_Predicted_Failure_Probability"])
                        .sort_values("Mean_Predicted_Failure_Probability", ascending=False))

            unmapped_count = None
            if HAS_AREA:
                unmapped_count = int(queue["Community_Area"].isna().sum())
            elif community_detail_path is not None:
                detail = read_csv_safe(community_detail_path)
                if detail is not None and "Community_Area" in detail.columns:
                    unmapped_count = int(detail["Community_Area"].isna().sum())

            k1, k2, k3 = st.columns(3)
            k1.metric("Areas represented", f"{len(scope)}")
            k2.metric("Areas meeting sample threshold", f"{len(eligible)}")
            k3.metric("Restaurants without an area", f"{unmapped_count:,}" if unmapped_count is not None else "N/A")

            small = scope[~scope["Meets_Sample_Threshold"]]
            

            if eligible.empty:
                st.warning("No community areas meet the selected minimum sample size.")
            else:
                top_n = int_slider("Areas to show", 1, min(30, len(eligible)), min(15, len(eligible)))
                left, right = st.columns([3, 2])
                with left:
                    chart_data = eligible.head(top_n).sort_values("Mean_Predicted_Failure_Probability")
                    fig = px.bar(
                        chart_data, x="Mean_Predicted_Failure_Probability", y="Community_Area", orientation="h",
                        hover_data=["Restaurant_Count", "High_Risk_Restaurant_Count"],
                        title="Highest mean predicted risk by community area (model output)",
                        labels={
                            "Mean_Predicted_Failure_Probability": "Mean predicted failure probability",
                            "Community_Area": "Chicago community area",
                            "Restaurant_Count": "Restaurants in queue",
                            "High_Risk_Restaurant_Count": "High-priority count",
                        },
                    )
                    if chart_data["Mean_Predicted_Failure_Probability"].max() <= 1:
                        fig.update_xaxes(tickformat=".0%")
                    st.plotly_chart(fig, use_container_width=True)
                with right:
                    st.markdown("**Highest-ranked areas**")
                    st.dataframe(
                        eligible[["Community_Area", "Restaurant_Count", "Mean_Predicted_Failure_Probability",
                                  "High_Risk_Restaurant_Count"]].head(top_n),
                        use_container_width=True, hide_index=True,
                    )

            st.markdown("**Community-area summary table**")
            summary_table = scope.sort_values("Mean_Predicted_Failure_Probability", ascending=False).copy()
            summary_table["Small_Sample_Warning"] = (~summary_table["Meets_Sample_Threshold"]).map(
                {True: f"< {min_sample} restaurants", False: ""})
            summary_table = summary_table.drop(columns=["Meets_Sample_Threshold"])
            st.dataframe(summary_table, use_container_width=True, hide_index=True)
            st.download_button(
                "Download community-area summary CSV",
                data=to_csv_bytes(summary_table),
                file_name="community_area_risk_summary.csv",
                mime="text/csv",
            )
            st.caption("No boundary map is shown: only point coordinates and area identifiers are used here.")

# ============================ 8. MODEL PERFORMANCE ===========================
with model_tab:
    st.subheader("Model performance")
    metrics_df = read_csv_safe(metrics_path)
    metrics = {}
    if metrics_df is None:
        st.warning("Model metrics CSV not found beside app.py.")
    else:
        metric_col = find_col(metrics_df, ["Metric", "Name"])
        value_col = find_col(metrics_df, ["Value", "Score"])
        if metric_col and value_col:  # long format
            metrics = dict(zip(metrics_df[metric_col].astype(str), metrics_df[value_col]))
        elif len(metrics_df):  # wide format: first row
            metrics = {str(k): v for k, v in metrics_df.iloc[0].to_dict().items()}

        current = {
            "PR-AUC": metric_from_dict(metrics, ("prauc",), ("averageprecision",), ("pr", "auc")),
            "ROC-AUC": metric_from_dict(metrics, ("rocauc",), ("roc", "auc"), ("auc",)),
            "Brier score": metric_from_dict(metrics, ("brier",)),
            "Precision @ top 5%": metric_from_dict(metrics, ("precision", "top5"), ("precision", "5")),
        }
        cols = st.columns(len(current), gap="large")
        for widget, (label, val) in zip(cols, current.items()):
            widget.metric(label, fmt_metric(val))
        if any(v is None for v in current.values()):
            missing_m = [k for k, v in current.items() if v is None]
            st.warning("Not found in the metrics file: " + ", ".join(missing_m) +
                       ". Check the metric names in model_test_metrics.csv.")

        st.markdown("**Previously reported results vs. the current metrics file**")
        cmp_rows = []
        for label, old in PREVIOUSLY_REPORTED.items():
            cur = current.get(label)
            cmp_rows.append({"Metric": label, "Previously reported": f"{old:.3f}",
                             "Current metrics file": fmt_metric(cur)})
        st.dataframe(pd.DataFrame(cmp_rows), hide_index=True, use_container_width=True)
        st.caption("Previously reported figures are historical references only; the numbers above "
                   "from the current metrics file are the ones to treat as current.")

        with st.expander("Full metrics file"):
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # --- optional diagnostic plots from supplementary files
    pr_df = read_csv_safe(pr_curve_path)
    cal_df = read_csv_safe(calibration_path)
    if pr_df is not None or cal_df is not None:
        left, right = st.columns(2)
        if pr_df is not None:
            p_col, r_col = find_col(pr_df, ["Precision"]), find_col(pr_df, ["Recall"])
            with left:
                if p_col and r_col:
                    fig = px.line(pr_df, x=r_col, y=p_col, title="Precision-recall curve (test set)",
                                  labels={r_col: "Recall", p_col: "Precision"})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("pr_curve.csv needs Precision and Recall columns.")
        if cal_df is not None:
            x_col = find_col(cal_df, ["Mean_Predicted_Probability", "Predicted", "Mean_Predicted", "prob_pred"])
            y_col = find_col(cal_df, ["Observed_Failure_Rate", "Observed", "Fraction_Positives", "prob_true"])
            with right:
                if x_col and y_col:
                    fig = go.Figure()
                    fig.add_scatter(x=cal_df[x_col], y=cal_df[y_col], mode="lines+markers", name="Model")
                    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                                    line=dict(dash="dash"))
                    fig.update_layout(title="Calibration plot (test set)",
                                      xaxis_title="Mean predicted probability",
                                      yaxis_title="Observed failure rate")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("calibration_curve.csv needs predicted and observed columns.")


    cm_df = read_csv_safe(confusion_path)
    if cm_df is not None:
        st.markdown("**Confusion matrix** (threshold as documented in the evaluation notebook)")
        cm = cm_df.set_index(cm_df.columns[0]) if not pd.api.types.is_numeric_dtype(cm_df.iloc[:, 0]) else cm_df
        fig = px.imshow(cm, text_auto=True, aspect="auto", labels=dict(x="Predicted", y="Actual", color="Count"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Confirm the decision threshold used for this matrix was chosen on validation data, "
                   "not tuned on the final test set.")

    cmp_df = read_csv_safe(comparison_path)
    if cmp_df is not None:
        st.markdown("**Model comparison**")
        st.dataframe(cmp_df, use_container_width=True, hide_index=True)

    st.markdown("**Model limitations**")
    st.markdown(
        "- Predictive power is modest: the model ranks restaurants better than chance but misses many failures.\n"
        "- Performance was measured on a held-out test period and may drift as inspection practices change.\n"
        "- Predictions rely on historical inspection records; restaurants with little history are less reliable.\n"
        "- Observed associations (and SHAP attributions) are not causal explanations.\n"
        "- Models and thresholds should be selected on validation data, never tuned on the final test set."
    )

    st.subheader("Global SHAP feature importance")
    st.caption("Dataset-wide importance. It does not explain any individual restaurant - see the "
               "*Restaurant Lookup & SHAP* tab for that.")
    shap_global = read_csv_safe(shap_path)
    if shap_global is not None and {
        "Feature", "Mean_Absolute_SHAP"
    }.issubset(shap_global.columns):
        top_n_shap = int_slider("Features to show", 5, min(30, max(5, len(shap_global))), min(15, max(5, len(shap_global))))
        sg = shap_global.sort_values("Mean_Absolute_SHAP", ascending=False).head(top_n_shap).copy()
        sg["Feature"] = sg["Feature"].map(prettify_feature)
        sg = sg.sort_values("Mean_Absolute_SHAP")
        fig = px.bar(sg, x="Mean_Absolute_SHAP", y="Feature", orientation="h",
                     title="Top features by mean absolute SHAP value",
                     labels={"Mean_Absolute_SHAP": "Mean |SHAP value|", "Feature": "Model feature"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("SHAP feature CSV is missing expected columns: Feature, Mean_Absolute_SHAP.")

st.divider()
st.caption(
    "Research prototype only. Predictions should support, not replace, professional inspection decisions. "
    "Risk tiers are relative ranking bands, not official food-safety grades. "
    f"{SNAPSHOT_LABEL}."
    + ("" if HAS_NAME else " The dataset does not include restaurant names; this dashboard uses Restaurant_ID.")
)
