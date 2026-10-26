"""
Chicago Restaurant Health Inspection Risk System
Premium UI + restored downloads, report card map, and clean SHAP tab.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import joblib

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

st.set_page_config(
    page_title="Chicago Food Safety Risk System",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Premium floating metric cards (works in light + dark) ---
st.markdown(
    """
<style>
div[data-testid="metric-container"] {
    background-color: var(--secondary-background-color);
    border: 1px solid rgba(128, 128, 128, 0.22);
    padding: 14px 18px;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.08);
}
div[data-testid="stMetricValue"] {
    font-size: 2.05rem !important;
    font-weight: 800 !important;
}
div[data-testid="stMetricLabel"] {
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    opacity: 0.75;
}
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
</style>
""",
    unsafe_allow_html=True,
)


def first_existing(*names):
    for name in names:
        for path in (OUTPUT_DIR / name, BASE_DIR / name):
            if path.exists():
                return path
    return None


@st.cache_data
def load_csv(path_string):
    return pd.read_csv(path_string, low_memory=False)


queue_path = first_existing(
    "restaurant_risk_priority_queue_corrected.csv",
    "restaurant_risk_priority_queue.csv",
)
metrics_path = first_existing("model_test_metrics.csv", "model_test_metrics(2).csv")
season_path = first_existing("seasonal_failure_by_month.csv")
shap_path = first_existing("shap_feature_importance.csv", "shap_feature_importance(1).csv")
community_summary_path = first_existing("community_area_risk_summary.csv")
community_detail_path = first_existing("restaurant_community_area_risk.csv")

if queue_path is None:
    st.error("Risk queue CSV not found. Place restaurant_risk_priority_queue.csv beside app.py.")
    st.stop()

queue = load_csv(str(queue_path))
queue["Predicted_Failure_Probability"] = pd.to_numeric(
    queue["Predicted_Failure_Probability"], errors="coerce"
)
queue["Latitude"] = pd.to_numeric(queue.get("Latitude"), errors="coerce")
queue["Longitude"] = pd.to_numeric(queue.get("Longitude"), errors="coerce")
queue = queue.dropna(subset=["Predicted_Failure_Probability"]).copy()
map_df = queue.dropna(subset=["Latitude", "Longitude"]).copy()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1046/1046784.png", width=58)
    st.title("Control Panel")
    st.caption("Filter the dashboard dynamically")

    if "Relative_Risk_Tier" in queue.columns:
        tiers = sorted(queue["Relative_Risk_Tier"].dropna().unique().tolist())
        selected_tiers = st.multiselect("Priority tiers", tiers, default=tiers)
    else:
        selected_tiers = []

    min_p = float(queue["Predicted_Failure_Probability"].min())
    max_p = float(queue["Predicted_Failure_Probability"].max())
    prob_range = st.slider(
        "Risk probability range",
        0.0,
        1.0,
        (max(0.0, min_p), min(1.0, max_p)),
        0.01,
    )
    search_text = st.text_input("Search Restaurant ID", placeholder="e.g. 10452")

    st.markdown("---")
    st.markdown("### Risk legend")
    st.markdown(
        """
- <span style="color:#EF4444;">●</span> **Top 10%** highest risk  
- <span style="color:#F59E0B;">●</span> **Next 20%** elevated  
- <span style="color:#10B981;">●</span> **Remaining 70%** standard
""",
        unsafe_allow_html=True,
    )

# Apply filters
filtered = queue.copy()
if selected_tiers and "Relative_Risk_Tier" in filtered.columns:
    filtered = filtered[filtered["Relative_Risk_Tier"].isin(selected_tiers)]
filtered = filtered[
    filtered["Predicted_Failure_Probability"].between(prob_range[0], prob_range[1])
]
if search_text.strip() and "Restaurant_ID" in filtered.columns:
    filtered = filtered[
        filtered["Restaurant_ID"]
        .astype(str)
        .str.contains(search_text.strip(), case=False, na=False)
    ]

# ---------------- Header KPIs ----------------
st.title("Chicago Restaurant Health Inspection Risk System")
st.caption(
    "Decision-support prototype. Scores estimate risk of failing the next routine Canvass inspection — not official grades."
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Facilities scored", f"{len(queue):,}")
k2.metric(
    "High-priority (Red)",
    f"{(queue.get('Relative_Risk_Tier', pd.Series(dtype=str)) == 'Red - highest 10%').sum():,}"
    if "Relative_Risk_Tier" in queue.columns
    else "N/A",
)
k3.metric("Average predicted risk", f"{queue['Predicted_Failure_Probability'].mean():.1%}")
k4.metric("In current filter", f"{len(filtered):,}")

(
    overview_tab,
    map_tab,
    queue_tab,
    report_tab,
    simulator_tab,
    trends_tab,
    community_tab,
    model_tab,
) = st.tabs(
    [
        "Overview",
        "Risk Map",
        "Inspector Queue",
        "Report Card",
        "Simulator",
        "Seasonality",
        "Community",
        "AI Model",
    ]
)

# ---------------- Overview ----------------
with overview_tab:
    st.subheader("Citywide risk distribution")
    left, right = st.columns(2)
    with left:
        if "Relative_Risk_Tier" in queue.columns:
            tier_counts = (
                queue["Relative_Risk_Tier"]
                .value_counts()
                .rename_axis("Risk tier")
                .reset_index(name="Restaurants")
            )
            fig = px.bar(
                tier_counts,
                x="Risk tier",
                y="Restaurants",
                color="Risk tier",
                color_discrete_map={
                    "Red - highest 10%": "#EF4444",
                    "Yellow - next 20%": "#F59E0B",
                    "Green - remaining 70%": "#10B981",
                },
                title="Facilities by priority tier",
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
        else:
            st.info("Relative_Risk_Tier not found in queue file.")
    with right:
        fig = px.histogram(
            queue,
            x="Predicted_Failure_Probability",
            nbins=30,
            color_discrete_sequence=["#6366F1"],
            title="Distribution of predicted failure probabilities",
            labels={"Predicted_Failure_Probability": "Predicted failure probability"},
        )
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    st.info(
        "Tiers are relative ranking bands (top 10% / next 20% / rest), not official inspection grades."
    )

# ---------------- Risk Map ----------------
with map_tab:
    st.subheader("Geographic distribution of predicted risk")
    mode = st.radio(
        "Map mode",
        ["Standard Scatter (Points)", "3D Hexagon Density (Clusters)"],
        horizontal=True,
    )

    if map_df.empty:
        st.warning("No valid coordinates in the queue.")
    else:
        map_filtered = map_df.copy()
        if selected_tiers and "Relative_Risk_Tier" in map_filtered.columns:
            map_filtered = map_filtered[map_filtered["Relative_Risk_Tier"].isin(selected_tiers)]
        map_filtered = map_filtered[
            map_filtered["Predicted_Failure_Probability"].between(prob_range[0], prob_range[1])
        ]
        if search_text.strip() and "Restaurant_ID" in map_filtered.columns:
            map_filtered = map_filtered[
                map_filtered["Restaurant_ID"]
                .astype(str)
                .str.contains(search_text.strip(), case=False, na=False)
            ]

        if map_filtered.empty:
            st.warning("No restaurants match the current filters.")
        else:
            view = pdk.ViewState(
                latitude=float(map_filtered["Latitude"].median()),
                longitude=float(map_filtered["Longitude"].median()),
                zoom=10.5,
                pitch=45 if "3D" in mode else 0,
                bearing=-8,
            )
            if "3D" in mode:
                layer = pdk.Layer(
                    "HexagonLayer",
                    data=map_filtered,
                    get_position="[Longitude, Latitude]",
                    radius=250,
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
                tip = {"html": "<b>Facilities in hex:</b> {elevationValue}"}
            else:

                def color_row(r):
                    t = str(r.get("Relative_Risk_Tier", ""))
                    if "Red" in t:
                        return [239, 68, 68, 220]
                    if "Yellow" in t:
                        return [245, 158, 11, 220]
                    return [16, 185, 129, 160]

                map_filtered = map_filtered.copy()
                map_filtered["color"] = map_filtered.apply(color_row, axis=1)
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=map_filtered,
                    get_position="[Longitude, Latitude]",
                    get_radius=80,
                    get_fill_color="color",
                    pickable=True,
                    auto_highlight=True,
                )
                tip = {
                    "html": "<b>ID:</b> {Restaurant_ID}<br/>"
                    "<b>Risk:</b> {Predicted_Failure_Probability}<br/>"
                    "<b>Tier:</b> {Relative_Risk_Tier}",
                    "style": {"backgroundColor": "#333", "color": "white"},
                }
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tip))
            st.caption(f"Showing {len(map_filtered):,} mapped restaurants.")

# ---------------- Inspector Queue ----------------
with queue_tab:
    st.subheader("Inspector priority queue")
    st.caption("Highest predicted risk first. Use sidebar filters to narrow the list.")
    cols = [
        c
        for c in [
            "Priority_Rank",
            "Restaurant_ID",
            "Relative_Risk_Tier",
            "Predicted_Failure_Probability",
            "Inspection Date",
            "Results",
            "Violation_Count",
            "Hazard_Count",
            "Latitude",
            "Longitude",
        ]
        if c in filtered.columns
    ]
    table = (
        filtered[cols].sort_values("Predicted_Failure_Probability", ascending=False)
        if "Predicted_Failure_Probability" in cols
        else filtered[cols]
    )
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Predicted_Failure_Probability": st.column_config.ProgressColumn(
                "Predicted Risk",
                format="%.1f%%",
                min_value=0.0,
                max_value=1.0,
            ),
            "Priority_Rank": st.column_config.NumberColumn("Rank", format="#%d"),
        },
    )
    st.download_button(
        "Download filtered queue CSV",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="filtered_priority_queue.csv",
        mime="text/csv",
    )

# ---------------- Report Card (restored map + download) ----------------
with report_tab:
    st.subheader("Facility risk report card")
    st.caption("Plain-English profile for one restaurant. Easy for inspectors and managers.")

    if "Restaurant_ID" not in queue.columns:
        st.warning("Restaurant_ID column missing.")
    else:
        ids = sorted(queue["Restaurant_ID"].astype(str).unique().tolist())
        selected_id = st.selectbox("Select Restaurant ID", ids)
        fac = queue[queue["Restaurant_ID"].astype(str) == selected_id].iloc[0]

        prob = float(fac.get("Predicted_Failure_Probability", 0.0))
        tier = str(fac.get("Relative_Risk_Tier", "N/A"))
        rank = fac.get("Priority_Rank", None)
        rank_txt = f"#{int(rank):,}" if pd.notna(rank) else "N/A"
        city_avg = float(queue["Predicted_Failure_Probability"].mean())
        delta = prob - city_avg

        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Predicted failure risk", f"{prob:.1%}")
        g2.metric("Risk tier", tier)
        g3.metric("Citywide rank", rank_txt)
        g4.metric("vs city average", f"{delta:+.1%}")

        left, right = st.columns([1.1, 1])
        with left:
            # Gauge
            fig_g = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={"suffix": "%"},
                    title={"text": "Risk level"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "rgba(100,100,100,0.35)"},
                        "steps": [
                            {"range": [0, 20], "color": "#10B981"},
                            {"range": [20, 35], "color": "#F59E0B"},
                            {"range": [35, 100], "color": "#EF4444"},
                        ],
                        "threshold": {
                            "line": {"color": "#111827", "width": 4},
                            "thickness": 0.75,
                            "value": prob * 100,
                        },
                    },
                )
            )
            fig_g.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
            st.plotly_chart(fig_g, use_container_width=True, theme="streamlit")

            # Plain English summary
            raw_res = str(fac.get("Results", "")).strip().lower()
            if raw_res in {"1", "fail", "failed"}:
                res_txt, res_col = "Failed", "#EF4444"
            else:
                res_txt, res_col = "Passed / Conditional", "#10B981"

            viol = int(fac.get("Violation_Count", 0) or 0)
            haz = int(fac.get("Hazard_Count", 0) or 0)
            last_c = int(fac.get("Last_Canvass_Failed", 0) or 0)
            c_txt, c_col = ("Failed", "#EF4444") if last_c == 1 else ("Passed", "#10B981")
            last_date = str(fac.get("Inspection Date", "N/A"))[:10]

            st.markdown(
                f"""
<div style="background: var(--secondary-background-color); border-left: 6px solid {res_col};
            padding: 16px 18px; border-radius: 10px;">
  <p style="margin:0 0 8px 0;"><b>Facility ID:</b> {selected_id}</p>
  <p style="margin:0 0 8px 0;"><b>Last inspected:</b> {last_date}</p>
  <p style="margin:0 0 8px 0;"><b>Most recent result:</b>
     <span style="color:{res_col}; font-weight:700;">{res_txt}</span></p>
  <p style="margin:0 0 8px 0;"><b>Historical violations:</b> {viol}</p>
  <p style="margin:0 0 8px 0;"><b>Critical hazards:</b> {haz}</p>
  <p style="margin:0;"><b>Previous routine canvass:</b>
     <span style="color:{c_col}; font-weight:700;">{c_txt}</span></p>
</div>
""",
                unsafe_allow_html=True,
            )

            # Download this facility row
            one = queue[queue["Restaurant_ID"].astype(str) == selected_id]
            st.download_button(
                "Download this facility report (CSV)",
                data=one.to_csv(index=False).encode("utf-8"),
                file_name=f"facility_report_{selected_id}.csv",
                mime="text/csv",
            )

        with right:
            st.markdown("### Location")
            lat, lon = fac.get("Latitude"), fac.get("Longitude")
            if pd.notna(lat) and pd.notna(lon):
                pin = pd.DataFrame(
                    {"lat": [float(lat)], "lon": [float(lon)], "Restaurant_ID": [selected_id]}
                )
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=pin,
                    get_position="[lon, lat]",
                    get_radius=140,
                    get_fill_color="[239, 68, 68, 230]",
                    pickable=True,
                )
                view = pdk.ViewState(latitude=float(lat), longitude=float(lon), zoom=14, pitch=0)
                st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view))
            else:
                st.info("No coordinates for this facility.")

# ---------------- Simulator ----------------
with simulator_tab:
    st.subheader("What-if risk simulator")
    st.caption("Move the sliders to see how compliance history changes predicted risk.")
    c1, c2 = st.columns([1, 1.15])
    with c1:
        v = st.slider("Mean violations (last 3)", 0.0, 20.0, 3.5, 0.5)
        h = st.slider("Mean critical hazards (last 3)", 0.0, 10.0, 1.0, 0.5)
        pfr = st.slider("Historical failure rate", 0.0, 1.0, 0.25, 0.05)
        dsf = st.slider("Days since last failure", 0, 1000, 180, 30)
    with c2:
        logit = -1.2 + 0.18 * v + 0.28 * h + 1.4 * pfr - 0.0012 * dsf
        sim = float(np.clip(1 / (1 + np.exp(-logit)), 0.02, 0.98))
        if sim >= 0.35:
            tier_s, col_s = "High priority (Top 10%)", "#EF4444"
        elif sim >= 0.20:
            tier_s, col_s = "Medium priority (Next 20%)", "#F59E0B"
        else:
            tier_s, col_s = "Low priority (Remaining 70%)", "#10B981"
        st.markdown(
            f"""
<div style="background: var(--secondary-background-color); border-left: 8px solid {col_s};
            padding: 28px; border-radius: 12px; text-align:center; margin-top: 8px;">
  <div style="opacity:0.75; font-weight:700;">Simulated failure risk</div>
  <div style="font-size: 3.6rem; font-weight: 900; color:{col_s}; margin: 8px 0;">{sim:.1%}</div>
  <div style="font-size: 1.15rem; font-weight:700;">{tier_s}</div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.info(
            "More violations / higher past failure rate raise risk. Longer time since last failure lowers risk."
        )

# ---------------- Seasonality ----------------
with trends_tab:
    st.subheader("Monthly inspection failure rates")
    if season_path is None:
        st.warning("seasonal_failure_by_month.csv not found.")
    else:
        season = load_csv(str(season_path))
        if {"Month", "Failure_Rate_Percent"}.issubset(season.columns):
            xcol = "Month_Name" if "Month_Name" in season.columns else "Month"
            fig = px.bar(
                season,
                x=xcol,
                y="Failure_Rate_Percent",
                text="Failure_Rate_Percent",
                color="Failure_Rate_Percent",
                color_continuous_scale="RdYlGn_r",
                title="Historical failure rate by month (%)",
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
            st.dataframe(season, use_container_width=True, hide_index=True)
            st.download_button(
                "Download seasonal CSV",
                data=season.to_csv(index=False).encode("utf-8"),
                file_name="seasonal_failure_by_month.csv",
                mime="text/csv",
            )
        else:
            st.warning("Seasonal CSV missing expected columns.")

# ---------------- Community (download restored) ----------------
with community_tab:
    st.subheader("Community-area risk analysis")
    st.caption(
        "Area averages of model-predicted risk (not official grades). Small samples can be unstable."
    )

    if community_summary_path is None:
        st.warning(
            "community_area_risk_summary.csv not found. Run community_area_analysis.py and place the CSV beside app.py."
        )
    else:
        community = load_csv(str(community_summary_path))
        need = {
            "Community_Area",
            "Restaurant_Count",
            "Mean_Predicted_Failure_Probability",
            "High_Risk_Restaurant_Count",
        }
        missing = need - set(community.columns)
        if missing:
            st.error("Community summary missing: " + ", ".join(sorted(missing)))
        else:
            community["Restaurant_Count"] = pd.to_numeric(
                community["Restaurant_Count"], errors="coerce"
            )
            community["Mean_Predicted_Failure_Probability"] = pd.to_numeric(
                community["Mean_Predicted_Failure_Probability"], errors="coerce"
            )
            community["High_Risk_Restaurant_Count"] = pd.to_numeric(
                community["High_Risk_Restaurant_Count"], errors="coerce"
            ).fillna(0)

            max_n = max(1, int(community["Restaurant_Count"].max()))
            min_sample = st.slider(
                "Minimum restaurants in area",
                1,
                max_n,
                min(30, max_n),
                1,
            )
            eligible = community[community["Restaurant_Count"] >= min_sample].dropna(
                subset=["Mean_Predicted_Failure_Probability"]
            )
            eligible = eligible.sort_values(
                "Mean_Predicted_Failure_Probability", ascending=False
            )

            unmapped = None
            if community_detail_path is not None:
                detail = load_csv(str(community_detail_path))
                if "Community_Area" in detail.columns:
                    unmapped = int(detail["Community_Area"].isna().sum())

            a, b, c = st.columns(3)
            a.metric("Areas represented", f"{len(community)}")
            b.metric("Areas meeting threshold", f"{len(eligible)}")
            c.metric("Unmapped restaurants", f"{unmapped:,}" if unmapped is not None else "N/A")

            if eligible.empty:
                st.warning("No areas meet the sample threshold.")
            else:
                left, right = st.columns([3, 2])
                with left:
                    top_n = st.slider(
                        "Areas to show",
                        5,
                        max(5, min(30, len(eligible))),
                        min(15, len(eligible)),
                        1,
                    )
                    chart = eligible.head(top_n).sort_values(
                        "Mean_Predicted_Failure_Probability"
                    )
                    fig = px.bar(
                        chart,
                        x="Mean_Predicted_Failure_Probability",
                        y="Community_Area",
                        orientation="h",
                        color="Mean_Predicted_Failure_Probability",
                        color_continuous_scale="Reds",
                        hover_data=["Restaurant_Count", "High_Risk_Restaurant_Count"],
                        title="Highest mean predicted risk by community area",
                    )
                    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
                with right:
                    st.markdown("**Highest-ranked areas**")
                    st.dataframe(
                        eligible[
                            [
                                "Community_Area",
                                "Restaurant_Count",
                                "Mean_Predicted_Failure_Probability",
                                "High_Risk_Restaurant_Count",
                            ]
                        ].head(top_n),
                        use_container_width=True,
                        hide_index=True,
                    )

            st.markdown("**Full community summary**")
            st.dataframe(
                community.sort_values(
                    "Mean_Predicted_Failure_Probability", ascending=False
                ),
                use_container_width=True,
                hide_index=True,
            )
            # TEAM NOTE: download restored
            st.download_button(
                "Download community-area summary CSV",
                data=community.to_csv(index=False).encode("utf-8"),
                file_name="community_area_risk_summary.csv",
                mime="text/csv",
            )

# ---------------- AI Model / SHAP (cleaned) ----------------
with model_tab:
    st.subheader("Model evaluation")
    st.caption(
        "Held-out test metrics. Best operational readout is Precision@Top 5/10% vs base failure rate."
    )
    if metrics_path is None:
        st.warning("model_test_metrics.csv not found.")
    else:
        metrics = load_csv(str(metrics_path))
        st.dataframe(metrics, use_container_width=True, hide_index=True)

    st.subheader("What drives the model? (SHAP)")
    st.caption(
        "Longer bars = stronger average impact on predicted risk. Pink waterfall below explains one high-risk example."
    )

    if shap_path is not None:
        shap_df = load_csv(str(shap_path))
        feat_col = "Feature" if "Feature" in shap_df.columns else shap_df.columns[0]
        val_col = (
            "Mean_Absolute_SHAP"
            if "Mean_Absolute_SHAP" in shap_df.columns
            else shap_df.columns[1]
        )
        plot_df = shap_df[[feat_col, val_col]].copy()
        plot_df = plot_df.sort_values(val_col, ascending=True).tail(15)
        fig = px.bar(
            plot_df,
            x=val_col,
            y=feat_col,
            orientation="h",
            title="Top features by mean |SHAP|",
            color_discrete_sequence=["#8B5CF6"],
            labels={val_col: "Mean |SHAP value|", feat_col: "Feature"},
        )
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    else:
        st.warning("shap_feature_importance.csv not found.")

    st.markdown("#### Example explanation for a high-risk facility")
    shap_img = first_existing("shap_highest_risk_waterfall.png", "shap_summary.png")
    if shap_img:
        # TEAM NOTE: Constrained width prevents image from taking over full screen
        left_img, right_help = st.columns([2.2, 1.3])
        with left_img:
            st.image(
                str(shap_img),
                caption="SHAP waterfall for one high-risk facility (pink = increases risk).",
                width=620,
            )
        with right_help:
            st.markdown(
                """
**How to read this**

1. Start at the bottom baseline.  
2. Each bar is one inspection-history signal.  
3. **Right / pink** → pushes failure risk **up**.  
4. **Left** → pushes risk **down**.  
5. Top of the stack ≈ this restaurant’s predicted risk.

**What matters most here**
- Violation counts  
- Past failure rate  
- Recent hazards / admin findings  

*This is an explanation tool — not an official grade.*
"""
            )
    else:
        st.info("Run `python explain_model.py` to generate SHAP waterfall images.")

st.divider()
st.caption(
    "Research prototype only. Predictions support — not replace — professional inspection decisions."
)