from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import pydeck as pdk

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

st.set_page_config(
    page_title="Chicago Food Safety Risk System",
    page_icon="🍽️",
    layout="wide",
)

def first_existing(*names):
    for name in names:
        for path in (OUTPUT_DIR / name, BASE_DIR / name):
            if path.exists(): return path
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

st.title("Chicago Restaurant Health Inspection Risk System")
st.caption(
    "Prototype decision-support dashboard. Predicted probabilities estimate the model's "
    "risk of failure at the next routine Canvass inspection; they are not official safety ratings."
)

if queue_path is None:
    st.error("Risk queue CSV not found. Place restaurant_risk_priority_queue_corrected.csv beside app.py.")
    st.stop()

queue = load_csv(str(queue_path))
required_cols = {"Latitude", "Longitude", "Predicted_Failure_Probability"}
missing = required_cols - set(queue.columns)
if missing:
    st.error(f"Risk queue is missing required columns: {', '.join(sorted(missing))}")
    st.stop()

queue["Predicted_Failure_Probability"] = pd.to_numeric(
    queue["Predicted_Failure_Probability"], errors="coerce"
)
queue["Latitude"] = pd.to_numeric(queue["Latitude"], errors="coerce")
queue["Longitude"] = pd.to_numeric(queue["Longitude"], errors="coerce")
queue = queue.dropna(subset=["Predicted_Failure_Probability"]).copy()
map_df = queue.dropna(subset=["Latitude", "Longitude"]).copy()

with st.sidebar:
    st.header("Filters")
    if "Relative_Risk_Tier" in queue.columns:
        tiers = sorted(queue["Relative_Risk_Tier"].dropna().unique().tolist())
        selected_tiers = st.multiselect("Risk tier", tiers, default=tiers)
    else:
        selected_tiers = []
    min_prob, max_prob = float(queue["Predicted_Failure_Probability"].min()), float(queue["Predicted_Failure_Probability"].max())
    prob_range = st.slider(
        "Predicted failure probability",
        min_value=0.0,
        max_value=1.0,
        value=(max(0.0, min_prob), min(1.0, max_prob)),
        step=0.01,
    )
    search_text = st.text_input("Restaurant ID contains", placeholder="Enter an ID")

filtered = queue.copy()
if selected_tiers and "Relative_Risk_Tier" in filtered.columns:
    filtered = filtered[filtered["Relative_Risk_Tier"].isin(selected_tiers)]
filtered = filtered[
    filtered["Predicted_Failure_Probability"].between(prob_range[0], prob_range[1])
]
if search_text.strip() and "Restaurant_ID" in filtered.columns:
    filtered = filtered[
        filtered["Restaurant_ID"].astype(str).str.contains(search_text.strip(), case=False, na=False)
    ]

# KPI cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Restaurants scored", f"{len(queue):,}")
c2.metric("High-priority restaurants", f"{(queue.get('Relative_Risk_Tier', pd.Series(dtype=str)) == 'Red - highest 10%').sum():,}" if "Relative_Risk_Tier" in queue.columns else "N/A")
c3.metric("Average predicted risk", f"{queue['Predicted_Failure_Probability'].mean():.1%}")
c4.metric("Restaurants in current filter", f"{len(filtered):,}")

overview_tab, map_tab, queue_tab, trends_tab, community_tab, model_tab = st.tabs(
    ["Overview", "Risk Map", "Inspector Queue", "Seasonal Trends", "Community Areas", "Model & SHAP"]
)

with overview_tab:
    st.subheader("Risk distribution")
    left, right = st.columns(2)
    with left:
        if "Relative_Risk_Tier" in queue.columns:
            tier_counts = queue["Relative_Risk_Tier"].value_counts().rename_axis("Risk tier").reset_index(name="Restaurants")
            fig = px.bar(tier_counts, x="Risk tier", y="Restaurants", title="Restaurants by relative risk tier")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Relative_Risk_Tier is not available in the selected queue file.")
    with right:
        fig = px.histogram(
            queue,
            x="Predicted_Failure_Probability",
            nbins=30,
            title="Distribution of predicted failure probabilities",
            labels={"Predicted_Failure_Probability": "Predicted failure probability"},
        )
        st.plotly_chart(fig, use_container_width=True)
    st.info("Relative tiers are ranking bands (top 10%, next 20%, remaining 70%), not official inspection grades.")

with map_tab:
    st.subheader("Geographic distribution of predicted risk")
    st.caption("Map points show restaurant locations; zoom in to inspect clusters. Colors represent predicted probability.")
    if map_df.empty:
        st.warning("No valid coordinates are available in the queue.")
    else:
        if selected_tiers and "Relative_Risk_Tier" in map_df.columns:
            map_filtered = map_df[map_df["Relative_Risk_Tier"].isin(selected_tiers)]
        else:
            map_filtered = map_df.copy()
        map_filtered = map_filtered[
            map_filtered["Predicted_Failure_Probability"].between(prob_range[0], prob_range[1])
        ]
        if search_text.strip() and "Restaurant_ID" in map_filtered.columns:
            map_filtered = map_filtered[
                map_filtered["Restaurant_ID"].astype(str).str.contains(search_text.strip(), case=False, na=False)
            ]
        if map_filtered.empty:
            st.warning("No restaurants match the current filters.")
        else:
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_filtered,
                get_position="[Longitude, Latitude]",
                get_radius=65,
                get_fill_color="[255 * Predicted_Failure_Probability, 120 * (1 - Predicted_Failure_Probability), 40]",
                pickable=True,
                auto_highlight=True,
            )
            view_state = pdk.ViewState(
                latitude=float(map_filtered["Latitude"].median()),
                longitude=float(map_filtered["Longitude"].median()),
                zoom=10,
                pitch=0,
            )
            tooltip = {
                "html": "<b>Restaurant ID:</b> {Restaurant_ID}<br/>"
                        "<b>Predicted failure probability:</b> {Predicted_Failure_Probability}<br/>"
                        "<b>Tier:</b> {Relative_Risk_Tier}",
                "style": {"backgroundColor": "steelblue", "color": "white"},
            }
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))
            st.caption(f"Showing {len(map_filtered):,} mapped restaurants.")

with queue_tab:
    st.subheader("Inspector priority queue")
    st.caption("Sorted by predicted failure probability. Use the filters in the sidebar to narrow the list.")
    display_cols = [
        c for c in [
            "Priority_Rank", "Restaurant_ID", "Relative_Risk_Tier",
            "Predicted_Failure_Probability", "Inspection Date", "Results",
            "Violation_Count", "Hazard_Count", "Latitude", "Longitude"
        ] if c in filtered.columns
    ]
    table = filtered[display_cols].sort_values(
        "Predicted_Failure_Probability", ascending=False
    ) if "Predicted_Failure_Probability" in display_cols else filtered[display_cols]
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered priority queue CSV",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="filtered_restaurant_priority_queue.csv",
        mime="text/csv",
    )

with trends_tab:
    st.subheader("Monthly inspection failure rates")
    if season_path:
        season = load_csv(str(season_path))
        if {"Month", "Failure_Rate_Percent"}.issubset(season.columns):
            fig = px.line(
                season,
                x="Month",
                y="Failure_Rate_Percent",
                markers=True,
                title="Historical failure rate by month (Canvass inspections only)",
                labels={"Failure_Rate_Percent": "Failure rate (%)", "Month": "Inspection month"},
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(season, use_container_width=True, hide_index=True)
        else:
            st.warning("Seasonal CSV is missing expected columns.")
    else:
        st.warning("seasonal_failure_by_month.csv not found beside app.py.")

with community_tab:
    st.subheader("Community-area risk analysis")
    st.caption(
        "Area-level figures summarize model-predicted failure probabilities, not observed "
        "failure rates or official safety grades. Areas with small samples can have unstable averages."
    )

    if community_summary_path is None:
        st.warning(
            "community_area_risk_summary.csv was not found beside app.py. "
            "Run community_area_analysis.py and copy its output into this folder."
        )
    else:
        community = load_csv(str(community_summary_path))
        required_area_cols = {
            "Community_Area",
            "Restaurant_Count",
            "Mean_Predicted_Failure_Probability",
            "High_Risk_Restaurant_Count",
        }
        missing_area_cols = required_area_cols - set(community.columns)

        if missing_area_cols:
            st.error(
                "Community-area summary is missing columns: "
                + ", ".join(sorted(missing_area_cols))
            )
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

            min_sample = st.slider(
                "Minimum restaurants in area",
                min_value=1,
                max_value=max(1, int(community["Restaurant_Count"].max())),
                value=min(30, max(1, int(community["Restaurant_Count"].max()))),
                step=1,
                help="Areas below this sample size are excluded from the ranking chart."
            )

            eligible = community[
                community["Restaurant_Count"] >= min_sample
            ].copy()
            eligible = eligible.dropna(
                subset=["Mean_Predicted_Failure_Probability"]
            ).sort_values(
                "Mean_Predicted_Failure_Probability", ascending=False
            )

            total_areas = len(community)
            eligible_areas = len(eligible)
            unmapped_count = None

            if community_detail_path is not None:
                detail = load_csv(str(community_detail_path))
                if "Community_Area" in detail.columns:
                    unmapped_count = int(detail["Community_Area"].isna().sum())

            k1, k2, k3 = st.columns(3)
            k1.metric("Areas represented", f"{total_areas}")
            k2.metric("Areas meeting sample threshold", f"{eligible_areas}")
            k3.metric(
                "Unmapped restaurants",
                f"{unmapped_count:,}" if unmapped_count is not None else "N/A"
            )

            if eligible.empty:
                st.warning("No community areas meet the selected minimum sample size.")
            else:
                left, right = st.columns([3, 2])
                with left:
                    top_n = st.slider(
                        "Areas to show",
                        min_value=5,
                        max_value=max(5, min(30, len(eligible))),
                        value=min(15, max(5, len(eligible))),
                        step=1,
                    )
                    chart_data = eligible.head(top_n).sort_values(
                        "Mean_Predicted_Failure_Probability"
                    )
                    fig = px.bar(
                        chart_data,
                        x="Mean_Predicted_Failure_Probability",
                        y="Community_Area",
                        orientation="h",
                        hover_data=[
                            "Restaurant_Count",
                            "High_Risk_Restaurant_Count",
                        ],
                        title="Highest mean predicted risk by community area",
                        labels={
                            "Mean_Predicted_Failure_Probability": "Mean predicted probability (%)",
                            "Community_Area": "Chicago community area",
                            "Restaurant_Count": "Restaurants in queue",
                            "High_Risk_Restaurant_Count": "High-priority count",
                        },
                    )
                    st.plotly_chart(fig, use_container_width=True)

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

            st.markdown("**Community-area summary table**")
            st.dataframe(
                community.sort_values(
                    "Mean_Predicted_Failure_Probability", ascending=False
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download community-area summary CSV",
                data=community.to_csv(index=False).encode("utf-8"),
                file_name="community_area_risk_summary.csv",
                mime="text/csv",
            )

with model_tab:
    st.subheader("Model evaluation")
    if metrics_path:
        metrics = load_csv(str(metrics_path))
        st.dataframe(metrics, use_container_width=True, hide_index=True)
    else:
        st.warning("Model metrics CSV not found beside app.py.")

    st.subheader("Global SHAP feature importance")
    if shap_path:
        shap_df = load_csv(str(shap_path))
        if {"Feature", "Mean_Absolute_SHAP"}.issubset(shap_df.columns):
            shap_df = shap_df.sort_values("Mean_Absolute_SHAP", ascending=True).tail(15)
            fig = px.bar(
                shap_df,
                x="Mean_Absolute_SHAP",
                y="Feature",
                orientation="h",
                title="Top features by mean absolute SHAP value",
                labels={"Mean_Absolute_SHAP": "Mean |SHAP value|", "Feature": "Model feature"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("SHAP feature CSV is missing expected columns.")
    else:
        st.warning("SHAP feature importance CSV not found beside app.py.")

st.divider()
st.caption(
    "Research prototype only. Predictions should support, not replace, professional inspection decisions. "
    "The dataset may not include restaurant names or addresses; this dashboard uses Restaurant_ID."
)
