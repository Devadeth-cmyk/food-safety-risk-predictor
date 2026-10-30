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

st.set_page_config(
    page_title="Chicago Restaurant Health Inspection Risk System",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Reduce excessive vertical spacing */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 1.5rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* Improve spacing between dashboard sections */
div[data-testid="stVerticalBlock"] {
    gap: 1rem;
}

/* Compact headings */
h1 {
    margin-bottom: 0.4rem !important;
}

h2 {
    margin-top: 0.8rem !important;
    margin-bottom: 0.5rem !important;
}

h3 {
    margin-top: 0.5rem !important;
    margin-bottom: 0.5rem !important;
}

/* Keep metric cards compact */
div[data-testid="metric-container"] {
    padding: 10px 14px;
    border-radius: 10px;
}

/* Reduce spacing around tabs */
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1rem;
}

/* Make charts and tables fit more cleanly */
div[data-testid="stPlotlyChart"] {
    margin-bottom: 0.5rem;
}

div[data-testid="stDataFrame"] {
    margin-top: 0.5rem;
}

div[data-testid="metric-container"] {
    background-color: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,.22);
    padding: 14px 18px;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0,0,0,.08);
}
div[data-testid="stMetricValue"] {
    font-size: 2.05rem !important;
    font-weight: 800 !important;
}
div[data-testid="stMetricLabel"] {
    font-size: .95rem !important;
    font-weight: 700 !important;
    opacity: .75;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
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

    lower_to_original = {
        str(c).strip().lower(): c for c in df.columns
    }

    for source, target in aliases.items():
        if target not in df.columns and source in lower_to_original:
            original = lower_to_original[source]
            df.rename(columns={original: target}, inplace=True)

    # Support exports that store probabilities as percentages.
    lower_to_original = {
        str(c).strip().lower(): c for c in df.columns
    }

    if "Predicted_Failure_Probability" not in df.columns:
        pct_col = lower_to_original.get(
            "predicted_next_failure_probability_pct"
        )
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

        df["Relative_Risk_Tier"] = (
            df["Relative_Risk_Tier"].map(normalize_tier)
        )

    if "Predicted_Failure_Probability" in df.columns:
        df["Predicted_Failure_Probability"] = pd.to_numeric(
            df["Predicted_Failure_Probability"],
            errors="coerce",
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
        df["Priority_Rank"] = pd.to_numeric(
            df["Priority_Rank"], errors="coerce"
        )

    if "Restaurant_ID" in df.columns:
        df["Restaurant_ID"] = df["Restaurant_ID"].astype("string")

    return df


def build_inspector_queue(df):
    """
    Build the queue presentation from the actual project data.
    Uses the reference CSV's intended field structure, not its records.
    """
    preferred_columns = [
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
        "Previous_Failure_Count",
        "Historical_Failure_Rate",
    ]

    # Display standard queue fields when available.
    available = [c for c in preferred_columns if c in df.columns]

    # Keep additional source columns available rather than discarding them.
    extra = [c for c in df.columns if c not in available]
    display_columns = available + extra

    table = df[display_columns].copy()

    # Highest predicted risk first.
    if "Predicted_Failure_Probability" in table.columns:
        table = table.sort_values(
            "Predicted_Failure_Probability",
            ascending=False,
            na_position="last",
        )

    # Generate a rank if the source does not contain one.
    if "Priority_Rank" not in table.columns:
        table.insert(
            0,
            "Priority_Rank",
            np.arange(1, len(table) + 1),
        )

    return table


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

queue = queue.dropna(
    subset=["Predicted_Failure_Probability"]
).copy()

if queue.empty:
    st.error("The priority-list file contains no usable risk predictions.")
    st.stop()

if "Relative_Risk_Tier" not in queue.columns:
    # Assign tiers based on risk ranking when no tier field is provided.
    queue = queue.sort_values(
        "Predicted_Failure_Probability",
        ascending=False,
    ).copy()

    percentile = queue["Predicted_Failure_Probability"].rank(
        method="first",
        ascending=False,
        pct=True,
    )

    queue["Relative_Risk_Tier"] = np.select(
        [percentile <= 0.10, percentile <= 0.30],
        ["Red - highest 10%", "Yellow - next 20%"],
        default="Green - remaining 70%",
    )

map_df = queue.dropna(subset=["Latitude", "Longitude"]).copy()


# ============================================================
# Sidebar filters
# ============================================================

with st.sidebar:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/1046/1046784.png",
        width=58,
    )
    st.title("Control Panel")
    st.caption("Filter the dashboard dynamically")

    tiers = sorted(
        queue["Relative_Risk_Tier"].dropna().unique().tolist()
    )

    selected_tiers = st.multiselect(
        "Priority tiers",
        tiers,
        default=tiers,
    )

    min_p = max(
        0.0,
        float(queue["Predicted_Failure_Probability"].min()),
    )
    max_p = min(
        1.0,
        float(queue["Predicted_Failure_Probability"].max()),
    )

    if max_p < min_p:
        min_p, max_p = 0.0, 1.0

    prob_range = st.slider(
        "Risk probability range",
        min_value=0.0,
        max_value=1.0,
        value=(min_p, max_p),
        step=0.01,
    )

    search_text = st.text_input(
        "Search Restaurant ID",
        placeholder="e.g. 10452",
    )

    st.markdown("---")
    st.markdown("### Risk legend")
    st.markdown("""
    - <span style="color:#EF4444;">●</span> **Highest priority**
    - <span style="color:#F59E0B;">●</span> **Elevated priority**
    - <span style="color:#10B981;">●</span> **Standard priority**
    """, unsafe_allow_html=True)


filtered = queue.copy()

if selected_tiers:
    filtered = filtered[
        filtered["Relative_Risk_Tier"].isin(selected_tiers)
    ]

filtered = filtered[
    filtered["Predicted_Failure_Probability"].between(*prob_range)
]

if search_text.strip() and "Restaurant_ID" in filtered.columns:
    filtered = filtered[
        filtered["Restaurant_ID"].astype(str).str.contains(
            search_text.strip(),
            case=False,
            na=False,
        )
    ]


# ============================================================
# Dashboard heading and metrics
# ============================================================

st.title("Chicago Restaurant Health Inspection Risk System")



k1, k2, k3, k4 = st.columns(4)

k1.metric("Facilities scored", f"{len(queue):,}")

high_risk_count = (
    queue["Relative_Risk_Tier"] == "Red - highest 10%"
).sum()

k2.metric("Highest-priority facilities", f"{high_risk_count:,}")

k3.metric(
    "Average predicted risk",
    f"{queue['Predicted_Failure_Probability'].mean():.1%}",
)

k4.metric("In current filter", f"{len(filtered):,}")


# Seasonality tab removed.
overview_tab, map_tab, queue_tab, report_tab, community_tab, model_tab = st.tabs([
    "Overview",
    "Risk Map",
    "Inspector Queue",
    "Report Card",
    "Community",
    "AI Model",
])


# ============================================================
# Overview
# ============================================================

with overview_tab:
    st.subheader("Citywide risk distribution")

    left, right = st.columns(2)

    with left:
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
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.histogram(
            queue,
            x="Predicted_Failure_Probability",
            nbins=30,
            title="Distribution of predicted failure probabilities",
            labels={
                "Predicted_Failure_Probability":
                    "Predicted failure probability"
            },
        )

        st.plotly_chart(fig, use_container_width=True)




# ============================================================
# Risk Map
# ============================================================

with map_tab:
    st.subheader("Geographic distribution of predicted risk")

    mode = st.radio(
        "Map mode",
        ["Standard Scatter (Points)", "3D Hexagon Density (Clusters)"],
        horizontal=True,
    )

    if map_df.empty:
        st.warning(
            "No valid Latitude/Longitude coordinates were found "
            "in the priority-list CSV."
        )
    else:
        map_filtered = map_df.copy()

        if selected_tiers:
            map_filtered = map_filtered[
                map_filtered["Relative_Risk_Tier"].isin(selected_tiers)
            ]

        map_filtered = map_filtered[
            map_filtered["Predicted_Failure_Probability"].between(
                *prob_range
            )
        ]

        if search_text.strip() and "Restaurant_ID" in map_filtered.columns:
            map_filtered = map_filtered[
                map_filtered["Restaurant_ID"].astype(str).str.contains(
                    search_text.strip(),
                    case=False,
                    na=False,
                )
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

                tooltip = {
                    "html": "<b>Facilities in hex:</b> {elevationValue}"
                }

            else:
                def color_row(row):
                    tier = str(row.get("Relative_Risk_Tier", ""))

                    if "Red" in tier:
                        return [239, 68, 68, 220]
                    if "Yellow" in tier:
                        return [245, 158, 11, 220]
                    return [16, 185, 129, 160]

                map_filtered = map_filtered.copy()
                map_filtered["color"] = map_filtered.apply(
                    color_row,
                    axis=1,
                )

                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=map_filtered,
                    get_position="[Longitude, Latitude]",
                    get_radius=80,
                    get_fill_color="color",
                    pickable=True,
                    auto_highlight=True,
                )

                tooltip = {
                    "html": (
                        "<b>ID:</b> {Restaurant_ID}<br/>"
                        "<b>Risk:</b> {Predicted_Failure_Probability}<br/>"
                        "<b>Tier:</b> {Relative_Risk_Tier}"
                    ),
                    "style": {
                        "backgroundColor": "#333",
                        "color": "white",
                    },
                }

            st.pydeck_chart(
                pdk.Deck(
                    layers=[layer],
                    initial_view_state=view,
                    tooltip=tooltip,
                )
            )

            st.caption(
                f"Showing {len(map_filtered):,} mapped restaurants."
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
        str(column).strip().lower(): column
        for column in queue_display.columns
    }

    for source_name, target_name in column_aliases.items():
        if target_name not in queue_display.columns:
            original_name = existing_lower_names.get(source_name)

            if original_name is not None:
                queue_display.rename(
                    columns={original_name: target_name},
                    inplace=True,
                )

    # Create rank if the source CSV does not provide it
    if "Priority_Rank" not in queue_display.columns:
        queue_display = queue_display.sort_values(
            "Predicted_Failure_Probability",
            ascending=False,
            na_position="last",
        )
        queue_display.insert(
            0,
            "Priority_Rank",
            range(1, len(queue_display) + 1),
        )

    # Include only the requested columns.
    # Missing fields are created as empty values rather than using
    # unrelated columns or data from the reference CSV.
    for column in queue_columns:
        if column not in queue_display.columns:
            queue_display[column] = pd.NA

    table = queue_display[queue_columns].copy()

    # Highest predicted failure probability first
    table["Predicted_Failure_Probability"] = pd.to_numeric(
        table["Predicted_Failure_Probability"],
        errors="coerce",
    )

    table = table.sort_values(
        "Predicted_Failure_Probability",
        ascending=False,
        na_position="last",
    )

    # Display-friendly column labels
    table = table.rename(
        columns={
            "Priority_Rank": "Priority rank",
            "Restaurant_ID": "Restaurant_ID",
            "Relative_Risk_Tier": "Relative_Risk_Tier",
            "Predicted_Failure_Probability": "Predicted Failure Probability",
            "Inspection Date": "Inspection Date",
            "Violation_Count": "Violation count",
            "Latitude": "Latitude",
            "Longitude": "Longitude",
            "Results_Label": "Result label",
        }
    )

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Priority rank": st.column_config.NumberColumn(
                "Priority rank",
                format="%d",
            ),
            "Restaurant_ID": st.column_config.TextColumn(
                "Restaurant_ID",
            ),
            "Relative_Risk_Tier": st.column_config.TextColumn(
                "Relative_Risk_Tier",
            ),
            "Predicted Failure Probability": st.column_config.ProgressColumn(
                "Predicted Failure Probability",
                format="%.1f%%",
                min_value=0.0,
                max_value=1.0,
            ),
            "Inspection Date": st.column_config.TextColumn(
                "Inspection Date",
            ),
            "Violation count": st.column_config.NumberColumn(
                "Violation count",
            ),
            "Latitude": st.column_config.NumberColumn(
                "Latitude",
                format="%.6f",
            ),
            "Longitude": st.column_config.NumberColumn(
                "Longitude",
                format="%.6f",
            ),
            "Result label": st.column_config.TextColumn(
                "Result label",
            ),
        },
    )

    st.caption(f"Showing {len(table):,} restaurants.")

    st.download_button(
        "Download filtered Inspector Queue CSV",
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
        st.warning(
            "Restaurant ID column is missing from the priority-list CSV."
        )

    elif queue.empty:
        st.warning("No scored restaurants are available.")

    else:
        ids = sorted(
            queue["Restaurant_ID"]
            .dropna()
            .astype(str)
            .unique()
        )

        selected_id = st.selectbox(
            "Select Restaurant ID",
            ids,
            key="report_card_restaurant_id",
        )

        fac = queue[
            queue["Restaurant_ID"].astype(str) == selected_id
        ].iloc[0]

        prob = float(fac["Predicted_Failure_Probability"])
        tier = str(fac.get("Relative_Risk_Tier", "N/A"))
        rank = fac.get("Priority_Rank", None)

        rank_txt = (
            f"#{int(rank):,}"
            if pd.notna(rank)
            else "N/A"
        )

        # Shorten tier labels for a cleaner UI
        tier_lower = tier.lower()

        if "red" in tier_lower:
            tier_display = "Red - High"
        elif "yellow" in tier_lower:
            tier_display = "Yellow - Medium"
        elif "green" in tier_lower:
            tier_display = "Green - Low"
        else:
            tier_display = tier

        # ----------------------------------------------------
        # Summary metrics
        # ----------------------------------------------------

        g1, g2, g3 = st.columns(3)

        g1.metric(
            "Predicted failure risk",
            f"{prob:.1%}",
        )

        g2.metric(
            "Risk tier",
            tier_display,
        )

        g3.metric(
            "Priority rank",
            rank_txt,
        )

        st.divider()

        # ----------------------------------------------------
        # Gauge and location
        # ----------------------------------------------------

        left, right = st.columns(
            [1, 1],
            gap="medium",
        )

        with left:
            fig_g = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={
                        "suffix": "%",
                        "font": {"size": 38},
                    },
                    title={
                        "text": "Predicted failure risk",
                        "font": {"size": 18},
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100],
                            "ticksuffix": "%",
                        },
                        "bar": {
                            "color": "#94A3B8",
                            "thickness": 0.25,
                        },
                        "bgcolor": "rgba(0,0,0,0)",
                        "borderwidth": 0,
                        "steps": [
                            {
                                "range": [0, 20],
                                "color": "#10B981",
                            },
                            {
                                "range": [20, 35],
                                "color": "#F59E0B",
                            },
                            {
                                "range": [35, 100],
                                "color": "#EF4444",
                            },
                        ],
                        "threshold": {
                            "line": {
                                "color": "#F8FAFC",
                                "width": 3,
                            },
                            "thickness": 0.8,
                            "value": prob * 100,
                        },
                    },
                )
            )

            fig_g.update_layout(
                height=300,
                margin=dict(
                    l=15,
                    r=15,
                    t=55,
                    b=10,
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#F8FAFC"},
            )

            st.plotly_chart(
                fig_g,
                use_container_width=True,
                config={"displayModeBar": False},
            )

        with right:
            st.subheader("Location")

            lat = fac.get("Latitude")
            lon = fac.get("Longitude")

            if (
                pd.notna(lat)
                and pd.notna(lon)
            ):
                lat = float(lat)
                lon = float(lon)

                pin = pd.DataFrame({
                    "lat": [lat],
                    "lon": [lon],
                    "Restaurant_ID": [selected_id],
                })

                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=pin,
                    get_position="[lon, lat]",
                    get_radius=100,
                    get_fill_color="[239, 68, 68, 230]",
                    pickable=True,
                    auto_highlight=True,
                )

                view = pdk.ViewState(
                    latitude=lat,
                    longitude=lon,
                    zoom=14,
                    pitch=0,
                )

                deck = pdk.Deck(
                    layers=[layer],
                    initial_view_state=view,
                    tooltip={
                        "text": "Restaurant ID: {Restaurant_ID}"
                    },
                )

                st.pydeck_chart(
                    deck,
                    use_container_width=True,
                    height=300,
                )

                st.caption(
                    f"Coordinates: {lat:.5f}, {lon:.5f}"
                )

            else:
                st.info(
                    "Location coordinates are unavailable "
                    "for this restaurant."
                )

        # ----------------------------------------------------
        # Restaurant details
        # ----------------------------------------------------

        st.subheader("Restaurant Details")

        result_value = str(
            fac.get(
                "Results_Label",
                fac.get("Results", "N/A"),
            )
        )

        inspection_date = fac.get(
            "Inspection Date",
            fac.get("Inspection_Date", "N/A"),
        )

        if pd.isna(inspection_date):
            inspection_date = "N/A"

        violations = fac.get("Violation_Count", "N/A")
        hazards = fac.get("Hazard_Count", "N/A")

        detail1, detail2, detail3 = st.columns(3)

        detail1.metric(
            "Latest result",
            result_value,
        )

        detail2.metric(
            "Violation count",
            str(violations),
        )

        detail3.metric(
            "Hazard count",
            str(hazards),
        )

        st.caption(f"Restaurant ID: {selected_id}")
        st.caption(f"Latest inspection: {inspection_date}")

        # ----------------------------------------------------
        # Download report
        # ----------------------------------------------------

        one = queue[
            queue["Restaurant_ID"].astype(str) == selected_id
        ]

        st.download_button(
            "Download this facility report (CSV)",
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
    st.caption(
        "Area averages of model-predicted risk"
    )

    if community_summary_path is None:
        st.warning("Neighborhood summary was not found.")

    else:
        community = load_csv(str(community_summary_path))

        if community.empty:
            st.warning(
                "The neighborhood summary exists but contains no data rows."
            )

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
                st.dataframe(
                    community,
                    use_container_width=True,
                    hide_index=True,
                )

            else:
                for col in [
                    "mean_predicted_failure_probability_pct",
                    "restaurants",
                    "high_risk_restaurants",
                ]:
                    community[col] = pd.to_numeric(
                        community[col],
                        errors="coerce",
                    )

                max_n = max(
                    1,
                    int(community["restaurants"].max()),
                )

                min_sample = st.slider(
                    "Minimum restaurants in area",
                    1,
                    max_n,
                    min(30, max_n),
                    1,
                )

                eligible = community[
                    community["restaurants"] >= min_sample
                ].dropna(
                    subset=["mean_predicted_failure_probability_pct"]
                )

                eligible = eligible.sort_values(
                    "mean_predicted_failure_probability_pct",
                    ascending=False,
                )

                a, b, c = st.columns(3)
                a.metric("Areas represented", len(community))
                b.metric("Areas meeting threshold", len(eligible))
                c.metric("Unmapped restaurants", "N/A")

                if not eligible.empty:
                    top_n = st.slider(
                        "Areas to show",
                        1,
                        min(30, len(eligible)),
                        min(15, len(eligible)),
                        1,
                    )

                    chart = eligible.head(top_n).sort_values(
                        "mean_predicted_failure_probability_pct"
                    )

                    fig = px.bar(
                        chart,
                        x="mean_predicted_failure_probability_pct",
                        y="neighborhood",
                        orientation="h",
                        color="mean_predicted_failure_probability_pct",
                        color_continuous_scale="Reds",
                        hover_data=[
                            "restaurants",
                            "high_risk_restaurants",
                        ],
                        title="Highest mean predicted risk by neighborhood",
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    st.dataframe(
                        eligible.head(top_n),
                        use_container_width=True,
                        hide_index=True,
                    )

                st.markdown("**Full neighborhood summary**")

                st.dataframe(
                    community,
                    use_container_width=True,
                    hide_index=True,
                )

                st.download_button(
                    "Download neighborhood summary CSV",
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
        cols[i % 3].metric(
            label,
            fmt.format(metrics[key]),
        )

    st.caption(
        f"Selected classification threshold: "
        f"{metrics['selected_threshold']:.2f}"
    )



    # Original SHAP summary image
    BASE_DIR = Path(__file__).resolve().parent

    shap_image_path = (
        BASE_DIR / "artifacts" / "shap_analysis" / "shap_summary_bar.png"
    )

    st.subheader("SHAP Feature Importance")

    if shap_image_path.exists():
        st.image(
            str(shap_image_path),
            use_container_width=True
        )
    else:
        st.error(f"SHAP image not found: {shap_image_path}")



# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "Research prototype only. Predictions support — not replace — "
    "professional inspection decisions."
)