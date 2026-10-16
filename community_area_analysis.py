
from pathlib import Path
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point, shape
from shapely.strtree import STRtree

BASE = Path(__file__).resolve().parent
OUTPUT_DIR = BASE / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

QUEUE_FILE = OUTPUT_DIR / "restaurant_risk_priority_queue_corrected.csv"
if not QUEUE_FILE.exists():
    QUEUE_FILE = OUTPUT_DIR / "restaurant_risk_priority_queue.csv"
if not QUEUE_FILE.exists():
    QUEUE_FILE = BASE / "restaurant_risk_priority_queue_corrected.csv"
if not QUEUE_FILE.exists():
    QUEUE_FILE = BASE / "restaurant_risk_priority_queue.csv"

SCORING_FILE = OUTPUT_DIR / "restaurant_scoring_latest.csv"
if not SCORING_FILE.exists(): SCORING_FILE = BASE / SCORING_FILE.name

BOUNDARY_URL = (
    "https://data.cityofchicago.org/api/v3/views/"
    "igwz-8jzy/query.geojson?accessType=DOWNLOAD"
)

OUTPUT_DETAIL = OUTPUT_DIR / "restaurant_community_area_risk.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "community_area_risk_summary.csv"
OUTPUT_MAP = OUTPUT_DIR / "community_area_risk_heatmap.png"


def find_column(df, candidates):
    lookup = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def main():
    if not QUEUE_FILE.exists():
        raise FileNotFoundError(
            f"Risk queue not found. Expected: {QUEUE_FILE}"
        )

    queue = pd.read_csv(QUEUE_FILE, low_memory=False)

    # Check whether the queue already includes coordinates.
    lat_col = find_column(queue, ["Latitude", "latitude", "lat"])
    lon_col = find_column(queue, ["Longitude", "longitude", "lon", "lng"])

    if lat_col is None or lon_col is None:
        if not SCORING_FILE.exists():
            raise ValueError(
                "Latitude/Longitude are missing from the queue, "
                "and restaurant_scoring_latest.csv was not found."
            )

        scoring = pd.read_csv(SCORING_FILE, low_memory=False)
        id_q = find_column(queue, ["Restaurant_ID", "restaurant_id"])
        id_s = find_column(scoring, ["Restaurant_ID", "restaurant_id"])
        lat_s = find_column(scoring, ["Latitude", "latitude", "lat"])
        lon_s = find_column(
            scoring, ["Longitude", "longitude", "lon", "lng"]
        )

        if None in (id_q, id_s, lat_s, lon_s):
            raise ValueError(
                "Could not identify restaurant ID and coordinate columns. "
                f"Queue columns: {queue.columns.tolist()}"
            )

        coordinates = scoring[[id_s, lat_s, lon_s]].drop_duplicates(
            subset=[id_s]
        )
        queue = queue.merge(
            coordinates,
            left_on=id_q,
            right_on=id_s,
            how="left",
            validate="many_to_one"
        )
        lat_col, lon_col = lat_s, lon_s

    probability_col = find_column(
        queue,
        [
            "Predicted_Failure_Probability",
            "predicted_failure_probability",
            "Failure_Probability"
        ]
    )

    if probability_col is None:
        raise ValueError(
            "Could not find predicted failure probability in the queue."
        )

    # Download official Chicago community-area boundaries.
    response = requests.get(BOUNDARY_URL, timeout=60)
    response.raise_for_status()
    geojson = response.json()

    if "features" not in geojson or not geojson["features"]:
        raise ValueError("The boundary download returned no features.")

    areas = []
    for feature in geojson["features"]:
        props = feature.get("properties", {})
        name_key = next(
            (
                key for key in props
                if key.lower() in {"community", "community_name", "name"}
            ),
            None
        )
        number_key = next(
            (
                key for key in props
                if key.lower() in {"area_numbe", "area_num_1", "area_number"}
            ),
            None
        )

        if not name_key or not feature.get("geometry"):
            continue

        polygon = shape(feature["geometry"])
        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        areas.append({
            "name": str(props[name_key]).strip().title(),
            "number": props.get(number_key) if number_key else None,
            "geometry": polygon
        })

    if not areas:
        raise ValueError(
            "Could not identify community names in boundary properties."
        )

    print(f"Loaded {len(areas)} community-area boundaries.")

        
    from shapely.geometry import Point

    print("\nBoundary diagnostics:")
    print("Boundary longitude bounds:",
        min(g.bounds[0] for g in geometries) if "geometries" in globals() else
        min(a["geometry"].bounds[0] for a in areas),
        max(a["geometry"].bounds[2] for a in areas) if "geometries" in globals() else
        max(a["geometry"].bounds[2] for a in areas))
    print("Boundary latitude bounds:",
        min(a["geometry"].bounds[1] for a in areas),
        max(a["geometry"].bounds[3] for a in areas))

    test_points = [
        (-87.609518, 41.892249),
        (-87.604045, 41.892338),
    ]

    for lon, lat in test_points:
        point = Point(lon, lat)
        matches = [
            a["name"] for a in areas
            if a["geometry"].covers(point)
        ]
        print(f"Point ({lat}, {lon}) -> {matches or 'NO MATCH'}")


    geometries = [area["geometry"] for area in areas]
    tree = STRtree(geometries)

    # STRtree query returns geometry indices in current Shapely versions.
    # The helper also supports versions returning geometry objects.
    def locate_area(lon, lat):
        if pd.isna(lon) or pd.isna(lat):
            return None, None

        lon, lat = float(lon), float(lat)
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            return None, None

        point = Point(lon, lat)

        for candidate in tree.query(point):
            if isinstance(candidate, int) or hasattr(candidate, "__index__"):
                idx = int(candidate)
                polygon = geometries[idx]
                area = areas[idx]
            else:
                polygon = candidate
                idx = next(
                    (i for i, geom in enumerate(geometries)
                     if geom.equals(polygon)),
                    None
                )
                if idx is None:
                    continue
                area = areas[idx]

            if polygon.covers(point):
                return area["name"], area["number"]

        return None, None

    # Chicago coordinates should be near longitude -87 and latitude 42.
    valid_coords = (
        pd.to_numeric(queue[lat_col], errors="coerce").between(41, 43)
        & pd.to_numeric(queue[lon_col], errors="coerce").between(-89, -87)
    )

    queue["Community_Area"] = None
    queue["Community_Area_Number"] = None

    for idx in queue.index[valid_coords]:
        name, number = locate_area(
            queue.at[idx, lon_col],
            queue.at[idx, lat_col]
        )
        queue.at[idx, "Community_Area"] = name
        queue.at[idx, "Community_Area_Number"] = number

    queue[probability_col] = pd.to_numeric(
        queue[probability_col], errors="coerce"
    )

    
    queue["Community_Area_Mapping_Status"] = "Mapped"

    queue.loc[
        queue["Community_Area"].isna(),
        "Community_Area_Mapping_Status"
    ] = "Outside or unmatched to community-area boundaries"

    print("\nCommunity-area mapping status:")
    print(queue["Community_Area_Mapping_Status"].value_counts())



    queue["High_Risk_Flag"] = queue[probability_col] >= 0.55

    queue.to_csv(OUTPUT_DETAIL, index=False)

    mapped = queue.dropna(subset=["Community_Area"]).copy()
    if mapped.empty:
        raise ValueError(
            "No restaurants were mapped. Check coordinate columns and "
            "confirm that longitude is negative for Chicago."
        )

    summary = (
        mapped.groupby(
            ["Community_Area", "Community_Area_Number"],
            dropna=False
        )
        .agg(
            Restaurant_Count=(probability_col, "size"),
            Mean_Predicted_Failure_Probability=(probability_col, "mean"),
            Median_Predicted_Failure_Probability=(probability_col, "median"),
            High_Risk_Restaurant_Count=("High_Risk_Flag", "sum")
        )
        .reset_index()
    )

    summary["Mean_Predicted_Failure_Probability"] *= 100
    summary["Median_Predicted_Failure_Probability"] *= 100

    summary = summary.sort_values(
        "Mean_Predicted_Failure_Probability",
        ascending=False
    )
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    plot_data = summary.sort_values(
        "Mean_Predicted_Failure_Probability"
    )

    plt.figure(figsize=(11, max(8, len(plot_data) * 0.22)))
    plt.barh(
        plot_data["Community_Area"],
        plot_data["Mean_Predicted_Failure_Probability"]
    )
    plt.xlabel("Mean predicted failure probability (%)")
    plt.ylabel("Chicago community area")
    plt.title("Restaurant Inspection Failure Risk by Community Area")
    plt.tight_layout()
    plt.savefig(OUTPUT_MAP, dpi=180, bbox_inches="tight")
    plt.close()

    print(f"Queue rows: {len(queue):,}")
    print(f"Mapped rows: {len(mapped):,}")
    print(f"Unmapped rows: {len(queue) - len(mapped):,}")
    print(f"Community areas represented: {summary.shape[0]}")
    print(f"Detail CSV: {OUTPUT_DETAIL}")
    print(f"Summary CSV: {OUTPUT_SUMMARY}")
    print(f"Heatmap: {OUTPUT_MAP}")
    print("\nHighest mean predicted risk:")
    print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
