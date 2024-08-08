import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import geopandas as gpd
import pandas as pd
import glacier_lengths
import shapely

S1_DIR = Path("/media/storage/Erik/Projects/UiO/S1_animation/")

def measure_lengths(positions: gpd.GeoDataFrame, centerline: shapely.geometry.LineString, domain: shapely.geometry.Polygon, radius: float = 200.):
    # centerline = shapely.from_wkt(centerline.wkt)
    # glacier_dir = S1_DIR / f"GIS/shapes/{glacier}"

    # cache_filepath = S1_dir / f"output/{glacier}/{glacier}_lengths.csv"


    # positions = gpd.read_file(filepath)

    # centerline_shp = gpd.read_file(glacier_dir / "centerline.geojson")
    # centerline = centerline_shp.iloc[0].geometry

    # domain_shp = gpd.read_file(glacier_dir / "domain.geojson")
    # domain = domain_shp.iloc[0].geometry

    buffered_centerlines = glacier_lengths.buffer_centerline(centerline, domain, max_radius=radius)

    lengths = []
    for (i, position) in positions.iterrows():

        for j, line in enumerate(buffered_centerlines.geoms):
            # cut = glacier_lengths.cut_centerlines(line, position.geometry, max_difference_fraction=0.005)

            cut_geoms = shapely.ops.split(line, position.geometry)

            for cut in cut_geoms.geoms:
                if line.coords[-1][0] == cut.coords[-1][0] and line.coords[-1][1] == cut.coords[-1][1]:
                    break

            cut_lengths = glacier_lengths.measure_lengths(cut)

            lengths.append({
                "line_i": j,
                "date": position.date,
                "length": cut_lengths[0],
                # "median": np.median(cut_lengths),
                # "std": np.std(cut_lengths),
                # "upper": np.percentile(cut_lengths, 75),
                # "lower": np.percentile(cut_lengths, 25),
            })

    lengths = pd.DataFrame.from_records(lengths)
    lengths["date"] = pd.to_datetime(lengths["date"])

    lengths_agg = []
    for date, values in lengths.groupby("date"):

        lengths_agg.append({
            "date": date,
            "median": np.median(values["length"]),
            "mean": np.mean(values["length"]),
            "std": np.std(values["length"]),
            "upper": np.percentile(values["length"], 75),
            "lower": np.percentile(values["length"], 25),
            "count": values.shape[0],
            

        })
    lengths_agg = pd.DataFrame.from_records(lengths_agg).sort_values("date")
    return lengths, lengths_agg

def get_front_positions(glacier: str, gis_key: str | None = None):

    if gis_key is None:
        gis_key = glacier

    gis_dir = S1_DIR / f"GIS/shapes/{gis_key}/"
    front_positions = gpd.read_file(gis_dir / "front_positions.geojson")

    try:
        centerline = gpd.read_file("GIS/shapes/centerlines.geojson").query(f"glacier == '{glacier}'").geometry.iloc[0]
    except IndexError as exception:
        if "single positional indexer" not in str(exception):
            raise exception
        raise ValueError(f"No key {glacier} in centerlines") from exception
    try:
        domain = gpd.read_file("GIS/shapes/domains.geojson").query(f"glacier == '{glacier}'").geometry.iloc[0]
    except IndexError as exception:
        if "single positional indexer" not in str(exception):
            raise exception
        raise ValueError(f"No key {glacier} in domain") from exception

    try:
        low_coh_boundary = gpd.read_file("GIS/shapes/low_coh_boundaries.geojson").query(f"glacier == '{glacier}'")
    except IndexError as exception:
        if "single positional indexer" not in str(exception):
            raise exception
        raise ValueError(f"No key {glacier} in low_coh_boundary") from exception
    missing_date = low_coh_boundary["date"].isna()

    low_coh_boundary.loc[missing_date, "date"] = pd.to_datetime(low_coh_boundary["year"].astype(str) + "-04-01")
    # low_coh_boundary["date"] = pd.to_datetime(low_coh_boundary["year"].astype(str) + "-04-01")
    
    return front_positions, centerline, domain, low_coh_boundary


def measure_velocity(lengths):
    dates = np.sort(lengths["date"].unique())

    diff_intervals = pd.IntervalIndex.from_arrays(left=dates[:-1], right=dates[1:])

    diffs = []
    for i, interval in enumerate(diff_intervals):

        before_vals = lengths[lengths["date"] == interval.left].set_index("line_i")
        after_vals = lengths[lengths["date"] == interval.right].set_index("line_i")
         
        diff = (after_vals - before_vals).dropna()

        if diff.shape[0] == 0:
            continue

        diffs.append({
            "date": interval,
            "mean": np.mean(diff["length"]),
            "median": np.median(diff["length"]),
            "lower": np.percentile(diff["length"], 25),
            "upper": np.percentile(diff["length"], 75),
            "std": np.std(diff["length"]),
            "count": diff.shape[0],
        })


    diffs = pd.DataFrame.from_records(diffs).set_index("date").sort_index()

    diff_per_day = diffs / np.repeat(((diffs.index.right - diffs.index.left).total_seconds() / (3600 * 24)).values[:, None], diffs.shape[1], 1)
    diff_per_day["count"] = diffs["count"]

    diff_per_day["date_from"] = diff_per_day.index.left
    diff_per_day["date_to"] = diff_per_day.index.right

    # diff_per_day = diff_per_day.set_index(diff_per_day.index.mid).resample("1M").mean().dropna()

    return diff_per_day

def plot_length_evolution(glacier: str = "edvard"):


    gis_key = {
        "vallakra": "vallakrabreen",
        "eton": "etonfront",
    }

    front_positions, centerline, domain, coh_boundary = get_front_positions(glacier=glacier, gis_key=gis_key.get(glacier, glacier))

    lower_coh_boundary = coh_boundary[coh_boundary["boundary_type"] == "lower"]
    upper_coh_boundary = coh_boundary[coh_boundary["boundary_type"] == "upper"]
    
    front_lengths_raw, front_lengths = measure_lengths(front_positions, centerline, domain)

    velocities = {
        "front": measure_velocity(front_lengths_raw)
    }

    try:
        lower_coh_lengths_raw, lower_coh_lengths = measure_lengths(lower_coh_boundary, centerline, domain)

        advancing_dates = velocities["front"][velocities["front"]["median"] > 0.2]["date_to"].astype("str").values
        advancing_front_positions = front_lengths[front_lengths["date"].isin(advancing_dates)]
        advancing_front_positions_raw = front_lengths_raw[front_lengths_raw["date"].isin(advancing_dates)]

        lower_coh_lengths = pd.concat([lower_coh_lengths, advancing_front_positions]).sort_values("date")
        lower_coh_lengths_raw = pd.concat([lower_coh_lengths_raw, advancing_front_positions_raw]).sort_values("date")


        # print(front_lengths[front_lengths["median"].diff() > 0])
        # lower_coh_lengths = lower_coh_lengths[lower_coh_lengths["median"] < front_lengths["median"].max()]

        # lower_coh_lengths = pd.concat([lower_coh_lengths, front_lengths[front_lengths["date"] > lower_coh_lengths["date"].max()]])

        velocities["lower_coh"] = measure_velocity(lower_coh_lengths_raw)
    except KeyError:
        # lower_coh_lengths = pd.DataFrame(columns=["median", "lower", "upper", "date"])

        lower_coh_lengths_raw = lower_coh_lengths = None

    try:
        upper_coh_lengths_raw, upper_coh_lengths = measure_lengths(upper_coh_boundary, centerline, domain)
        velocities["upper_coh"] = measure_velocity(upper_coh_lengths_raw)
        # print(upper_coh_lengths_raw)
    except KeyError:
        upper_coh_lengths_raw = upper_coh_lengths = None


    # velocities = {key: measure_velocity(data) for key, data in [("terminus", front_lengths_raw), "lower_coh", "upper_coh")]

    Path("figures/").mkdir(exist_ok=True)

    fig = plt.figure(figsize=(8, 5))
    for data, params in [(front_lengths, {"color": "blue", "label": "Terminus", "zorder": 2, "key": "front"}), (lower_coh_lengths, {"color": "orange", "label": "Lower instability", "zorder": 1, "key": "lower_coh"}), (upper_coh_lengths, {"color": "green", "label": "Upper instability", "zorder": 1, "key": "upper_coh"})]:
        if data is None:
            continue

        plt.subplot(121)
        plt.fill_between(data["date"], data["lower"] / 1e3, data["upper"] / 1e3, alpha=0.5, color=params["color"], zorder=params["zorder"])
        plt.plot(data["date"], data["median"] / 1e3, color=params["color"], label=params["label"], zorder=params["zorder"])
        plt.scatter(data["date"], data["median"] / 1e3, color=params["color"], zorder=params["zorder"])
        plt.ylabel("Glacier length (km)")
        plt.xlabel("Year")
        plt.legend()

        plt.subplot(122)
        plt.fill_between(np.ravel(np.column_stack((velocities[params["key"]].index.left, velocities[params["key"]].index.right))), np.repeat(velocities[params["key"]]["lower"], 2), np.repeat(velocities[params["key"]]["upper"], 2), alpha=0.3, color=params["color"], zorder=params["zorder"])
        plt.plot(np.ravel(np.column_stack((velocities[params["key"]].index.left, velocities[params["key"]].index.right))), np.repeat(velocities[params["key"]]["median"], 2), color=params["color"], zorder=params["zorder"])
        plt.ylabel("Advance/retreat rate (m/d)")
        plt.xlabel("Year")
        plt.gca().yaxis.tick_right()
        plt.gca().yaxis.set_label_position("right")

    plt.tight_layout()
    plt.savefig(f"figures/{glacier}_front_change.jpg", dpi=300)
    plt.show()

def main():

    plot_length_evolution()


if __name__ == "__main__":
    main()
    
