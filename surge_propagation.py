import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import geopandas as gpd
import pandas as pd
import glacier_lengths
import shapely
import fnmatch
import rasterio
import rasterio.features
import itertools

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
            else:
                raise NotImplementedError("Line was not cut properly")

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

    low_coh_boundary["date"] = pd.to_datetime(low_coh_boundary["date"])
   
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

def get_length_evolution(glacier: str):
    gis_key = {
        "vallakra": "vallakrabreen",
        "eton": "etonfront",
    }

    # Load the digitized data
    front_positions, centerline, domain, coh_boundary = get_front_positions(glacier=glacier, gis_key=gis_key.get(glacier, glacier))

    # Split the lower and upper coherence boundaries
    lower_coh_boundary = coh_boundary[coh_boundary["boundary_type"] == "lower"]
    upper_coh_boundary = coh_boundary[coh_boundary["boundary_type"] == "upper"]

    # Calculate the glacier front lengths
    front_lengths_raw, front_lengths = measure_lengths(front_positions, centerline, domain)
    front_lengths = front_lengths.set_index("date")

    # The final output index will be conformed to this date list
    all_dates = np.unique(np.r_[coh_boundary["date"], front_lengths.index])
    
    def to_multiindex(df, name: str, mi_name: str = "kind"):
        """From https://stackoverflow.com/a/42094658."""
        return pd.concat({name: df}, names=[mi_name]) 

    # Interpolate and back-/front-fill the front positions to the whole time range, and create the output dataframe.
    data = to_multiindex(front_lengths.reindex(all_dates).interpolate("linear").bfill().ffill(), "front")

    # The "exact" label tells if the data are measured or interpolated
    data["exact"] = False
    data.loc[("front", front_lengths.index), "exact"] = True

    # If there are data on the lower coherence boundary, add them.
    if lower_coh_boundary.shape[0] > 0:
        lower_coh_lengths_raw, lower_coh_lengths = measure_lengths(lower_coh_boundary, centerline, domain)
        lower_coh_lengths["exact"] = True
        
        data = pd.concat([data, to_multiindex(lower_coh_lengths.set_index("date"), "lower_coh")], join="outer")

    # If there are data on the upper coherence boundary, add them.
    if upper_coh_boundary.shape[0] > 0:
        upper_coh_lengths_raw, upper_coh_lengths = measure_lengths(upper_coh_boundary, centerline, domain)
        upper_coh_lengths["exact"] = True
        data = pd.concat([data, to_multiindex(upper_coh_lengths.set_index("date"), "upper_coh")], join="outer")

    # Assert whether it's a top-down surge or a bottom up. Start with the top-down assumption.
    top_down = True
    if "upper_coh" in data.index.get_level_values(0):
        # If the dataset has an upper_coh column (almost exclusive to bottom-up) and ...
        # the upper_coh column is generally going up-glacier, it's assumed to be a bottom-up surge
        if data.loc[("upper_coh", slice(None)), "median"].diff().mean() < 0:
            top_down = False

    if top_down:
        # If there's no upper coherence boundary, assume that it starts at zero
        if "upper_coh" not in data.index.get_level_values(0):
            new_data = pd.Series({key: 0. for key in front_lengths.columns} | {"exact": False}).to_frame(all_dates[0]).T.reindex(all_dates).ffill()

            data = pd.concat([data, to_multiindex(new_data, "upper_coh")])
        # This has never been encountered so far, so it's not implemented.
        else:
            raise NotImplementedError("Upper coh exists for a top-down surge")

    # If a bottom-up surge:
    else:
        # If there's no lower coherence boundary in the data, assume that it's exactly at the terminus
        if "lower_coh" not in data.index.get_level_values(0):
            front_data = data.loc["front"].copy()
            front_data["exact"] = False
            data = pd.concat([data, to_multiindex(front_data, "lower_coh")])
        # If there is a boundary, assume that it starts at the terminus but then moves up-glacier;
        # It will be equal to the terminus before the measurements and interpolated/ffilled after.
        else:

            idx = data.loc["lower_coh"].index
            new_idx = all_dates[all_dates >= idx.min()]

            lower_coh = data.loc["lower_coh"].drop(columns="exact").reindex(new_idx).interpolate("slinear").bfill().ffill()
            lower_coh["exact"] = lower_coh.index.isin(idx)
            front_data = data.loc["front"].copy()
            front_data["exact"] = False
            front_data.loc[lower_coh.index] = lower_coh
            
            data = pd.concat([data.drop(index="lower_coh"), to_multiindex(front_data, "lower_coh")])
            
    data = data.sort_index()

    # Interpolate/bfill/ffill the main measurement for the type of surge.
    col = "lower_coh" if top_down else "upper_coh"
    coh = data.loc[col]
    orig_idx = coh.index.copy()
    coh = coh.reindex(all_dates).drop(columns=["exact"]).interpolate("slinear").bfill().ffill()
    coh["exact"] = coh.index.isin(orig_idx)
    data = pd.concat([data.drop(index=col), to_multiindex(coh, col)])

    # Identify areas where the boundaries are outside of the terminus, and correct it.
    num_cols = ["upper", "lower", "mean", "median"]
    for idx in ["lower_coh", "upper_coh"]:
        mask = (data.loc["front", num_cols] < data.loc[idx, num_cols])
        replacement = data.loc[["front"]]
        replacement.index = data.loc[[idx]].index
        data[to_multiindex(mask, idx)] = replacement

    expected_length = all_dates.shape[0] * 3
    if expected_length != data.shape[0]:
        raise ValueError(f"Expected length of the dataset ({expected_length}) is different from its shape ({data.shape})")


    # Calculate front propagation velocities in m/d
    dt_days = pd.Series(data.loc["front"].index, data.loc["front"].index).diff().dt.total_seconds() / (3600 * 24)
    for kind, kind_data in data.groupby(level=0):
        shifted = kind_data.shift(1)
        dt_days_multi = to_multiindex(dt_days, str(kind))

        data.loc[kind, "vel"] = (kind_data["median"] - shifted["median"]) / dt_days_multi

        vel_upper = kind_data["upper"] - shifted["upper"]
        vel_lower = kind_data["lower"] - shifted["lower"]

        data.loc[kind, "vel_upper"] = np.where(vel_upper > vel_lower, vel_upper, vel_lower) / dt_days_multi
        data.loc[kind, "vel_lower"] = np.where(vel_upper > vel_lower, vel_lower, vel_upper) / dt_days_multi

    # Convert units from m to km
    data[num_cols] /= 1000

    data["surge_kind"] = "top-down" if top_down else "bottom-up"

    return data
    


def plot_length_evolution(glacier: str = "arnesen", show: bool = False):

    names = {
        "vallakra": "Vallåkrabreen",
        "natascha": "Paulabreen",
    }

    data = get_length_evolution(glacier=glacier)
    
    plt.fill_between(np.unique(data.index.get_level_values(1)), data.loc["front", "median"], color="#" + "c" * 4 + "ff")
    plt.fill_between(np.unique(data.index.get_level_values(1)), data.loc["upper_coh", "median"], data.loc["lower_coh", "median"], color="gray")
    plt.plot(data.loc["front", "median"], color="blue")

    all_params = {
        "front": {
            "color": "blue",
            "zorder": 3,
        },
        "lower_coh": {
            "color": "red",
            "zorder": 2,
        },
        "upper_coh": {
            "color": "green",
            "zorder": 1,
        },
    }
    # zorders = {"front": 3, "lower_coh": 2, "upper_coh": 1}
    # colors = {"front": "blue", "lower_coh": "red", "upper_coh": "green"}
    for kind, kind_data in data.groupby(level=0):
        params = all_params[str(kind)]

        kind_data = kind_data.loc[kind]
        exact_data = kind_data[kind_data["exact"]]
        plt.plot(kind_data.index, kind_data["median"], zorder=params["zorder"], color=params["color"], linestyle=":") 
        plt.scatter(exact_data.index, exact_data["median"], zorder=params["zorder"], color=params["color"], s=6)
        plt.plot(np.repeat(exact_data.index, 3), np.array((exact_data["lower"].values, exact_data["upper"].values, np.zeros(exact_data.shape[0]) + np.nan)).T.ravel(), zorder=params["zorder"], color=params["color"])

    ymax = data[data["exact"]]["median"].max()
    ymin = data[data["exact"]]["median"].min()
    yrange = ymax - ymin

    import matplotlib.dates as mdates
    from matplotlib.ticker import StrMethodFormatter
    plt.ylim(max(ymin - yrange * 0.1, 0), ymax + yrange * 0.2)
    # yrange = plt.gca().get_ylim()[1] - data[data["exact"]]["median"].min()
    # plt.ylim(max(plt.gca().get_ylim()[1] - (yrange * 1.1), 0), plt.gca().get_ylim()[1])

    plt.xlim(np.min(data.index.get_level_values(1)), np.max(data.index.get_level_values(1)))

    plt.text(0.5, 0.98, names.get(glacier, glacier.capitalize() + "breen"), transform=plt.gca().transAxes, va="top", ha="center")

    xticks = plt.gca().get_xticks()

    plt.xticks([int(xticks[1]), xticks[int(len(xticks) / 2)], xticks[-1]])
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

    if show:
        plt.tight_layout()
        plt.show()

def old_plot_length_evolution(glacier: str = "arnesen"):


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
    # plt.subplot(1, 2, 1)
    # all_idx = np.unique(np.r_[upper_coh_lengths["date"], lower_coh_lengths["date"]])
    # all_coh = pd.DataFrame(index=all_idx)
    # for name, df in [("lower", lower_coh_lengths), ("upper", upper_coh_lengths)]:
    #     renamed = df.set_index("date").add_prefix(name + "_")
    #     all_coh[renamed.columns] = renamed.reindex(all_idx).interpolate("linear").bfill().ffill()
    # plt.fill_between(all_coh.index, all_coh["upper_median"] / 1e3, all_coh["lower_median"] / 1e3, color="gray")
    for data, params in [(front_lengths, {"color": "blue", "label": "Terminus", "zorder": 2, "key": "front"}), (lower_coh_lengths, {"color": "orange", "label": "Lower low-coh. bnd.", "zorder": 1, "key": "lower_coh"}), (upper_coh_lengths, {"color": "green", "label": "Upper low-coh. bnd.", "zorder": 1, "key": "upper_coh"})]:
        if data is None:
            continue

        plt.subplot(121)
        plt.fill_between(data["date"], data["lower"] / 1e3, data["upper"] / 1e3, alpha=0.5, color=params["color"], zorder=params["zorder"])
        plt.plot(data["date"], data["median"] / 1e3, color=params["color"], label=params["label"], zorder=params["zorder"])
        plt.scatter(data["date"], data["median"] / 1e3, color=params["color"], zorder=params["zorder"])
        plt.ylabel("Glacier length (km)")
        plt.xlabel("Year")

        plt.subplot(122)
        plt.fill_between(np.ravel(np.column_stack((velocities[params["key"]].index.left, velocities[params["key"]].index.right))), np.repeat(velocities[params["key"]]["lower"], 2), np.repeat(velocities[params["key"]]["upper"], 2), alpha=0.3, color=params["color"], zorder=params["zorder"])
        plt.plot(np.ravel(np.column_stack((velocities[params["key"]].index.left, velocities[params["key"]].index.right))), np.repeat(velocities[params["key"]]["median"], 2), color=params["color"], zorder=params["zorder"],label=params["label"] )
        plt.ylabel("Advance/retreat rate (m/d)")
        plt.xlabel("Year")
        plt.gca().yaxis.tick_right()
        plt.gca().yaxis.set_label_position("right")

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{glacier}_front_change.jpg", dpi=300)
    plt.show()

def main():

    fig = plt.figure(figsize=(8, 5), dpi=200)

    glaciers = [
        ["arnesen", "kval", "bore", "eton"],
        ["natascha", "scheele", "vallakra", "penck"],
    ]

    n_rows = len(glaciers)
    n_cols = max((len(col) for col in glaciers))

    for row_n, row in enumerate(glaciers):
        for col_n, glacier in enumerate(row):
            plt.subplot(n_rows, n_cols, col_n + 1 + n_cols * row_n)
            plot_length_evolution(glacier)

            
    # for i, glacier in enumerate(glaciers):
    #     plt.subplot(1, 4, i +1)
    plt.tight_layout(w_pad=-0.5, rect=(0.02, 0., 1., 1.))
    plt.text(0.01, 0.5, "Distance (km)", rotation=90, ha="center", va="center", transform=fig.transFigure)

    plt.savefig("figures/front_change.jpg", dpi=300)
    plt.show()


def load_coh(glacier: str):
    gis_key = {
        "vallakra": "vallakrabreen",
        "eton": "etonfront",
    }

    default_files_hh = {
        2016: [
            "*S1AA_20160329*HH*",
        ],
        2018: [
            "*S1AA_20180319*HH*",
        ],
        2019: [
            "*S1AA_20190302*HH*",
        ],
        2020: [
            "*S1AA_20200401*HH*",
        ],
        2021: [
            "*S1AA_20210207*HH*",
        ],
        2022: [
            "*S1AA_20220403*HH*",
        ],
        2023: [
            "*S1AA_20230422*HH*",
        ],
        2024: [
            "*S1AA_20240403*HH*",
            "*S1AA_20240404*HH*",
        ],
    }

    front_positions, centerline, domain, coh_boundary = get_front_positions(glacier=glacier, gis_key=gis_key.get(glacier, glacier))

    bounds = rasterio.coords.BoundingBox(*domain.buffer(200).bounds)

    res = (40., 40.)
    transform = rasterio.transform.from_origin(bounds.left, bounds.top, *res)
    out_shape = int((bounds.top - bounds.bottom) / res[1]), int(np.ceil((bounds.right - bounds.left) /res[0])) 

    # plt.imshow(centerline_rst)
    # plt.show()

    # rasterio.features.rasterize((domain
    meta = {}
    data = {"coh": {}}

    for year in default_files_hh:
        for filepath in itertools.chain(*(map(Path, fnmatch.filter(map(str, Path("insar/").glob("*.zip")), pattern)) for pattern in default_files_hh[year])):
        # for pattern in default_files_hh[year]:
        #     for filepath in map(Path, fnmatch.filter(map(str, Path("insar/").glob("*.zip")), pattern)):

            filename = f"/vsizip/{filepath}/{filepath.stem}/{filepath.stem}_corr.tif"

            with rasterio.open(filename) as raster:
                if (raster.bounds.left < domain.centroid.x < raster.bounds.right) and (raster.bounds.bottom < domain.centroid.y < raster.bounds.top):
                    test_val = raster.sample([[domain.centroid.x, domain.centroid.y]], masked=True).__next__().filled(np.nan)[0]
                    if not np.isfinite(test_val):
                        continue

                    window = rasterio.windows.from_bounds(*bounds, transform=raster.transform)
                    data["coh"][year] = raster.read(1, window=window, boundless=True)
                    meta["shape"] = data["coh"][year].shape


    max_centerline_length = {"natascha": 18300}.get(glacier, centerline.length)

    width = res[0] * 3
    centerline_pts = []
    for dist in np.arange(width, max_centerline_length, step=width):
        centerline_pts.append((centerline.interpolate(dist).buffer(width), int(dist)))

    data["centerline"] = rasterio.features.rasterize(centerline_pts, out_shape=meta["shape"], transform=transform)

    
    bins = np.r_[[-100], np.linspace(width, max_centerline_length, num=20), [centerline.length * 2]]
    centerline_binned = np.digitize(data["centerline"], bins=bins)

    plt.figure(figsize=(8, 5))
    for i, year in enumerate(sorted(data["coh"])):
        plt.subplot(1, len(data["coh"]), i + 1)

        xvals = []
        yvals = []
        for value in np.unique(centerline_binned):
            if value < 2:
                continue
            mask = centerline_binned == value

            xvals.append(bins[value] / 1000)
            yvals.append(np.mean(data["coh"][year][mask]))

        plt.title(year)
        plt.plot(yvals, xvals, color="black")
        ylim = plt.gca().get_ylim()
        plt.ylim(ylim[::-1])

        for j in range(1, len(yvals)):
            if yvals[j] > 0.54:
                plt.scatter(yvals[j - 1], xvals[j -1], marker="x", s=60, color="red")
                break

        yticks = plt.gca().get_yticks()
        if i != 0:
            plt.yticks(yticks, [""] * len(yticks))
        else:
            plt.yticks(yticks)
            plt.ylabel("Distance (km)")
            plt.xlabel("Coherence")

        plt.xlim(0, 1)
        plt.xticks([0., 0.5, 1.], None if i == 0 else [""] * 3)

        plt.grid(alpha=0.5)
        # plt.xticks(plt.gca().get_xticks()[[0, -1]])

            
        # plt.title(year)
        # plt.imshow(data[year], cmap="Greys_r")

    # plt.legend()

    if glacier == "natascha":
        plt.subplots_adjust(left=0.08, bottom=0.05, right=0.99, top=0.95, wspace=0.01)
    else:
        plt.tight_layout()
    plt.savefig(f"figures/{glacier}_coh_along_center.jpg", dpi=300)
    plt.show()
    


if __name__ == "__main__":
    main()
    
