import collections.abc
import fnmatch
import hashlib
import itertools
import json
from pathlib import Path

import geopandas as gpd
import glacier_lengths
import matplotlib.dates
import matplotlib.patheffects
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import rasterio.transform
import rasterio.warp
import shapely
import shapely.geometry
import shapely.ops
import textalloc

from query_insar import CACHE_DIR

GIS_KEYS = {
    "vallakra": "vallakrabreen",
    "eton": "etonfront",
}

TEX_VELOCITY_UNIT = "m~d$^{-1}$"

ETONBREEN_SURGE_START = pd.Timestamp(
    "2023-11-11"
)  # See the paper for why we hardcode it


def order_surges(glaciers: list[str], all_default: bool = False) -> list[str]:
    """Sort the glacier key list in a predefined order.

    If they are not in the list, they get appended at the end, sorted alphabetically.

    Examples
    --------
    >>> order_surges(["b", "a"])
    ['a', 'b']

    >>> order_surges(["kval", "osborne", "b", "a"])
    ['osborne', 'kval', 'a', 'b']

    """
    glaciers = glaciers.copy()
    preordered = [
        "arnesen",
        "osborne",
        "kval",
        "stone",
        "eton",
        "bore",
        "morsnev",
        "penck",
        "scheele",
        "vallakra",
        "delta",
        "natascha",
        "nordsyssel",
        "sefstrom",
        "doktor",
        "edvard",
    ]

    ordered = []
    if all_default:
        for glacier in preordered:
            if glacier in glaciers:
                glaciers.remove(glacier)

    for glacier in preordered:
        if glacier in glaciers:
            glaciers.remove(glacier)
        elif all_default:
            pass
        else:
            continue
        ordered.append(glacier)

    glaciers.sort()
    return ordered + glaciers


def measure_lengths(
    positions: gpd.GeoDataFrame,
    centerline: shapely.geometry.LineString,
    domain: shapely.geometry.Polygon,
    radius: float = 200.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure lengths of the provided (front) position linestrings.

    The centerline is buffered with the given radius and cut to the domain.
    Then, the buffered centerlines are used for measuring the front position lengths.

    Parameters
    ----------
    positions
        (Front) position linestrings to measure lengths of. It is assumed to contain two columns: ["date", "goemetry"]
    centerline
        The centerline to measure lengths along. NOTE: Direction is important; the start vertex indicates the start.
    domain
        The domain to use for cutting the buffered centerlines properly.
    radius
        The radius (distance to the original centerline) at which to measure.

    Returns
    -------
    The raw lengths measurements (described below) and aggregate length statistics for each position.

    They come in the order "(raw, aggregates)".

    Raw length measurements are for each individual buffered centerline. These are optimal for measuring velocities.

    """
    buffered_centerlines = glacier_lengths.buffer_centerline(
        centerline, domain, max_radius=radius
    )

    # Initialize the raw length measurement data
    lengths = []
    for _, position in positions.iterrows():
        for i, line in enumerate(buffered_centerlines.geoms):
            # cut = glacier_lengths.cut_centerlines(line, position.geometry, max_difference_fraction=0.005)

            # Split the front position using the centerline. Assuming they overlap, it will generate AT LEAST two lines.
            cut_geoms = shapely.ops.split(line, position.geometry)

            # Loop through all cut lines and find the one that is connected to the start vertex (the beginning).
            for cut in cut_geoms.geoms:
                if (
                    line.coords[-1][0] == cut.coords[-1][0]
                    and line.coords[-1][1] == cut.coords[-1][1]
                ):
                    break
            else:
                # This might get triggered if the centerline and front position don't overlap..
                raise NotImplementedError("Line was not cut properly")

            cut_length = glacier_lengths.measure_lengths(cut)[0]
            lengths.append(
                {
                    "line_i": i,
                    "date": position.date,
                    "length": cut_length,
                }
            )

    lengths = pd.DataFrame.from_records(lengths)
    lengths["date"] = pd.to_datetime(lengths["date"])

    # Aggregate the length data for each original line
    lengths_agg = []
    for date, values in lengths.groupby("date"):
        lengths_agg.append(
            {
                "date": date,
                "median": np.median(values["length"]),
                "mean": np.mean(values["length"]),
                "std": np.std(values["length"]),
                "upper": np.percentile(values["length"], 75),
                "lower": np.percentile(values["length"], 25),
                "count": values.shape[0],
            }
        )
    lengths_agg = pd.DataFrame.from_records(lengths_agg).sort_values("date")
    return lengths, lengths_agg


def get_glacier_data(
    glacier: str,
) -> tuple[
    gpd.GeoDataFrame,
    shapely.geometry.LineString,
    shapely.geometry.Polygon,
    gpd.GeoDataFrame,
]:
    """Load data about a glacier from its key.

    Parameters
    ----------
    glacier
        The glacier key (e.g. "eton"). It's GIS key may be different and is defined in `GIS_KEYS`.

    Returns
    -------
    A tuple of data in the following form:
    1. Front positions. Linestrings with a "date" field to separate them.
    2. Centerline. The glacier centerline linestring.
    3. Domain. The evaluation domain to cut the buffered centerlines within.
    4. Low-coherence boundaries. The measured upper/lower boundaries of a low-coherence front.

    """
    gis_key = GIS_KEYS.get(glacier, glacier)

    front_positions = gpd.read_file(f"GIS/shapes/{gis_key}/front_positions.geojson")
    front_positions["date"] = pd.to_datetime(front_positions["date"])

    crs = front_positions.crs

    try:
        centerline = (
            gpd.read_file("GIS/shapes/centerlines.geojson")
            .query(f"glacier == '{glacier}'")
            .to_crs(crs)
            .geometry.iloc[0]
        )
    except IndexError as exception:
        if "single positional indexer" not in str(exception):
            raise exception
        raise ValueError(f"No key {glacier} in centerlines") from exception
    try:
        domain = (
            gpd.read_file("GIS/shapes/domains.geojson")
            .query(f"glacier == '{glacier}'")
            .to_crs(crs)
            .geometry.iloc[0]
        )
    except IndexError as exception:
        if "single positional indexer" not in str(exception):
            raise exception
        raise ValueError(f"No key {glacier} in domain") from exception

    try:
        low_coh_boundary = (
            gpd.read_file("GIS/shapes/low_coh_boundaries.geojson")
            .query(f"glacier == '{glacier}'")
            .to_crs(crs)
        )
    except IndexError as exception:
        if "single positional indexer" not in str(exception):
            raise exception
        raise ValueError(f"No key {glacier} in low_coh_boundary") from exception

    low_coh_boundary["date"] = pd.to_datetime(low_coh_boundary["date"])

    return front_positions, centerline, domain, low_coh_boundary


def measure_velocity(raw_lengths: pd.DataFrame) -> pd.DataFrame:
    """Measure length advance/retreat velocities.

    The lower/upper columns represent the 25th/75th percentiles.

    Parameters
    ----------
    raw_lengths
        The raw length measurements for each cut centerline. See `measure_lengths()`.

    Returns
    -------
    Aggregated velocity statistics in m/d.

    """
    # Generate the intervals at which to calculate velocities.
    dates = np.sort(raw_lengths["date"].unique())
    diff_intervals = pd.IntervalIndex.from_arrays(left=dates[:-1], right=dates[1:])

    diffs = []
    for interval in diff_intervals:
        # Extract the before and after lengths and set the index to the centerline it was measured on.
        # This means that upon subsequent subtraction, differences are only measured along the same centerlines, not...
        # ... between different centerlines. This ensures proper resultant spreads.
        before_vals = raw_lengths[raw_lengths["date"] == interval.left].set_index(
            "line_i"
        )
        after_vals = raw_lengths[raw_lengths["date"] == interval.right].set_index(
            "line_i"
        )

        diff = (after_vals - before_vals).dropna()

        if diff.shape[0] == 0:
            continue

        diffs.append(
            {
                "date": interval,
                "mean": np.mean(diff["length"]),
                "median": np.median(diff["length"]),
                "lower": np.percentile(diff["length"], 25),
                "upper": np.percentile(diff["length"], 75),
                "std": np.std(diff["length"]),
                "count": diff.shape[0],
            }
        )

    diffs = pd.DataFrame.from_records(diffs).set_index("date").sort_index()

    # Convert the differences to velocities in m/d
    diff_per_day = diffs / np.repeat(
        ((diffs.index.right - diffs.index.left).total_seconds() / (3600 * 24)).values[
            :, None
        ],
        diffs.shape[1],
        1,
    )
    diff_per_day["count"] = diffs["count"]

    diff_per_day["date_from"] = diff_per_day.index.left
    diff_per_day["date_to"] = diff_per_day.index.right

    return diff_per_day


def get_length_evolution(glacier: str, force_redo: bool = False) -> pd.DataFrame:
    """Get the formatted length (terminus and low-coherence front) evolution data.

    Parameters
    ----------
    glacier
        The identifier key of the glacier.
    force_redo
        Ignore any previous cached result and redo it.

    Returns
    -------
    A table of lengths (in km), velocities (m/d) of the different fronts along the glacier.
    The "kind" multiindex specifies the type of front:
    - "front": The terminus
    - "upper_coh": The upper boundary of a low-coherence front.
    - "lower_coh": The lower boundary of a low-coherence front.

    """
    # Load the digitized data
    front_positions, centerline, domain, coh_boundary = get_glacier_data(
        glacier=glacier
    )

    front_positions = (
        front_positions.set_index("date", drop=False)
        .resample("2M", label="right")
        .last()
        .dropna()
        .reset_index(drop=True)
    )

    # Load the results from cache if it exists and the inputs have not changed.
    checksum = hashlib.sha256("".join(map(str, [front_positions, centerline, domain, coh_boundary])).encode()).hexdigest()

    cache_filepath = (
        CACHE_DIR
        / f"get_length_evolution/get_length_evolution-{glacier}-{checksum}.csv"
    )
    if cache_filepath.is_file() and not force_redo:
        cached = pd.read_csv(
            cache_filepath, index_col=[0, 1], parse_dates=True, date_format="%Y-%m-%d"
        )
        # cached.index.names = ["kind", "date"]
        return cached

    # Split the lower and upper coherence boundaries
    lower_coh_boundary: gpd.GeoDataFrame = coh_boundary.query(
        "boundary_type == 'lower'"
    )
    upper_coh_boundary: gpd.GeoDataFrame = coh_boundary.query(
        "boundary_type == 'upper'"
    )

    lengths_raw: dict[str, pd.DataFrame] = {}

    # Calculate the glacier front lengths
    lengths_raw["front"], front_lengths = measure_lengths(
        front_positions, centerline, domain
    )

    front_lengths = front_lengths.set_index("date")

    # The final output index will be conformed to this date list
    all_dates = np.unique(np.r_[coh_boundary["date"], front_lengths.index])

    def to_multiindex(
        df: pd.DataFrame, name: str, mi_name: str = "kind"
    ) -> pd.DataFrame:
        """From https://stackoverflow.com/a/42094658."""
        return pd.concat({name: df}, names=[mi_name])

    # Interpolate and back-/front-fill the front positions to the whole time range, and create the output dataframe.
    data = to_multiindex(
        front_lengths.reindex(all_dates).interpolate("linear").bfill().ffill(), "front"
    )

    # The "exact" label tells if the data are measured or interpolated
    data["exact"] = False
    data.loc[("front", front_lengths.index), "exact"] = True

    # If there are data on the lower coherence boundary, add them.
    if lower_coh_boundary.shape[0] > 0:
        lengths_raw["lower_coh"], lower_coh_lengths = measure_lengths(
            lower_coh_boundary, centerline, domain
        )
        lower_coh_lengths["exact"] = True

        data = pd.concat(
            [data, to_multiindex(lower_coh_lengths.set_index("date"), "lower_coh")],
            join="outer",
        )

    # If there are data on the upper coherence boundary, add them.
    if upper_coh_boundary.shape[0] > 0:
        lengths_raw["upper_coh"], upper_coh_lengths = measure_lengths(
            upper_coh_boundary, centerline, domain
        )
        upper_coh_lengths["exact"] = True
        data = pd.concat(
            [data, to_multiindex(upper_coh_lengths.set_index("date"), "upper_coh")],
            join="outer",
        )

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
            new_data = (
                pd.Series(
                    {key: 0.0 for key in front_lengths.columns} | {"exact": False}
                )
                .to_frame(all_dates[0])
                .T.reindex(all_dates)
                .ffill()
            )

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
            lengths_raw["lower_coh"] = lengths_raw["front"]
        # If there is a boundary, assume that it starts at the terminus but then moves up-glacier;
        # It will be equal to the terminus before the measurements and interpolated/ffilled after.
        else:
            idx = data.loc["lower_coh"].index
            # The index to interpolate is everything after the first "lower_coh" measurement
            new_idx = all_dates[all_dates >= idx.min()]

            new_idx_vals = all_dates[~pd.Index(all_dates).isin(idx)]
            lengths_raw["lower_coh"] = pd.concat(
                [
                    lengths_raw["front"][
                        lengths_raw["front"]["date"].isin(new_idx_vals)
                    ],
                    lengths_raw["lower_coh"],
                ]
            ).sort_values("date")

            # Interpolate the values in between
            lower_coh = (
                data.loc["lower_coh"]
                .drop(columns="exact")
                .reindex(new_idx)
                .interpolate("slinear")
                .bfill()
                .ffill()
            )
            lower_coh["exact"] = lower_coh.index.isin(idx)
            front_data = data.loc["front"].copy()
            front_data["exact"] = False
            front_data.loc[lower_coh.index] = lower_coh

            data = pd.concat(
                [data.drop(index="lower_coh"), to_multiindex(front_data, "lower_coh")]
            )

    data = data.sort_index()

    # Interpolate/bfill/ffill the main measurement for the type of surge.
    col = "lower_coh" if top_down else "upper_coh"
    coh = data.loc[col]
    orig_idx = coh.index.copy()
    coh = (
        coh.reindex(all_dates)
        .drop(columns=["exact"])
        .interpolate("slinear")
        .bfill()
        .ffill()
    )
    coh["exact"] = coh.index.isin(orig_idx)
    data = pd.concat([data.drop(index=col), to_multiindex(coh, col)])

    # Identify areas where the boundaries are outside of the terminus, and correct it.
    num_cols = ["upper", "lower", "mean", "median"]
    for idx in ["lower_coh", "upper_coh"]:
        mask = data.loc["front", num_cols] < data.loc[idx, num_cols]
        replacement = data.loc[["front"]]
        replacement.index = data.loc[[idx]].index
        data[to_multiindex(mask, idx)] = replacement

    expected_length = all_dates.shape[0] * 3
    if expected_length != data.shape[0]:
        raise ValueError(
            f"Expected length of the dataset ({expected_length}) is different from its shape ({data.shape})"
        )

    # Calculate front propagation velocities in m/d
    dt_days = pd.Series(
        data.loc["front"].index, data.loc["front"].index
    ).diff().dt.total_seconds() / (3600 * 24)
    for kind, kind_data in data.groupby(level=0):
        shifted = kind_data.shift(1)
        dt_days_multi = to_multiindex(dt_days, str(kind))

        data.loc[kind, "vel"] = (
            kind_data["median"] - shifted["median"]
        ) / dt_days_multi

        vel_upper = kind_data["upper"] - shifted["upper"]
        vel_lower = kind_data["lower"] - shifted["lower"]

        data.loc[kind, "vel_upper"] = (
            np.where(vel_upper > vel_lower, vel_upper, vel_lower) / dt_days_multi
        )
        data.loc[kind, "vel_lower"] = (
            np.where(vel_upper > vel_lower, vel_lower, vel_upper) / dt_days_multi
        )

    # Convert units from m to km
    data[num_cols] /= 1000

    data["surge_kind"] = "top-down" if top_down else "bottom-up"

    data.index.names = ["kind", "date"]
    # data["coh_zone_length"] = data.loc["lower_coh", "median"] - data.loc["upper_coh", "median"]
    coh_zone_length = data.loc["lower_coh", "median"] - data.loc["upper_coh", "median"]
    coh_zone_frac = coh_zone_length / data.loc["front", "median"]

    # Really messy way to reindex to include the "kind", and repeat the values over that axis.
    for key, new_data in [
        ("coh_zone_frac", coh_zone_frac),
        ("coh_zone_length", coh_zone_length),
    ]:
        data[key] = (
            pd.concat(
                {
                    key: new_data
                    for key in np.unique(data.index.get_level_values("kind"))
                },
                names=["kind"],
            )
            .reorder_levels(data.index.names)
            .reindex(data.index)
        )

    cache_filepath.parent.mkdir(exist_ok=True, parents=True)
    data.to_csv(cache_filepath)

    return data


def get_all_length_evolution(force_redo: bool = False):
    glaciers = gpd.read_file("GIS/shapes/glaciers.geojson")

    data = {}
    for key in glaciers["key"].values:
        data[key] = get_length_evolution(key, force_redo=force_redo)
    data = pd.concat(data, names=["key"])

    data.index.names = ["key", "kind", "date"]

    return data


def format_float(number: float, decimals: int = 1) -> str:
    """

    Examples
    --------
    >>> format_float(1.3456)
    '1.35'
    >>> format_float(1.3)
    '1.30'
    """
    rounded = str(round(float(number), decimals))

    if "." not in rounded:
        rounded += "." + "0" * decimals
    elif (n_zeros := len(rounded.split(".")[-1])) != decimals:
        rounded += "0" * (decimals - n_zeros)

    return rounded


def render_stats_table(
    stats,
    out_filepath: Path | None = None,
    max_title_line_len: int = 5,
    midrule_col: str | None = None,
):
    nice_names = {
        "reaching_front": "Surge reaching front",
        "surge_start": "Surge start",
        "surge_termination": "Surge termination",
        "surge_propagation_rate": f"Surge propagation rate ({TEX_VELOCITY_UNIT})",
        "instability_rate": f"Instability propagation rate ({TEX_VELOCITY_UNIT})",
        "predicted_advance_date": "Latest surge date",
        "pre_surge_bulge_speed": f"Bulge propagation rate ({TEX_VELOCITY_UNIT})",
        "surge_advance_rate": f"Surge advance rate ({TEX_VELOCITY_UNIT})",
        "post_surge_stagnation_rate": f"Post-surge relaxation rate ({TEX_VELOCITY_UNIT})",
        "surge_kind": "Surge kind",
    }

    for key, name in nice_names.items():
        if len(name) < max_title_line_len:
            continue
        words = name.split(" ")
        new_name = r"\specialcell{"
        lines = [words[0]]
        for word in words[1:]:
            if len(lines[-1]) >= max_title_line_len:
                lines.append(word)
            else:
                lines[-1] += " " + word

        nice_names[key] = r"\specialcell{" + "\\\\".join(lines) + "}"
    glaciers = gpd.read_file("GIS/shapes/glaciers.geojson")
    glacier_names = glaciers.set_index("key")["name"]
    tex = [
        r"% Needs 'booktabs' and the following command",
        r"% \newcommand{\specialcell}[2][c]{\begin{tabular}[#1]{@{}c@{}}#2\end{tabular}}",
        r"\begin{tabular}{l|" + "".join(["c"] * len(stats.columns)) + "}",
        "\t\\toprule",
        "\t"
        + r"\textbf{Glacier}&"
        + "&".join(
            [r"\textbf{" + nice_names.get(col, col) + "}" for col in stats.columns]
        )
        + r"\\",
        "\t\\midrule",
    ]

    category = "" if midrule_col is None else stats[midrule_col].iloc[0]

    for key, data in stats.iterrows():
        row = [glacier_names[key]]

        if midrule_col is not None:
            if data[midrule_col] != category:
                tex.append("\t\\midrule")
            category = data[midrule_col]

        for _, value in data.items():
            if pd.isna(value):
                value = "--"
            elif isinstance(value, pd.Timestamp):
                months = [
                    "Jan",
                    "Feb",
                    "Mar",
                    "Apr",
                    "May",
                    "Jun",
                    "Jul",
                    "Aug",
                    "Sep",
                    "Oct",
                    "Nov",
                    "Dec",
                ]
                value = f"{months[value.month - 1]}. {value.year}"
            elif isinstance(value, float):
                # value = round(value * 10) / 10
                value = format_float(value)
            row.append(str(value))

        tex.append("\t" + "&".join(row) + r"\\")
        # tex.append([glacier_names[key]

    tex += ["\t\\bottomrule", r"\end{tabular}"]

    if out_filepath is not None:
        with open(out_filepath, "w") as outfile:
            outfile.write("\n".join(tex))

    return "\n".join(tex)


def rec_update_dict(dictionary: dict, new: dict, i: int = 0, max_i: int = 20):
    if i == max_i:
        raise RecursionError("Max recursion limit reached")

    for key, value in new.items():
        if isinstance(value, collections.abc.Mapping):
            dictionary[key] = rec_update_dict(
                dictionary.get(key, {}), value, i=i + 1, max_i=max_i
            )
        else:
            dictionary[key] = value
    return dictionary


def rec_to_tex_macros(
    dictionary: dict, pre_key: str = "", i: int = 0, max_i: int = 20
) -> list[str]:
    if i == max_i:
        raise RecursionError("Max recursion limit reached")

    macros = []

    if i == 0:
        macros.append("% Autogenerated macro file.\n")
    for key, value in dictionary.items():
        new_key = ".".join([pre_key, key]) if len(pre_key) > 0 else key
        if isinstance(value, collections.abc.Mapping):
            macros += rec_to_tex_macros(value, pre_key=new_key, i=i + 1, max_i=max_i)
        else:
            macros += [
                f"% {new_key}",
                "\\def\\dataX"
                + new_key.replace(".", "X").replace("_", "")
                + "/{"
                + str(value)
                + "}",
            ]

    return macros


def format_values(obj: object, i: int = 0, max_i: int = 20) -> object:
    if isinstance(obj, float):
        formatted = format_float(obj)
        return formatted
    if isinstance(obj, collections.abc.Mapping):
        new_obj = {}
        for key, value in obj.items():
            new_obj[key] = format_values(value, i=i + 1, max_i=max_i)
        return new_obj
    elif isinstance(obj, collections.abc.Sequence):
        new_obj = []
        for value in obj:
            new_obj.append(format_values(value, i=i + 1, max_i=max_i))

    return obj


def record_information(new_info: dict[str, object] | None = None) -> dict[str, object]:
    info_path = Path("tables/statistics.json")
    info_tex_path = info_path.with_suffix(".tex")

    if info_path.is_file():
        with open(info_path) as infile:
            info = json.load(infile)
    else:
        info = {}

    if new_info is not None:
        info = rec_update_dict(info, new_info)

        with open(info_path, "w") as outfile:
            json.dump(
                format_values(info),
                outfile,
                indent=2,
            )
        with open(info_path) as infile:
            info = json.load(infile)

        info_tex = rec_to_tex_macros(info)
        with open(info_tex_path, "w") as outfile:
            outfile.write("\n".join(info_tex))

    return info


def classify_surge_stages(
    key: str, data: pd.DataFrame, bottom_up_surge_frac_threshold: float
):
    data = data.droplevel("key")

    is_top_down = (data["surge_kind"] == "top-down").iloc[0]

    if is_top_down:
        # stats = top_down_stats
        reached_front = (
            data.loc["lower_coh", "median"] >= (data.loc["front", "median"] * 0.99)
        )[data.loc["lower_coh", "exact"]]

        # Super annoying bug that I can't figure out
        if key == "morsnev":
            reached_front.loc["2016-08-21":"2018-01-01"] = True

        stage = reached_front.astype(int).diff().fillna(0).abs().cumsum()

        # If the surge has already reached the front by the first date, mark it as already surging.
        if reached_front.iloc[0]:
            stage += 1

        # If stagnation is caught, set the last surge point as part of the stagnation.
        # It makes more sense when seen in the figure.
        if (stage == 2).any():
            stage.loc[stage[stage == 1].index[-1]] = 2
    else:
        # stats = bottom_up_stats

        # coh_zone_width = data.loc["front", "median"] - data.loc["upper_coh", "median"]
        if key == "eton":
            surge_start_date = ETONBREEN_SURGE_START
        else:
            surge_start_date = (
                data.loc["front", "coh_zone_frac"]
                .pipe(lambda f: f[f > bottom_up_surge_frac_threshold])
                .sort_index()
                .index[0]
            )

        stage = pd.Series(
            np.zeros(data.loc["lower_coh"].shape[0]), data.loc["lower_coh"].index
        )

        stage.loc[slice(surge_start_date, None)] += 1

        surge_stop = data.loc["lower_coh", "median"] < (
            data.loc["front", "median"] * 0.99
        )
        if surge_stop.any():
            stage.loc[slice(surge_stop[~surge_stop].index[-1], None)] += 1

    if stage.max() > 2:
        # print(reached_front)
        print(stage)
        raise NotImplementedError(f"{key} had more stages than expected")

    indexes = {
        "pre_surge": [*stage[stage == 0].index, *stage[stage == 1].index[:1]],
        "surge": stage[stage == 1].index,
        "post_surge": stage[stage == 2].index,
    }

    return {
        key: data.loc[(slice(None), idx), :].sort_index()
        for key, idx in indexes.items()
    }


def plot_advance_rate_vs_coh_frac(bins, dig, binned_vel):
    plt.figure(figsize=(4, 2.5))

    binsize = np.mean(np.diff(bins))

    xmid = (bins[1:] - np.diff(bins) / 2)[np.unique(dig) - 1]
    plt.hlines(0, bins[0], bins[-1], color="black", linestyles=":", alpha=0.4)
    box_width = binsize / 2
    plt.boxplot(
        binned_vel,
        positions=xmid,
        widths=5,
        manage_ticks=False,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 2},
    )
    plt.xlabel("Fraction of glacier covered (%)")
    plt.ylabel(f"Advance/retreat rate ({TEX_VELOCITY_UNIT.replace('~', ' ')})")
    plt.gca().get_yaxis().set_label_coords(-0.1, 0.4)
    plt.xlim(xmid[0] - box_width, xmid[-1] + box_width)
    plt.xticks(
        xmid,
        labels=[f"{bins[i]}-{min(bins[i + 1], 100)}" for i in range(len(xmid))],
        rotation=30,
        ha="right",
    )
    plt.subplots_adjust(left=0.13, bottom=0.26, right=0.99, top=0.99)
    plt.savefig("figures/bottom_up_vel_vs_coh_frac.jpg", dpi=300)

    plt.close()


def calc_bottom_up_surge_threshold(
    data: pd.DataFrame,
    binsize: int = 10,
    vel_threshold: float = 0.1,
    plot: bool = False,
) -> float:
    """Calculate the low-coherence expanse threshold after which to call a bottom-up propagation a surge.

    Parameters
    ----------
    data
        Front data for bottom-up surges
    binsize
        The binsize in % to use for classification and plotting.
    vel_threshold
        The threshold (m/d) to use in order to consider a glacier advancing.
    plot
        Plot the data classification in a figure.

    Returns
    -------
    The fraction after which to call a bottom-up propagation a surge.
    """

    def get_col(kind: str, col: str = "median", data: pd.DataFrame = data) -> pd.Series:
        return data.loc[(slice(None), kind, slice(None)), col].droplevel(1)

    frac_data = (get_col("front", "coh_zone_frac", data.drop("eton")) * 100).values
    vel_data = (get_col("front", "vel", data.drop("eton"))).values
    valid_mask = np.isfinite(vel_data)
    frac_data = frac_data[valid_mask]
    vel_data = vel_data[valid_mask]

    bins = np.arange(0, 100 + binsize, binsize)
    dig = np.digitize(frac_data, bins)
    binned_vel = [vel_data[dig == i] for i in np.unique(dig)]

    advancing_mask = np.array([np.median(vals) for vals in binned_vel]) > vel_threshold
    bottom_up_surge_frac_threshold = bins[1:][advancing_mask][0] / 100

    if plot:
        plot_advance_rate_vs_coh_frac(bins=bins, dig=dig, binned_vel=binned_vel)

    return bottom_up_surge_frac_threshold


def surge_statistics(force_redo: bool = False) -> pd.DataFrame:
    """Classify and render surge statistics tables.

    All glacier data are read, classified (pre-surge/surge/post-surge) and statistics are extracted.

    Returns
    -------
    A dataframe of statistics for each glacier.

    """
    all_data = get_all_length_evolution(force_redo=force_redo)
    bottom_up = all_data[all_data["surge_kind"] == "bottom-up"]

    bottom_up_surge_frac_threshold = calc_bottom_up_surge_threshold(
        data=bottom_up, plot=True
    )
    print(
        f"Bottom-up surges start at {bottom_up_surge_frac_threshold * 100}% low-coh coverage"
    )

    bulge_stats = pd.DataFrame()

    top_down_stats = pd.DataFrame()
    bottom_up_stats = pd.DataFrame()
    for key, data in all_data.groupby(level=0):
        is_top_down = (data["surge_kind"] == "top-down").iloc[0]
        pre_surge, surge, post_surge = classify_surge_stages(
            key=key,
            data=data,
            bottom_up_surge_frac_threshold=bottom_up_surge_frac_threshold,
        ).values()

        stats = top_down_stats if is_top_down else bottom_up_stats

        if pre_surge.shape[0] > 0:
            if is_top_down:
                new_bulge_stats = pre_surge.loc[
                    "lower_coh", ["coh_zone_frac", "vel"]
                ].reset_index()
                new_bulge_stats["front_distance"] = (
                    pre_surge.loc["front", "median"]
                    - pre_surge.loc["lower_coh", "median"]
                ).values
                new_bulge_stats["vel_norm"] = (
                    new_bulge_stats["vel"] / new_bulge_stats["vel"].median()
                )
                new_bulge_stats["key"] = key

                bulge_stats = pd.concat(
                    [bulge_stats, new_bulge_stats], ignore_index=True
                )

                if surge.shape[0] > 0:
                    way_pre_date = pre_surge.index.get_level_values(
                        "date"
                    ).max() - pd.Timedelta(days=365)
                    way_pre_surge = pre_surge.loc[
                        (slice(None), slice(None, way_pre_date)), :
                    ]
                else:
                    way_pre_surge = pre_surge

                # Keep only the last three points to get the most "recent" trend.
                dates_to_keep = np.unique(way_pre_surge.index.get_level_values("date"))[
                    -3:
                ]
                if dates_to_keep.shape[0] > 1:
                    way_pre_surge = way_pre_surge.loc[(slice(None), dates_to_keep), :]

                stats.loc[key, "instability_rate"] = pre_surge.loc[
                    "lower_coh", "vel"
                ].mean()

                if way_pre_surge.loc["lower_coh"].shape[0] > 1:
                    times = way_pre_surge.loc["lower_coh"].index.values.astype(float)
                    model = np.polyfit(
                        times,
                        (
                            way_pre_surge.loc["front", "median"]
                            - way_pre_surge.loc["lower_coh", "median"]
                        ).values,
                        deg=1,
                    )
                    intercept_time = pd.Timestamp(-model[1] / model[0])
                    stats.loc[key, "predicted_advance_date"] = intercept_time

        if surge.shape[0] > 0:
            if key == "eton":
                stats.loc[key, "surge_start"] = ETONBREEN_SURGE_START
            else:
                stats.loc[key, "surge_start"] = surge.loc["lower_coh"].index[0]
            if is_top_down:
                stats.loc[key, "reaching_front"] = surge.loc["lower_coh"].index[0]
            first_year_surge = surge.loc[
                (
                    slice(None),
                    slice(
                        None,
                        surge.index.get_level_values(1).min()
                        + pd.Timedelta(days=365 * 1),
                    ),
                ),
                :,
            ]
            if not is_top_down:
                stats.loc[key, "instability_rate"] = -surge.loc[
                    "upper_coh", "vel"
                ].mean()
            stats.loc[key, "surge_advance_rate"] = first_year_surge.loc[
                "front", "vel"
            ].mean()

        if post_surge.shape[0] > 0:
            stats.loc[key, "surge_termination"] = post_surge.loc["lower_coh"][
                post_surge.loc["lower_coh", "exact"]
            ].index[0]

            stats.loc[key, "post_surge_stagnation_rate"] = (
                post_surge.loc["lower_coh", "vel"].mean() * -1
            )

    plt.figure()
    for i, (key, key_data) in enumerate(bulge_stats.groupby("key")):
        xcol = "coh_zone_frac"
        plt.scatter(
            key_data[xcol],
            key_data["vel_norm"],
            label=key,
            marker="o" if i < 9 else "s",
        )

        plt.ylabel("Velocity frac. (value / med)")
        plt.xlabel(
            {
                "front_distance": "Distance to front (km)",
                "coh_zone_frac": "Low coherence expanse fraction",
            }[xcol]
        )

        # key_data.plot.scatter("front_distance", "vel_norm", label=key, ax=axis)

    xlim = plt.gca().get_xlim()
    plt.hlines(1, *xlim, color="black", zorder=0)
    plt.xlim(xlim)
    plt.legend()
    plt.savefig("figures/bulge_stats.jpg", dpi=300)
    plt.close()

    top_down_stats = top_down_stats.sort_values("reaching_front")
    bottom_up_stats = bottom_up_stats.sort_values("surge_start")

    # print(stats.rename(columns=nice_names))
    tables_dir = Path("tables")
    tables_dir.mkdir(exist_ok=True)
    _tex = render_stats_table(
        top_down_stats, tables_dir / "top_down_surge_stats.tex", 5
    )
    _tex2 = render_stats_table(
        bottom_up_stats, tables_dir / "bottom_up_surge_stats.tex"
    )

    top_down_stats["surge_kind"] = r"$\downarrow$"
    bottom_up_stats["surge_kind"] = r"$\uparrow$"
    combined_stats = pd.concat([top_down_stats, bottom_up_stats])
    combined_stats.insert(0, "surge_kind", combined_stats.pop("surge_kind"))

    out_info = {
        "n_glaciers": {},
        "advance_rates": {
            "terminus": {},
            "instability": {},
            "stagnation": {},
            "other": {
                "top_down_bulge_vs_advance_rate_corr": top_down_stats[
                    "instability_rate"
                ].corr(top_down_stats["surge_advance_rate"]),
                "bottom_up_propagation_vs_advance_rate_corr": bottom_up_stats[
                    "instability_rate"
                ].corr(bottom_up_stats["surge_advance_rate"]),
            },
        },
    }

    for key, data in [
        ("bottom_up", bottom_up_stats),
        ("top_down", top_down_stats),
        ("all", combined_stats),
    ]:
        out_info["n_glaciers"][key] = data.shape[0]

        out_info["advance_rates"]["terminus"][f"{key}"] = data[
            "surge_advance_rate"
        ].median()
        out_info["advance_rates"]["terminus"][f"{key}_std"] = data[
            "surge_advance_rate"
        ].std()

        out_info["advance_rates"]["instability"][key] = data[
            "instability_rate"
        ].median()
        out_info["advance_rates"]["instability"][f"{key}_std"] = data[
            "instability_rate"
        ].std()

        out_info["advance_rates"]["stagnation"][key] = data[
            "post_surge_stagnation_rate"
        ].median()
        out_info["advance_rates"]["stagnation"][f"{key}_std"] = data[
            "post_surge_stagnation_rate"
        ].std()

    for key in ["terminus", "instability"]:
        fact = (
            out_info["advance_rates"][key]["top_down"]
            / out_info["advance_rates"][key]["bottom_up"]
        )
        out_info["advance_rates"][key]["top_down_vs_bottom_up_fact"] = round(fact)
        out_info["advance_rates"][key]["bottom_up_vs_top_down_fact"] = round(1 / fact)

    info = record_information(out_info)

    combined_stats.drop(columns=["reaching_front"], inplace=True)

    _tex3 = render_stats_table(
        combined_stats,
        tables_dir / "combined_surge_stats.tex",
        midrule_col="surge_kind",
    )
    return combined_stats


def surge_geometry_statistics():
    glaciers = gpd.read_file("GIS/shapes/glaciers.geojson")

    rgi = gpd.read_file(
        "zip://GIS/shapes/RGI2000-v7.0-G-07_svalbard_jan_mayen.zip/RGI2000-v7.0-G-07_svalbard_jan_mayen.shp"
    ).to_crs(glaciers.crs)

    geometric_stats = glaciers.sjoin(rgi).set_index("key")

    stats = surge_statistics()
    geometric_stats["zrange"] = geometric_stats["zmax_m"] - geometric_stats["zmin_m"]
    geometric_stats["tidewater"] = (geometric_stats["term_type"] == 1).astype(int)
    geometric_stats["geometry"] = geometric_stats["geometry"].centroid

    cols = ["zrange", "area_km2", "lmax_m", "slope_deg", "tidewater", "geometry"]
    stats[cols] = geometric_stats.loc[stats.index, cols]

    stats = gpd.GeoDataFrame(
        stats.drop(columns=["geometry"]), geometry=stats["geometry"], crs=glaciers.crs
    )

    stats.to_file("temp/surge_stats_with_geometric.geojson")

    stats.to_csv("temp/surge_stats_with_geometric.csv")
    stats["down_glacier_propagation"] = (stats["surge_kind"] == "D").astype(int)

    if False:
        from sklearn.inspection import permutation_importance
        from sklearn.linear_model import LinearRegression

        data = stats.select_dtypes(np.number).dropna(axis="columns", how="any")

        ycols = ["instability_rate", "surge_advance_rate", "post_surge_stagnation_rate"]

        for ycol in ycols:
            data = (
                stats.select_dtypes(np.number)
                .dropna(axis="rows", subset=ycol, how="any")
                .dropna(axis="columns", how="any")
            )

            xvals = data.drop(columns=[ycol])
            yvals = data[[ycol]]

            model = LinearRegression().fit(xvals, yvals)

            result = permutation_importance(
                model,
                xvals,
                yvals,
                n_repeats=30,
                random_state=42,
                scoring="neg_mean_squared_error",
            )

            print(
                f"Predicting {ycol} with n={data.shape[0]}: score: r={model.score(xvals, yvals):.2f}"
            )
            permutation_importance_df = pd.DataFrame(
                {
                    "Feature": xvals.columns,
                    "Importance": result.importances_mean,
                    "Std_Dev": result.importances_std,
                    "Coefficient": model.coef_.ravel(),
                }
            )

            print(permutation_importance_df.sort_values("Importance").round(2))
            print("\n\n")

    if True:
        numeric = stats.select_dtypes(np.number)
        corr = numeric.corr()
        plt.figure(figsize=(8.3, 8.3))
        plt.imshow(corr, cmap="coolwarm", interpolation="nearest", vmin=-1, vmax=1)

        nice_names = {
            "reaching_front": "Surge reaching front",
            "surge_start": "Surge start",
            "surge_termination": "Surge termination",
            "surge_propagation_rate": f"Surge propagation rate",
            "instability_rate": f"Low-coherence front propagation rate",
            "predicted_advance_date": "Latest surge date",
            "pre_surge_bulge_speed": f"Bulge propagation rate",
            "surge_advance_rate": f"Surge advance rate",
            "post_surge_stagnation_rate": f"Post-surge relaxation rate",
            "surge_kind": "Surge kind",
            "zrange": "Elev. range",
            "area_km2": "Area",
            "lmax_m": "Length",
            "slope_deg": "Slope",
            "tidewater": "Is tidewater",
            "down_glacier_propagation": "Propagates down-glacier",
        }

        plt.xticks(
            np.arange(corr.shape[1])[:-1],
            [nice_names.get(s, s) for s in corr.columns][:-1],
            rotation=30,
            ha="right",
        )
        plt.yticks(
            np.arange(corr.shape[0])[1:], [nice_names.get(s, s) for s in corr.index][1:]
        )
        for (i, j), val in np.ndenumerate(corr):
            plt.text(
                j,
                i,
                f"{val:.2g}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight=("bold" if abs(val) > 0.5 else "normal"),
            )

            if i == j == 5:
                print(~numeric[[numeric.columns[i], numeric.columns[j]]].isna())
            n_points = np.count_nonzero(
                (~numeric[[numeric.columns[i], numeric.columns[j]]].isna()).all(
                    axis="columns"
                )
            )
            plt.text(j, i + 0.3, f"n={n_points}", ha="center", va="top", fontsize=8)

        mask = np.triu(np.ones_like(corr, dtype=bool))
        # Apply mask
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                if mask[i, j]:
                    plt.gca().add_patch(
                        plt.Rectangle(
                            (j - 0.5, i - 0.5), 1, 1, color="white", lw=0, zorder=100
                        )
                    )

        plt.tight_layout()

        plt.savefig("figures/surge_stats_correlation.jpg", dpi=300)

        plt.show()


def plot_length_evolution(
    glacier: str = "arnesen",
    show: bool = False,
    force_redo: bool = False,
    fontsize: int = 8,
):
    try:
        name = (
            gpd.read_file("GIS/shapes/glaciers.geojson")
            .query(f"key == '{glacier}'")
            .iloc[0]["name"]
        )
    except IndexError as exception:
        raise ValueError(f"Key '{glacier}' not in glaciers.geojson") from exception

    data = get_length_evolution(glacier=glacier, force_redo=force_redo)

    plt.fill_between(
        np.unique(data.index.get_level_values(1)),
        data.loc["front", "median"],
        color="#" + "c" * 4 + "ff",
    )
    plt.fill_between(
        np.unique(data.index.get_level_values(1)),
        data.loc["upper_coh", "median"],
        data.loc["lower_coh", "median"],
        color="gray",
    )
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
    for kind, kind_data in data.groupby(level=0):
        params = all_params[str(kind)]

        kind_data = kind_data.loc[kind]
        exact_data = kind_data[kind_data["exact"]]
        plt.plot(
            kind_data.index,
            kind_data["median"],
            zorder=params["zorder"],
            color=params["color"],
            linestyle=":",
        )
        plt.scatter(
            exact_data.index,
            exact_data["median"],
            zorder=params["zorder"],
            color=params["color"],
            s=6,
        )
        plt.plot(
            np.repeat(exact_data.index, 3),
            np.array(
                (
                    exact_data["lower"].values,
                    exact_data["upper"].values,
                    np.zeros(exact_data.shape[0]) + np.nan,
                )
            ).T.ravel(),
            zorder=params["zorder"],
            color=params["color"],
        )

    ymax = data[data["exact"]]["median"].max()
    ymin = data[data["exact"]]["median"].min()
    yrange = ymax - ymin

    import matplotlib.dates as mdates
    from matplotlib.ticker import StrMethodFormatter

    if glacier in ["eton"]:
        plt.ylim(max(ymin - yrange * 0.4, 0), ymax + yrange * 0.2)
    else:
        plt.ylim(0, ymax * 1.15)

    plt.xlim(
        np.min(data.index.get_level_values(1)), np.max(data.index.get_level_values(1))
    )

    plt.text(
        0.5,
        0.97,
        name,
        transform=plt.gca().transAxes,
        va="top",
        ha="center",
        fontsize=fontsize,
    )

    xticks = plt.gca().get_xticks()

    # Late phase review quick-fix
    if glacier == "nordsyssel":
        plt.xticks([int(xticks[2]), int(xticks[-2])], fontsize=fontsize)
    else:
        plt.xticks(
            [int(xticks[1]), int(xticks[int(len(xticks) / 2)]), int(xticks[-2])],
            fontsize=fontsize,
        )
    plt.yticks(fontsize=fontsize)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

    if show:
        plt.tight_layout()
        plt.show()

    return data


def old_plot_length_evolution(glacier: str = "arnesen"):
    buffer_radius = 200 if glacier not in ["basin3"] else 2000.0

    front_positions, centerline, domain, coh_boundary = get_glacier_data(
        glacier=glacier
    )

    lower_coh_boundary = coh_boundary[coh_boundary["boundary_type"] == "lower"]
    upper_coh_boundary = coh_boundary[coh_boundary["boundary_type"] == "upper"]

    front_lengths_raw, front_lengths = measure_lengths(
        front_positions, centerline, domain, buffer_radius
    )

    velocities = {"front": measure_velocity(front_lengths_raw)}

    try:
        lower_coh_lengths_raw, lower_coh_lengths = measure_lengths(
            lower_coh_boundary, centerline, domain, buffer_radius
        )

        advancing_dates = (
            velocities["front"][velocities["front"]["median"] > 0.2]["date_to"]
            .astype("str")
            .values
        )
        advancing_front_positions = front_lengths[
            front_lengths["date"].isin(advancing_dates)
        ]
        advancing_front_positions_raw = front_lengths_raw[
            front_lengths_raw["date"].isin(advancing_dates)
        ]

        lower_coh_lengths = pd.concat(
            [lower_coh_lengths, advancing_front_positions]
        ).sort_values("date")
        lower_coh_lengths_raw = pd.concat(
            [lower_coh_lengths_raw, advancing_front_positions_raw]
        ).sort_values("date")


        velocities["lower_coh"] = measure_velocity(lower_coh_lengths_raw)
    except KeyError:
        lower_coh_lengths_raw = lower_coh_lengths = None

    try:
        upper_coh_lengths_raw, upper_coh_lengths = measure_lengths(
            upper_coh_boundary, centerline, domain, buffer_radius
        )
        velocities["upper_coh"] = measure_velocity(upper_coh_lengths_raw)
    except KeyError:
        upper_coh_lengths_raw = upper_coh_lengths = None

    Path("figures/").mkdir(exist_ok=True)

    fig = plt.figure(figsize=(8, 5))
    for data, params in [
        (
            front_lengths,
            {"color": "blue", "label": "Terminus", "zorder": 2, "key": "front"},
        ),
        (
            lower_coh_lengths,
            {
                "color": "orange",
                "label": "Lower low-coh. bnd.",
                "zorder": 1,
                "key": "lower_coh",
            },
        ),
        (
            upper_coh_lengths,
            {
                "color": "green",
                "label": "Upper low-coh. bnd.",
                "zorder": 1,
                "key": "upper_coh",
            },
        ),
    ]:
        if data is None:
            continue

        plt.subplot(121)
        plt.fill_between(
            data["date"],
            data["lower"] / 1e3,
            data["upper"] / 1e3,
            alpha=0.5,
            color=params["color"],
            zorder=params["zorder"],
        )
        plt.plot(
            data["date"],
            data["median"] / 1e3,
            color=params["color"],
            label=params["label"],
            zorder=params["zorder"],
        )
        plt.scatter(
            data["date"],
            data["median"] / 1e3,
            color=params["color"],
            zorder=params["zorder"],
        )
        plt.ylabel("Glacier length (km)")
        plt.xlabel("Year")

        plt.subplot(122)
        plt.fill_between(
            np.ravel(
                np.column_stack(
                    (
                        velocities[params["key"]].index.left,
                        velocities[params["key"]].index.right,
                    )
                )
            ),
            np.repeat(velocities[params["key"]]["lower"], 2),
            np.repeat(velocities[params["key"]]["upper"], 2),
            alpha=0.3,
            color=params["color"],
            zorder=params["zorder"],
        )
        plt.plot(
            np.ravel(
                np.column_stack(
                    (
                        velocities[params["key"]].index.left,
                        velocities[params["key"]].index.right,
                    )
                )
            ),
            np.repeat(velocities[params["key"]]["median"], 2),
            color=params["color"],
            zorder=params["zorder"],
            label=params["label"],
        )
        plt.ylabel("Advance/retreat rate (m/d)")
        plt.xlabel("Year")
        plt.gca().yaxis.tick_right()
        plt.gca().yaxis.set_label_position("right")

    plt.legend()
    plt.tight_layout()
    # plt.savefig(f"figures/{glacier}_front_change.jpg", dpi=300)
    plt.show()


def plot_multi_front_velocity(show: bool = True, force_redo: bool = False):
    n_cols = 4
    glaciers = [[]]
    for glacier in order_surges([], all_default=True):
        if len(glaciers[-1]) >= n_cols:
            glaciers.append([])
        glaciers[-1].append(glacier)

    all_data = {
        key: get_length_evolution(key, force_redo=force_redo)
        for key in np.ravel(glaciers)
    }

    all_params = {
        "front": {
            "color": "blue",
            "zorder": 3,
            "label": "Terminus",
        },
        "lower_coh": {
            "color": "red",
            "zorder": 2,
            "label": "Lower coh.",
        },
        "upper_coh": {
            "color": "green",
            "zorder": 4,
            "label": "Upper coh.",
        },
    }

    fig = plt.figure(figsize=(8, 2 * len(glaciers)), dpi=100)
    n_rows = len(glaciers)
    for row_n, row in enumerate(glaciers):
        for col_n, glacier in enumerate(row):
            i = col_n + n_cols * row_n
            plt.subplot(n_rows, n_cols, i + 1)
            data = all_data[glacier]
            try:
                name = (
                    gpd.read_file("GIS/shapes/glaciers.geojson")
                    .query(f"key == '{glacier}'")
                    .iloc[0]["name"]
                )
            except IndexError as exception:
                raise ValueError(
                    f"Key '{glacier}' not in glaciers.geojson"
                ) from exception

            xlim = (
                data.index.get_level_values("date").min() - pd.Timedelta(days=120),
                data.index.get_level_values("date").max() + pd.Timedelta(days=120),
            )

            plt.hlines(0, *xlim, color="black", linestyles="--", alpha=0.3)
            for kind, kind_data in data.groupby(level="kind"):
                params = all_params[kind]

                kind_data = kind_data.droplevel("kind")

                if (
                    kind == "upper_coh"
                    and (kind_data["vel"].fillna(0).abs() < 0.01).all()
                ):
                    continue

                kind_data = kind_data[kind_data["exact"]]

                plt.fill_between(
                    kind_data.index,
                    kind_data["vel_lower"],
                    kind_data["vel_upper"],
                    color=params["color"],
                    zorder=params["zorder"],
                    alpha=0.3,
                )

                plt.plot(
                    kind_data.index,
                    kind_data["vel"],
                    color=params["color"],
                    zorder=params["zorder"],
                    label=params["label"],
                )
                plt.scatter(
                    kind_data.index,
                    kind_data["vel"],
                    zorder=params["zorder"],
                    color=params["color"],
                    s=6,
                )


            plt.text(
                0.5, 0.97, name, transform=plt.gca().transAxes, va="top", ha="center"
            )
            ymax = data["vel_upper"].max()
            ymin = data["vel_lower"].min()
            yrange = ymax - ymin

            plt.ylim(ymin - yrange * 0.15, ymax + yrange * 0.15)

            plt.xlim(xlim)
            xticks = plt.gca().get_xticks()
            import matplotlib.dates as mdates
            from matplotlib.ticker import StrMethodFormatter

            plt.xticks([int(xticks[1]), xticks[int(len(xticks) / 2)], xticks[-1]])
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            plt.gca().yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

            plt.text(
                0.01,
                0.99,
                "abcdefghijklmnopqrstuvx"[i] + ")",
                transform=plt.gca().transAxes,
                fontsize=9,
                ha="left",
                va="top",
            )

    plt.tight_layout(w_pad=-0.5, rect=(0.02, 0.0, 1.0, 1.0))
    plt.text(
        0.01,
        0.5,
        "Advance/retreat rate (m/d)",
        rotation=90,
        ha="center",
        va="center",
        transform=fig.transFigure,
    )

    plt.savefig("figures/front_velocity.jpg", dpi=300)
    if show:
        plt.show()
    plt.close()


def get_npi_data(layer: str):
    if "CryoClim" in layer:
        url = "https://next.api.npolar.no/dataset/89f430f8-862f-11e2-8036-005056ad0004/attachment/4494bb0d-8b90-480b-aed1-47f7955ce81b/_blob"
        uri = "/vsizip/{" + f"/vsicurl/{url}" + "}" + f"/{layer}.shp"

    else:
        url = "https://public.data.npolar.no/kartdata/NP_S100_SHP.zip"
        uri = f"/vsizip/vsicurl/{url}/NP_S100_SHP/{layer}.shp"

    cache_filepath = CACHE_DIR / (layer + ".gpkg")

    if cache_filepath.is_file():
        return gpd.read_file(cache_filepath, layer="outlines")

    outlines = gpd.read_file(uri)
    # Remove Bjørnøya
    outlines = outlines[outlines["geometry"].centroid.y > 8.4e6]

    # The coordinate systems are essentially interchangeable
    # Converting did not work as one value seems invalid
    outlines.crs = rasterio.CRS.from_epsg(32633)

    outlines.to_file(cache_filepath, driver="GPKG", layer="outlines")
    return outlines


def get_svalbard_outlines() -> gpd.GeoDataFrame:
    return get_npi_data("S100_Land_f")


def get_svalbard_glaciers() -> gpd.GeoDataFrame:
    return get_npi_data("CryoClim_GAO_SJ_2001-2010")


def plot_multi_front_evolution(show: bool = True, force_redo: bool = False):
    n_cols = 4
    glacier_points = gpd.read_file("GIS/shapes/glaciers.geojson")

    glaciers = [[]]
    for glacier in order_surges([], all_default=True):
        if len(glaciers[-1]) >= n_cols:
            glaciers.append([])
        glaciers[-1].append(glacier)

    stats = surge_statistics()

    # These groups will each have associated glaciers plotted and a rectangle outline around them.
    groups = {
        "Progressing surge bulges": {
            "loc": (2, 0),
            "glaciers": [["edvard", "doktor"]],
        },
        "Top-down (bulge) initiated surges": {
            "loc": (0, 2),
            "glaciers": [
                ["sefstrom", "natascha", "delta"],
                ["nordsyssel", "vallakra", "scheele"],
                ["liestol", "penck", "morsnev"],
            ],
        },
        "Terminus initiated surges": {
            "loc": (0, 5),
            "glaciers": [
                ["arnesen", "osborne"],
                ["kval", "stone"],
                ["eton", "bore"],
            ],
        },
    }
    per_group_n_cols = [
        max(len(row) for row in group["glaciers"]) for group in groups.values()
    ]
    n_cols = sum(per_group_n_cols)

    groups[list(groups.keys())[1]]["loc"] = (0, per_group_n_cols[0])
    groups[list(groups.keys())[2]]["loc"] = (0, sum(per_group_n_cols[:2]))

    fig = plt.figure(figsize=(n_cols * 10 / 6, 6))

    # The shape of the subplot2grid layout
    gridshape = (3, n_cols)
    margins = {
        "left": 0.04,
        "bottom": 0.056,
        "right": 0.99,
        "top": 0.987,
        "wspace": 0.274,
        "hspace": 0.256,
    }
    # The separator lines for different parts of the figure needs shifting a bit
    grid_shift_top = 0.03
    grid_shift_right = 0.005

    # The overview map for some reason has very weird bounds (it starts too much to the left and down)
    # These are used at the very end to fix that.
    overview_bottom = 1 - 2 / gridshape[0] + grid_shift_top
    overview_right = 2 / gridshape[1] + grid_shift_right

    # Plot the overview map
    overview_axis = plt.subplot2grid(gridshape, (0, 0), rowspan=2, colspan=2)
    overview_axis.set_axis_off()
    overview_axis.margins(0)
    outlines = get_svalbard_outlines()
    overview_plot_kwargs = {
        "ax": overview_axis,
        "edgecolor": "black",
        "linewidth": 0.05,
    }
    outlines.dissolve().plot(color="lightgray", **overview_plot_kwargs)
    # glacier_outlines.plot(column="used")
    label_fontsize = 11
    plt.text(
        0.01, 0.99, "a", fontsize=label_fontsize, va="top", transform=fig.transFigure
    )

    # Add an outline rectangle to the overview map
    fig.patches.append(
        plt.Rectangle(
            (0, overview_bottom),
            overview_right,
            1 - overview_bottom,
            transform=fig.transFigure,
            figure=fig,
            zorder=1000,
            facecolor="none",
            edgecolor="black",
        )
    )

    # Initialize lists for the label annotations that should go on the overview map.
    overview_labels = {"x": [], "y": [], "s": []}
    i = 1  # This counter assigns the letter. It should start at b) (i=1)
    for group in groups:
        start_loc = groups[group]["loc"]
        glaciers = groups[group]["glaciers"]

        # Define the bounds of the rectangles to draw
        rect = {
            "start_x": start_loc[1] / gridshape[1],
            "height": (len(glaciers) / gridshape[0]),
            "width": (max(map(len, glaciers)) / gridshape[1]),
        }
        rect["start_y"] = (1 - start_loc[0] / gridshape[0]) - rect["height"]
        # Clamp the "end_y" to 1
        rect["height"] = (
            min(1, rect["start_y"] + rect["height"] + grid_shift_top) - rect["start_y"]
        )
        # Shift the whole rectangle grid a bit
        if rect["start_x"] > 0.0:
            rect["start_x"] += grid_shift_right
        else:
            rect["width"] += grid_shift_right

        # Make sure the "end_x" never surpasses 1
        rect["width"] = min(1.0, rect["start_x"] + rect["width"]) - rect["start_x"]
        # The bounds are needed later for labelling. This stores the bounds info.
        groups[group]["rect"] = rect

        # Add each glacier
        for row_n, row in enumerate(glaciers):
            for col_n, glacier in enumerate(row):
                glacier_point = glacier_points.query(f"key == '{glacier}'").iloc[0]
                axis = plt.subplot2grid(
                    gridshape, (start_loc[0] + row_n, start_loc[1] + col_n)
                )

                data = plot_length_evolution(
                    glacier, force_redo=force_redo, fontsize=label_fontsize - 2
                )

                lims = {}
                ylim = plt.gca().get_ylim()
                xlim = plt.gca().get_xlim()
                if not pd.isna(surge_start := stats.loc[glacier, "surge_start"]):
                    data["date_diff_to_start"] = np.abs(
                        (data.index.get_level_values("date") - surge_start)
                        .total_seconds()
                        .values
                    ) / (3600 * 24)

                    closest_lower = (
                        data.loc["lower_coh"].sort_values("date_diff_to_start").iloc[0]
                    )
                    closest_upper = (
                        data.loc["upper_coh"].sort_values("date_diff_to_start").iloc[0]
                    )
                    lims["surge_start_coh"] = (
                        closest_upper["median"],
                        closest_lower["median"],
                    )

                    plt.vlines(
                        surge_start,
                        ylim[0],
                        closest_lower["median"],
                        linestyles="--",
                        color="darkgray",
                        linewidth=2,
                    )
                    surge_start = matplotlib.dates.date2num(surge_start)
                else:
                    surge_start = xlim[0]
                if not pd.isna(
                    surge_termination := stats.loc[glacier, "surge_termination"]
                ):
                    data["date_diff_to_end"] = np.abs(
                        (data.index.get_level_values("date") - surge_termination)
                        .total_seconds()
                        .values
                    ) / (3600 * 24)

                    closest_lower = (
                        data.loc["lower_coh"].sort_values("date_diff_to_end").iloc[0]
                    )
                    closest_upper = (
                        data.loc["upper_coh"].sort_values("date_diff_to_end").iloc[0]
                    )

                    lims["surge_termination_coh"] = (
                        closest_upper["median"],
                        closest_lower["median"],
                    )

                    plt.vlines(
                        surge_termination,
                        ylim[0],
                        closest_lower["median"],
                        linestyles=":",
                        color="darkgray",
                        linewidth=2,
                    )
                    surge_termination = matplotlib.dates.date2num(surge_termination)
                else:
                    surge_termination = xlim[1]

                if surge_termination != xlim[1] or surge_start != xlim[0]:
                    surge_ylim = (
                        max(
                            lims.get("surge_start_coh", ylim)[0],
                            lims.get("surge_termination_coh", ylim)[0],
                        ),
                        min(
                            lims.get("surge_start_coh", ylim)[1],
                            lims.get("surge_termination_coh", ylim)[1],
                        ),
                    )
                    y_line = np.mean(surge_ylim)
                    x_mid = np.mean([surge_start, surge_termination])

                    text = plt.annotate(
                        "Surge",
                        (x_mid, y_line),
                        ha="center",
                        va="bottom",
                        fontsize=label_fontsize,
                        color="white",
                    )  # , path_effects=[matplotlib.patheffects.withStroke(linewidth=0.5, foreground="black")])

                    bbox = matplotlib.transforms.TransformedBbox(
                        text.get_window_extent(), plt.gca().transData.inverted()
                    )

                    remove_label = False

                    if bbox.x0 < xlim[0] or bbox.x1 > xlim[1]:
                        remove_label = True

                    if remove_label:
                        text.remove()

                    plt.hlines(
                        y_line,
                        surge_start,
                        surge_termination,
                        color="darkgray",
                        linewidth=1,
                    )

                letter = "abcdefghijklmnopqrstuvx"[i]
                plt.text(
                    0.01,
                    0.99,
                    letter,
                    transform=axis.transAxes,
                    fontsize=label_fontsize,
                    ha="left",
                    va="top",
                )
                # Again, these data are for showing on the overview map
                overview_labels["x"].append(glacier_point.geometry.x)
                overview_labels["y"].append(glacier_point.geometry.y)
                overview_labels["s"].append(letter)
                i += 1

    glacier_outlines = get_svalbard_glaciers()
    glacier_outlines["used"] = glacier_outlines.geometry.intersects(
        shapely.geometry.MultiPoint(
            np.transpose([overview_labels["x"], overview_labels["y"]])
        )
    )

    glacier_outlines.query("used == False").plot(
        color="lightblue", **overview_plot_kwargs
    )
    glacier_outlines.query("used == True").plot(color="orange", **overview_plot_kwargs)

    # Annotate the locations of the glaciers with their labels. This is nontrivial as the labels would
    # overlap without this package that makes sure they don't. The lines are stupid though so I'm making them myself.
    new_positions, _, texts, *_ = textalloc.allocate(
        overview_axis,
        overview_labels["x"],
        overview_labels["y"],
        overview_labels["s"],
        x_scatter=overview_labels["x"],
        y_scatter=overview_labels["y"],
        textsize=9,
        draw_lines=False,
        min_distance=0.0,
        margin=0.0,
        ha="center",
        va="center",
    )

    # Draw lines from each label to the location's exact position
    for i, point in enumerate(new_positions):
        # This is the uncut line from label to the exact location
        line = shapely.geometry.LineString(
            [[point[0], point[1]], [overview_labels["x"][i], overview_labels["y"][i]]]
        )

        # We don't want the line within the bounding box of the label itself.
        bbox = (
            texts[i].get_window_extent().transformed(overview_axis.transData.inverted())
        )
        # Extract and then plot the part of the line that's outside the bounding box.
        line_diff = line.difference(shapely.geometry.box(*bbox.extents))
        overview_axis.plot(*line_diff.xy, color="gray", linewidth=0.5)

    plt.subplots_adjust(**margins)
    # For some reason, this call is needed here and not above. Maybe from the subplots_adjust?
    overview_axis.set_position(
        [
            0.01,
            overview_bottom + 0.01,
            overview_right - 0.01,
            1 - overview_bottom - 0.01,
        ]
    )
    # Add a y-axis label
    plt.text(
        0.01,
        list(groups.values())[0]["rect"]["height"] / 2,
        "Distance (km)",
        rotation=90,
        ha="center",
        va="center",
        transform=fig.transFigure,
        fontsize=11,
    )

    plt.savefig("figures/surge_front_evolution.jpg", dpi=300)

    if show:
        plt.show()

    plt.close()


def main():
    plot_multi_front_evolution()


def load_coh(glacier: str):
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

    front_positions, centerline, domain, coh_boundary = get_glacier_data(
        glacier=glacier
    )

    bounds = rasterio.coords.BoundingBox(*domain.buffer(200).bounds)

    res = (40.0, 40.0)
    transform = rasterio.transform.from_origin(bounds.left, bounds.top, *res)
    out_shape = (
        int((bounds.top - bounds.bottom) / res[1]),
        int(np.ceil((bounds.right - bounds.left) / res[0])),
    )

    meta = {}
    data = {"coh": {}}

    for year in default_files_hh:
        for filepath in itertools.chain(
            *(
                map(
                    Path,
                    fnmatch.filter(map(str, Path("insar/").glob("*.zip")), pattern),
                )
                for pattern in default_files_hh[year]
            )
        ):
            # for pattern in default_files_hh[year]:
            #     for filepath in map(Path, fnmatch.filter(map(str, Path("insar/").glob("*.zip")), pattern)):

            filename = f"/vsizip/{filepath}/{filepath.stem}/{filepath.stem}_corr.tif"

            with rasterio.open(filename) as raster:
                if (raster.bounds.left < domain.centroid.x < raster.bounds.right) and (
                    raster.bounds.bottom < domain.centroid.y < raster.bounds.top
                ):
                    test_val = (
                        raster.sample(
                            [[domain.centroid.x, domain.centroid.y]], masked=True
                        )
                        .__next__()
                        .filled(np.nan)[0]
                    )
                    if not np.isfinite(test_val):
                        continue

                    window = rasterio.windows.from_bounds(
                        *bounds, transform=raster.transform
                    )
                    data["coh"][year] = raster.read(1, window=window, boundless=True)
                    meta["shape"] = data["coh"][year].shape

    max_centerline_length = {"natascha": 18300}.get(glacier, centerline.length)

    width = res[0] * 3
    centerline_pts = []
    for dist in np.arange(width, max_centerline_length, step=width):
        centerline_pts.append((centerline.interpolate(dist).buffer(width), int(dist)))

    data["centerline"] = rasterio.features.rasterize(
        centerline_pts, out_shape=meta["shape"], transform=transform
    )

    bins = np.r_[
        [-100],
        np.linspace(width, max_centerline_length, num=20),
        [centerline.length * 2],
    ]
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
                plt.scatter(yvals[j - 1], xvals[j - 1], marker="x", s=60, color="red")
                break

        yticks = plt.gca().get_yticks()
        if i != 0:
            plt.yticks(yticks, [""] * len(yticks))
        else:
            plt.yticks(yticks)
            plt.ylabel("Distance (km)")
            plt.xlabel("Coherence")

        plt.xlim(0, 1)
        plt.xticks([0.0, 0.5, 1.0], None if i == 0 else [""] * 3)

        plt.grid(alpha=0.5)

    if glacier == "natascha":
        plt.subplots_adjust(left=0.08, bottom=0.05, right=0.99, top=0.95, wspace=0.01)
    else:
        plt.tight_layout()
    plt.savefig(f"figures/{glacier}_coh_along_center.jpg", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
