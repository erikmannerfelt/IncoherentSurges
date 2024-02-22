import asf_search
import shapely
import shapely.geometry
import geopandas as gpd
import pandas as pd
import warnings
from typing import Literal
import projectfiles
from pathlib import Path
import pyproj
import matplotlib.pyplot as plt
import tqdm
import tqdm.contrib.concurrent
import fnmatch
import hyp3_sdk
import numpy as np
import datetime
import glob


CACHE_DIR = Path(__file__).parent / "cache"

def query_scenes(roi: shapely.geometry.Polygon | shapely.geometry.Point, max_results: int | None = 5, platform: Literal["S1A"] | Literal["S1B"] | Literal["both"] = "S1A", start_date: pd.Timestamp | None = None, end_date: pd.Timestamp | None = None) -> gpd.GeoDataFrame:

    checksum = projectfiles.get_checksum([roi.wkt, max_results, platform, start_date, end_date])

    cache_path = CACHE_DIR / f"query_scenes-{checksum}.geojson"

    if cache_path.is_file():
        meta = gpd.read_file(cache_path)

    else:
        CACHE_DIR.mkdir(exist_ok=True)

        match platform:
            case "S1A": 
                platforms = [asf_search.PLATFORM.SENTINEL1A]
            case "S1B":
                platforms = [asf_search.PLATFORM.SENTINEL1B]
            case "both":
                platforms = [asf_search.PLATFORM.SENTINEL1A, asf_search.PLATFORM.SENTINEL1B]

        print("Querying ASF search")
        results = asf_search.geo_search(
            platform=platforms,
            intersectsWith=roi.wkt,
            maxResults=max_results,
            beamMode="IW",
            processingLevel="SLC",
            start=start_date.isoformat() if start_date is not None else None,
            end=end_date.isoformat() if end_date is not None else None,
        )

        meta = gpd.GeoDataFrame.from_features(results.geojson())
        meta.crs = pyproj.CRS.from_epsg(4326)

        meta.to_file(cache_path)

        meta = gpd.read_file(cache_path)

    meta["stopTime"] = pd.to_datetime(meta["stopTime"], format="ISO8601")

    return meta


def query_baselines(scene_names: list[str], max_temporal_baseline_days: int = 13, min_temporal_baseline_days: int = 11) -> gpd.GeoDataFrame:

    scene_names = [str(name) for name in scene_names]

    checksum = projectfiles.get_checksum([scene_names, max_temporal_baseline_days, min_temporal_baseline_days])

    cache_path = CACHE_DIR / f"query_baselines-{checksum}.geojson"

    if cache_path.is_file():
        basepairs = gpd.read_file(cache_path)
    else:

        basepairs = []

        def get_baseline(scene_name):
            ref = asf_search.granule_search([scene_name])[0]
            stop_time = pd.to_datetime(ref.geojson()["properties"]["stopTime"])

            search_start = stop_time - pd.Timedelta(days=max_temporal_baseline_days)
            search_stop = stop_time

            # help(result.get_stack_opts)
            stack_opts = ref.get_stack_opts()
            stack_opts.start = search_start.isoformat()
            stack_opts.end = search_stop.isoformat()
            stack_opts.platform = asf_search.PLATFORM.SENTINEL1A

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    stack = ref.stack(opts=stack_opts)
            except ValueError as exception:
                if "No products found matching" in str(exception):
                    return None
                raise exception

            if len(stack) < 2:
                return None

            other = stack[0]

            ref = gpd.GeoDataFrame.from_features([ref.geojson()]).iloc[0]
            other = gpd.GeoDataFrame.from_features([other.geojson()]).iloc[0]

            intersection = ref["geometry"].intersection(other["geometry"])

            return {
                    "geometry": intersection,
                    "ref_name": ref["sceneName"],
                    "ref_time": ref["startTime"],
                    "ref_platform": ref["platform"],
                    "other_name": other["sceneName"],
                    "other_time": other["startTime"],
                    "other_platform": other["platform"],
                    "overlap_frac": intersection.area / ref["geometry"].area,
                } | other[["pathNumber", "frameNumber", "polarization", "flightDirection", "perpendicularBaseline", "temporalBaseline"]].to_dict()
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = tqdm.contrib.concurrent.thread_map(get_baseline, scene_names, desc="Querying baselines")

        basepairs = [pair for pair in result if pair is not None]
            
        basepairs = gpd.GeoDataFrame.from_records(basepairs)
        # basepairs = basepairs.set_geometry("geometry")
        basepairs.crs = pyproj.CRS.from_epsg(4326)

        basepairs = basepairs[basepairs["temporalBaseline"] < (-min_temporal_baseline_days)]

        basepairs.to_file(cache_path)
        basepairs = gpd.read_file(cache_path)

    for col in ["ref_time", "other_time"]:
        basepairs[col] = pd.to_datetime(basepairs[col], format="ISO8601")

    basepairs["product_glob"] = (
        "S1" + 
        basepairs["other_platform"].str.replace("Sentinel-1", "") +
        basepairs["ref_platform"].str.replace("Sentinel-1", "") +
        "_" +
        basepairs["other_name"].apply(lambda name: name.split("_")[5]) + 
        "_" +
        basepairs["ref_name"].apply(lambda name: name.split("_")[5][:9]) + 
        "*_" + 
        basepairs["polarization"].str.replace(r"\+.*", "", regex=True) +
        "*INT40_G_ueF*.zip"
    )

    basepairs["job_name"] = basepairs["product_glob"].str.replace("*", "").str.replace("\_G_ueF.*", "", regex=True)

    return basepairs

def get_downloaded_files(download_dir: Path = CACHE_DIR / "../insar"):

    files = []
    extra_files = Path("files.txt")
    if extra_files.is_file():
        with open(extra_files) as infile:
            for filename in infile.read().splitlines():
                if any(p in filename for p in ["S1AB", "S1BA", "S1BB", "INT80"]):
                    continue
                files.append(filename)

    for filepath in download_dir.rglob("*.zip"):

        files.append(filepath.name)

    return files

def filter_existing_baselines(baselines: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    filenames = get_downloaded_files()

    files_present = []
    for i, baseline in baselines.iterrows():
        matches = fnmatch.filter(filenames, baseline["product_glob"])
        if len(matches) == 0:
            continue

        if len(matches) > 1:
            raise ValueError(f"Pattern {baseline['product_glob']} matched multiple files")

        files_present.append(i)

    return baselines.drop(files_present).sort_values("ref_time")
    

def now_str() -> str:
    return datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    

def report_jobs(hyp3_instance: hyp3_sdk.HyP3, jobs: hyp3_sdk.Batch, download_location: Path, verbose: bool = True):

    running_jobs = jobs.filter_jobs(running=True)

    for job in running_jobs:
        print(job.files)

    if len(running_jobs) > 0:
        if verbose:
            print(f"{now_str()}: Awaiting {len(running_jobs)} jobs")

        hyp3_instance.watch(running_jobs)

        
        jobs = hyp3_instance.refresh(jobs)
        
    succeeded_jobs = jobs.filter_jobs(succeeded=True, running=False, failed=False)
    failed_jobs = jobs.filter_jobs(succeeded=False, running=False, failed=True)

    if len(failed_jobs) > 0:
        print(f"{len(failed_jobs)} failed jobs: {failed_jobs}")

    jobs_to_download = []
    for job in succeeded_jobs:
        for inner_file in job.files:
            filepath = download_location / inner_file["filename"]
            if not filepath.is_file():
                jobs_to_download.append(job)

    for job in tqdm.tqdm(jobs_to_download, desc="Downloading data"):
        job.download_files(download_location)

        
    

def download_baselines(baselines: gpd.GeoDataFrame, jobs_per_batch: int = 10, download_location: Path = CACHE_DIR / "../insar"):
    hyp3 = hyp3_sdk.HyP3()

    report_jobs(
        hyp3,
        hyp3.find_jobs(name="insar-download-python"),
        download_location=download_location,
        verbose=True,
    )
    baselines = filter_existing_baselines(baselines)

    if baselines.shape[0] == 0:
        return

    batches = []
    splits = np.arange(baselines.shape[0] + jobs_per_batch, step=jobs_per_batch)[1:]
    splits = splits[splits < baselines.shape[0]]
    batches = np.split(baselines.index, splits)


    print(f"Temporary break: {baselines.shape[0]} jobs to go")
    return
    for indices in batches:
        jobs = hyp3_sdk.Batch()

        for _, baseline in baselines.loc[indices].iterrows():
            jobs += hyp3.submit_insar_job(
                    granule1=baseline["other_name"],
                    granule2=baseline["ref_name"],
                    looks="10x2", 
                    include_look_vectors=True,
                    include_dem=True,
                    include_displacement_maps=False,
                    include_inc_map=False,
                    include_wrapped_phase=False,
                    apply_water_mask=False,
                    phase_filter_parameter=0.6,
                    name="insar-download-python",
            )

        print(f"{now_str()}: Submitted {len(jobs)} jobs")
        jobs = hyp3.watch(jobs)

        report_jobs(hyp3, jobs, download_location=download_location, verbose=False)

    

def main():

    # Mid-Svalbard
    # roi = shapely.geometry.box(18.43336, 78.49224, 18.91799, 78.58287)

    # All of Svalbard
    roi = gpd.read_file("GIS/shapes/svalbard_roi.geojson").geometry[0]
    # roi = shapely.geometry.box(10, 76, 30, 81)

    all_meta = query_scenes(roi=roi, max_results=None)

    baselines = query_baselines(all_meta["sceneName"].values.tolist())

    baselines = baselines[(baselines["ref_time"].dt.month < 5) & (baselines["other_time"].dt.month >= 1)]


    # TEMPORARY. For testing
    baselines = baselines[baselines["ref_time"].dt.year == 2024]
    baselines = baselines.iloc[:3]


    # baselines = baselines.query('flightDirection == "DESCENDING"')

    to_process = filter_existing_baselines(baselines)
    print(f"{len(baselines) - len(to_process)} files present. {len(to_process)} to process.")

    if to_process.shape[0] == 0:
        print("All scenes exist")
        return

    download_baselines(to_process)


    
    # print(baselines[baselines["other_time"].dt.month < 5])
    # print(files_present)

    # print(help(results[0].stack))

    # start_time = all_meta.iloc[0]["stopTime"] - pd.Timedelta(days=24)
    # end_time = all_meta.iloc[0]["stopTime"] + pd.Timedelta(days=24)


    # print(results[0].stack(opts=asf_search.ASFSearchOptions(start=start_time.isoformat().replace("+00:00", "Z"), end=end_time.isoformat().replace("+00:00", "Z"))))

    # print(result[0].stack)
    

if __name__ == "__main__":
    main()