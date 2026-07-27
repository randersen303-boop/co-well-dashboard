"""
Exports co_well_data.sqlite into the static JSON files the dashboard reads.

Scoped to the DJ Basin only (see DJ_BASIN_COUNTIES below) -- all three outputs are
filtered to that set of wells.

Three kinds of output, all under dashboard/data/:
  - wells.json: one compact record per well (header info + location + HZ flag) for
    every well with a lat/long. Loaded once by the map on page load.
  - production/{bucket}.json: monthly oil/gas/water totals per well, summed across
    formations and sidetracks, sharded into N_BUCKETS files by api number (int(api) %
    N_BUCKETS) so the browser only ever fetches the one slice it needs, lazily, on
    first click of a well in that bucket -- instead of loading all 15M+ production
    rows up front. Sharding by county first (i.e. one file per county) was tried and
    rejected: Weld County alone produced a 116MB shard since it dominates CO oil & gas
    activity, which would make clicking any Wattenberg-field well painfully slow.
    Hashing by API number instead spreads every county evenly across all shards.
  - directional_lines.json: the actual wellbore/lateral survey path for every
    directional or horizontal well, reprojected from ECMC's NAD83 UTM Zone 13N to
    lat/lon and decimated to a max point count (some raw surveys have 700+ vertices,
    which is far more detail than useful at map scale). This isn't in co_well_data.sqlite
    at all -- it's fetched directly from ECMC's DIRECTIONAL_LINES_SHP.ZIP here, since the
    geometry is only needed for this dashboard, not the rest of the pipeline. Loaded
    once, shown as a map layer (not lazily per-click), matching how ECMC's own GIS
    viewer displays wellbores.

Run this after co_well_data.py update/backfill to refresh the dashboard's data.
"""

import json
import shutil
import sqlite3
import zipfile
from pathlib import Path

import requests
import shapefile
from pyproj import Transformer

DB_PATH = Path(__file__).resolve().parent.parent / "co_well_data.sqlite"
RAW_DIR = Path(__file__).resolve().parent.parent / "raw"
DATA_DIR = Path(__file__).resolve().parent / "data"
N_BUCKETS = 128

DIRECTIONAL_LINES_URL = "https://ecmc.state.co.us/documents/data/downloads/gis/DIRECTIONAL_LINES_SHP.ZIP"
MAX_LINE_POINTS = 40
HEADERS = {"User-Agent": "Mozilla/5.0 (Horizon Resources data pipeline; contact: randersen303@gmail.com)"}

# Standard industry/EIA definition of the DJ (Denver-Julesburg) Basin's Colorado
# footprint -- county-based, since ECMC's own "Basin" label (shown on scout card pages)
# isn't in any bulk download. Weld County alone (Wattenberg field) is ~60% of these wells.
# ECMC's api_county codes, from https://ecmc.state.co.us/documents/about/COGIS_Help/API_County_codes.pdf
DJ_BASIN_COUNTIES = {
    "123",  # WELD
    "001",  # ADAMS
    "005",  # ARAPAHOE
    "013",  # BOULDER
    "014",  # BROOMFIELD
#    "031",  # DENVER
    "035",  # DOUGLAS
    "039",  # ELBERT
    "069",  # LARIMER
#    "075",  # LOGAN
#    "087",  # MORGAN
#    "095",  # PHILLIPS
#    "115",  # SEDGWICK
#    "121",  # WASHINGTON
#    "125",  # YUMA
}

WELL_COLUMNS = [
    "api", "well_name", "well_num", "operator", "field_name", "well_class",
    "facility_status", "spud_date", "status_date", "latitude", "longitude", "max_md", "max_tvd", "hz",
]


def export_wells(conn: sqlite3.Connection) -> set:
    # One directional record per well api -- prefer sidetrack '00' (the primary
    # wellbore) when present, otherwise whichever sidetrack has a survey on file.
    cur = conn.execute(
        """
        SELECT api, deviation FROM directional
        WHERE (api, api_sidetrack) IN (
            SELECT api, MIN(api_sidetrack) FROM directional GROUP BY api
        )
        """
    )
    hz_by_api = {api: deviation for api, deviation in cur.fetchall()}

    placeholders = ",".join("?" * len(DJ_BASIN_COUNTIES))
    cur = conn.execute(
        f"""
        SELECT api, well_name, well_num, operator, field_name, well_class,
               facility_status, spud_date, status_date, latitude, longitude, max_md, max_tvd
        FROM wells
        WHERE latitude IS NOT NULL AND longitude IS NOT NULL AND api_county IN ({placeholders})
        """,
        tuple(DJ_BASIN_COUNTIES),
    )
    rows = []
    dj_apis = set()
    for r in cur.fetchall():
        api = r[0]
        dj_apis.add(api)
        rows.append(list(r) + [hz_by_api.get(api)])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / "wells.json", "w") as f:
        json.dump({"columns": WELL_COLUMNS, "wells": rows}, f, separators=(",", ":"))
    print(f"Exported {len(rows):,} DJ Basin wells to wells.json")
    return dj_apis


def export_production(conn: sqlite3.Connection, dj_apis: set) -> tuple[int, int]:
    prod_dir = DATA_DIR / "production"
    prod_dir.mkdir(parents=True, exist_ok=True)

    cur = conn.execute(
        """
        SELECT api, report_year, report_month,
               SUM(oil_produced), SUM(gas_produced), SUM(water_produced)
        FROM production
        GROUP BY api, report_year, report_month
        ORDER BY api, report_year, report_month
        """
    )
    buckets = [{} for _ in range(N_BUCKETS)]
    total_wells = set()
    for api, year, month, oil, gas, water in cur.fetchall():
        if api not in dj_apis:
            continue
        bucket = buckets[int(api) % N_BUCKETS]
        bucket.setdefault(api, []).append([
            year, month,
            round(oil, 1) if oil else 0,
            round(gas, 1) if gas else 0,
            round(water, 1) if water else 0,
        ])
        total_wells.add(api)

    for i, bucket in enumerate(buckets):
        with open(prod_dir / f"{i}.json", "w") as f:
            json.dump(bucket, f, separators=(",", ":"))

    return N_BUCKETS, len(total_wells)


def _decimate(points, max_points):
    """Keeps first + last + an even subset of interior points -- a cheap stand-in for
    proper polyline simplification (e.g. Douglas-Peucker), good enough at map scale."""
    if len(points) <= max_points:
        return points
    step = (len(points) - 1) / (max_points - 1)
    idxs = sorted({round(i * step) for i in range(max_points)})
    return [points[i] for i in idxs]


def export_directional_lines(dj_apis: set) -> int:
    dest = RAW_DIR / "DIRECTIONAL_LINES_SHP.zip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(DIRECTIONAL_LINES_URL, headers=HEADERS, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)

    extract_dir = RAW_DIR / "directional_lines_shp"
    with zipfile.ZipFile(dest) as z:
        z.extractall(extract_dir)
    shp_path = next(extract_dir.glob("*.shp"))

    # ECMC's directional shapefiles are NAD83 / UTM Zone 13N (confirmed via the .prj).
    to_wgs84 = Transformer.from_crs("EPSG:26913", "EPSG:4326", always_xy=True)

    lines = {}
    with shapefile.Reader(str(shp_path)) as sf:
        field_names = [f[0] for f in sf.fields[1:]]
        for sr in sf.iterShapeRecords():
            d = dict(zip(field_names, sr.record))
            raw_api = d.get("API")
            if not raw_api or len(raw_api) < 8:
                continue
            api = raw_api[:8]
            if api not in dj_apis:
                continue
            points = [to_wgs84.transform(x, y) for x, y in sr.shape.points]
            points = [[round(lat, 5), round(lon, 5)] for lon, lat in points]
            points = _decimate(points, MAX_LINE_POINTS)
            lines.setdefault(api, {"deviation": d.get("Deviation"), "paths": []})["paths"].append(points)

    with open(DATA_DIR / "directional_lines.json", "w") as f:
        json.dump(lines, f, separators=(",", ":"))

    shutil.rmtree(extract_dir, ignore_errors=True)
    return len(lines)


def main():
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    dj_apis = export_wells(conn)
    n_buckets, n_prod_wells = export_production(conn, dj_apis)
    print(f"Exported production for {n_prod_wells:,} wells across {n_buckets} shards")
    conn.close()

    n_lines = export_directional_lines(dj_apis)
    print(f"Exported {n_lines:,} directional wellbore paths to directional_lines.json")


if __name__ == "__main__":
    main()
