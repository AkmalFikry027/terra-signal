"""Download NASA POWER monthly data for a global 5-degree grid."""
from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "history" / "nasa_power_global_monthly_2011_2025.json"
CHECKPOINT = ROOT / "data" / "history" / "nasa_power_download_checkpoint.json"
LAND_MASK = ROOT / "data" / "history" / "natural_earth_110m_land.geojson"
LAND_MASK_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_land.geojson"
COUNTRY_BOUNDARIES = ROOT / "data" / "history" / "natural_earth_110m_countries.geojson"
COUNTRY_BOUNDARIES_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
ADMIN1_BOUNDARIES = ROOT / "data" / "history" / "natural_earth_10m_admin_1.geojson.gz"
ADMIN1_BOUNDARIES_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_1_states_provinces.geojson"
PARAMETERS = ("T2M", "PRECTOTCORR", "GWETROOT")


def grid_points():
    for lat_index in range(36):
        lat = -87.5 + lat_index * 5
        for lon_index in range(72):
            lon = -177.5 + lon_index * 5
            yield {
                "id": f"G5-L{int(abs(lat) * 10):04d}{'N' if lat >= 0 else 'S'}"
                      f"-O{int(abs(lon) * 10):04d}{'E' if lon >= 0 else 'W'}",
                "name": f"Grid cell {abs(lat):g}°{'N' if lat >= 0 else 'S'}, "
                        f"{abs(lon):g}°{'E' if lon >= 0 else 'W'}",
                "lat": round(lat, 1),
                "lon": round(lon, 1),
            }


def _inside_ring(lon, lat, ring):
    inside = False
    previous_lon, previous_lat = ring[-1]
    for current_lon, current_lat in ring:
        if (current_lat > lat) != (previous_lat > lat):
            crossing_lon = (previous_lon - current_lon) * (lat - current_lat) / \
                           (previous_lat - current_lat) + current_lon
            if lon < crossing_lon:
                inside = not inside
        previous_lon, previous_lat = current_lon, current_lat
    return inside


def _on_land(lon, lat, geometry):
    polygons = geometry["coordinates"]
    if geometry["type"] == "Polygon":
        polygons = [polygons]
    for polygon in polygons:
        if _inside_ring(lon, lat, polygon[0]) and not any(
                _inside_ring(lon, lat, hole) for hole in polygon[1:]):
            return True
    return False


def land_grid_points():
    if not LAND_MASK.exists():
        LAND_MASK.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(LAND_MASK_URL, headers={"User-Agent": "SAT-SCOPE/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            LAND_MASK.write_bytes(response.read())
    geojson = json.loads(LAND_MASK.read_text(encoding="utf-8"))
    geometries = [feature["geometry"] for feature in geojson["features"]]
    return [cell for cell in grid_points()
            if any(_on_land(cell["lon"], cell["lat"], geometry) for geometry in geometries)]


def country_for_point(lon, lat, features):
    for feature in features:
        if _on_land(lon, lat, feature["geometry"]):
            return feature["properties"].get("ADMIN")
    return None


def country_features():
    if not COUNTRY_BOUNDARIES.exists():
        COUNTRY_BOUNDARIES.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(COUNTRY_BOUNDARIES_URL, headers={"User-Agent": "SAT-SCOPE/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            COUNTRY_BOUNDARIES.write_bytes(response.read())
    return json.loads(COUNTRY_BOUNDARIES.read_text(encoding="utf-8"))["features"]


def admin1_features():
    if not ADMIN1_BOUNDARIES.exists():
        ADMIN1_BOUNDARIES.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(ADMIN1_BOUNDARIES_URL, headers={"User-Agent": "SAT-SCOPE/1.0"})
        with urllib.request.urlopen(request, timeout=180) as response, gzip.open(
                ADMIN1_BOUNDARIES, "wb", compresslevel=6) as compressed:
            shutil.copyfileobj(response, compressed)
    with gzip.open(ADMIN1_BOUNDARIES, "rt", encoding="utf-8") as stream:
        return json.load(stream)["features"]


def area_for_point(lon, lat, features):
    for feature in features:
        if _on_land(lon, lat, feature["geometry"]):
            properties = feature["properties"]
            return (properties.get("name_en") or properties.get("name"),
                    properties.get("type_en") or properties.get("type") or "Administrative area")
    return None, None


def parse_response(payload, cell, start_year, end_year):
    parameters = payload.get("properties", {}).get("parameter", {})
    values = {name: parameters.get(name, {}) for name in PARAMETERS}
    series = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            key = f"{year}{month:02d}"
            row = {}
            for name in PARAMETERS:
                value = values[name].get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= -900:
                    return None
                row[name] = round(float(value), 4)
            series.append({
                "date": f"{year}-{month:02d}",
                "temperature_c": row["T2M"],
                "precipitation_mm_day": row["PRECTOTCORR"],
                "root_zone_soil_wetness": row["GWETROOT"],
            })
    if len(series) != (end_year - start_year + 1) * 12:
        return None
    return {**cell, "series": series}


def _request(cell, start_year, end_year, retries=4):
    query = urllib.parse.urlencode({
        "parameters": ",".join(PARAMETERS),
        "community": "AG",
        "longitude": cell["lon"],
        "latitude": cell["lat"],
        "start": start_year,
        "end": end_year,
        "format": "JSON",
    })
    url = f"https://power.larc.nasa.gov/api/temporal/monthly/point?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "SAT-SCOPE/1.0"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
            sources = payload.get("header", {}).get("sources", [])
            if "MERRA2" not in sources:
                raise RuntimeError(f"Unexpected NASA POWER source for {cell['id']}: {sources}")
            return parse_response(payload, cell, start_year, end_year)
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == retries:
                raise
            try:
                delay = float(error.headers.get("Retry-After", min(2 ** (attempt + 2), 60)))
            except (TypeError, ValueError):
                delay = min(2 ** (attempt + 2), 60)
            time.sleep(delay + random.uniform(0, 1))
            continue
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt == retries:
                raise
            time.sleep(min(2 ** (attempt + 2), 60) + random.uniform(0, 1))
    raise RuntimeError(f"NASA POWER request failed for {cell['id']}")


def _save_checkpoint(results, start_year, end_year):
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    temporary = CHECKPOINT.with_suffix(".tmp")
    temporary.write_text(json.dumps({
        "start_year": start_year,
        "end_year": end_year,
        "results": results,
    }, separators=(",", ":")), encoding="utf-8")
    temporary.replace(CHECKPOINT)


def download(start_year=2011, end_year=2025, workers=8):
    if start_year > end_year or end_year - start_year + 1 < 2:
        raise ValueError("Choose an increasing year range of at least two years.")
    if not 1 <= workers <= 16:
        raise ValueError("workers must be between 1 and 16.")
    points = land_grid_points()
    results = {}
    if CHECKPOINT.exists():
        checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        if (checkpoint.get("start_year"), checkpoint.get("end_year")) == (start_year, end_year):
            results = checkpoint.get("results", {})

    pending = [cell for cell in points if cell["id"] not in results]
    failures = []
    print(f"NASA POWER: {len(points)} land grid points; "
          f"{len(results)} cached, {len(pending)} remaining.")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_request, cell, start_year, end_year): cell for cell in pending}
        for index, future in enumerate(as_completed(futures), start=1):
            cell = futures[future]
            try:
                results[cell["id"]] = future.result()
            except Exception as error:
                failures.append((cell["id"], str(error)))
            if index % 50 == 0:
                _save_checkpoint(results, start_year, end_year)
                print(f"Completed {index}/{len(pending)} requests.")
    _save_checkpoint(results, start_year, end_year)
    if failures:
        examples = "; ".join(f"{cell}: {error}" for cell, error in failures[:5])
        raise RuntimeError(f"{len(failures)} NASA POWER requests failed; rerun to retry. {examples}")

    countries = country_features()
    country_names = sorted({feature["properties"]["ADMIN"] for feature in countries})
    country_name_set = set(country_names)
    areas_by_country = {}
    area_catalog = {}
    for feature in admin1_features():
        properties = feature["properties"]
        country = properties.get("admin")
        name = properties.get("name_en") or properties.get("name")
        if country not in country_name_set or not name:
            continue
        area_catalog[(country, name)] = {
            "country": country, "name": name,
            "type": properties.get("type_en") or properties.get("type") or "Administrative area",
        }
        areas_by_country.setdefault(country, []).append(feature)

    valid_cells = []
    for cell in points:
        result = results.get(cell["id"])
        if not result:
            continue
        country = country_for_point(cell["lon"], cell["lat"], countries)
        if not country:
            raise RuntimeError(f"Unable to assign a country to land cell {cell['id']}.")
        area, area_type = area_for_point(cell["lon"], cell["lat"], areas_by_country.get(country, []))
        if not area:
            area = cell["name"]
            area_type = "NASA grid cell"
        area_catalog.setdefault((country, area), {
            "country": country, "name": area, "type": area_type,
        })
        valid_cells.append({**result, "country": country, "area": area})
    if not valid_cells:
        raise RuntimeError("NASA POWER returned no complete grid cells.")
    archive = {
        "title": "Global NASA POWER monthly environmental history",
        "source": "NASA POWER Monthly and Annual API; MERRA-2 meteorological reanalysis",
        "source_url": "https://power.larc.nasa.gov/",
        "land_mask_source": LAND_MASK_URL,
        "country_boundaries_source": COUNTRY_BOUNDARIES_URL,
        "administrative_areas_source": ADMIN1_BOUNDARIES_URL,
        "countries": country_names,
        "areas": sorted(area_catalog.values(), key=lambda area: (area["country"], area["name"])),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "years": list(range(start_year, end_year + 1)),
        "baseline_years": list(range(start_year, min(end_year, start_year + 9) + 1)),
        "grid_spacing_degrees": 5,
        "grid_method": "NASA POWER monthly point values sampled at 5-degree cell centers; not area-averaged",
        "land_mask": "Natural Earth 1:110m land polygons (public domain); used only to select land grid centers",
        "variables": {
            "T2M": {"name": "2 m air temperature", "unit": "°C"},
            "PRECTOTCORR": {"name": "Corrected precipitation rate", "unit": "mm/day"},
            "GWETROOT": {"name": "Root-zone soil wetness", "unit": "fraction"},
        },
        "cell_count": len(valid_cells),
        "cells": valid_cells,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(archive, stream, separators=(",", ":"), ensure_ascii=True)
    temporary.replace(OUT)
    CHECKPOINT.unlink(missing_ok=True)
    print(f"Wrote {len(valid_cells)} complete 5-degree land grid cells to {OUT}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2011)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    download(args.start_year, args.end_year, args.workers)


if __name__ == "__main__":
    main()
