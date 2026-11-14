import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from data_provider import GlobalPowerArchiveProvider  # noqa: E402
from download_nasa_power_history import parse_response  # noqa: E402
from ml.stats import soil_wetness_status  # noqa: E402
from services import dashboard as dashboard_service  # noqa: E402
from services.dashboard import _location_history  # noqa: E402


def test_global_archive_builds_monthly_baselines_and_area_catalog(tmp_path, monkeypatch):
    rows = []
    for year in (2011, 2012):
        for month in range(1, 13):
            rows.append({
                "date": f"{year}-{month:02d}",
                "temperature_c": 10 + month + (year - 2011),
                "precipitation_mm_day": 1 + month / 10,
                "root_zone_soil_wetness": 0.2 + month / 100,
            })
    archive_path = tmp_path / "history.json"
    archive_path.write_text(json.dumps({
        "years": [2011, 2012],
        "baseline_years": [2011],
        "grid_spacing_degrees": 5,
        "countries": ["Exampleland", "Otherland"],
        "areas": [
            {"country": "Exampleland", "name": "North", "type": "Province"},
            {"country": "Exampleland", "name": "South", "type": "Province"},
            {"country": "Otherland", "name": "East", "type": "Province"},
        ],
        "cells": [{
            "id": "G5-L0000N-O0000E",
            "name": "Grid cell 0°N, 0°E",
            "country": "Exampleland",
            "area": "North",
            "lat": 0,
            "lon": 0,
            "series": rows,
        }],
    }), encoding="utf-8")

    provider = GlobalPowerArchiveProvider(archive_path)
    bundle = provider.get()

    assert bundle["temperature"]["source_mode"] == "ARCHIVE"
    assert bundle["temperature"]["observations"][0]["current"] == 23
    assert bundle["temperature"]["observations"][0]["baseline"] == 22
    assert bundle["temperature"]["observations"][0]["region"] == "North, Exampleland"
    assert len(bundle["temperature"]["observations"][0]["history"]) == 24
    selected_history = _location_history(
        bundle, "temperature", [{"id": "G5-L0000N-O0000E", "lat": 0}], "current", "baseline"
    )
    assert len(selected_history) == 24
    assert selected_history[-1]["date"] == "2012-12-01"
    assert selected_history[-1]["current"] == 23
    assert bundle["vegetation"]["label"] == "Root-zone soil wetness"
    assert bundle["fires"]["source_mode"] == "UNAVAILABLE"

    monkeypatch.setattr(dashboard_service, "get_provider", lambda: provider)
    dashboard_service.reset_memo()
    locations = dashboard_service.list_locations()
    assert locations[0]["name"] == "Global"
    assert next(item for item in locations if item["name"] == "Exampleland")["cells"] == 1
    no_data_area = next(item for item in locations if item["name"] == "South")
    assert no_data_area["cells"] == 0
    with pytest.raises(dashboard_service.LocationDataUnavailable):
        dashboard_service._resolve(dashboard_service._get_context(), no_data_area["id"])
    dashboard_service.reset_memo()


def test_power_parser_ignores_annual_slot_and_rejects_missing_months():
    parameters = {
        key: {f"2011{month:02d}": value for month in range(1, 13)}
        for key, value in (("T2M", 20), ("PRECTOTCORR", 2), ("GWETROOT", 0.4))
    }
    for values in parameters.values():
        values["201113"] = 999
    payload = {"properties": {"parameter": parameters}}
    cell = {"id": "cell", "name": "cell", "lat": 0, "lon": 0}

    parsed = parse_response(payload, cell, 2011, 2011)

    assert len(parsed["series"]) == 12
    assert parsed["series"][-1]["date"] == "2011-12"
    del parameters["GWETROOT"]["201106"]
    assert parse_response(payload, cell, 2011, 2011) is None


def test_root_zone_soil_wetness_uses_drying_status():
    assert soil_wetness_status(-16) == "DRYING"
    assert soil_wetness_status(-5) == "WATCH"
    assert soil_wetness_status(0) == "NORMAL"
