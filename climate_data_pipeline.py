"""Climate Data Pipeline for PestCast.

Fetches forecast and historical weather data from Open-Meteo for the
California Central Valley and converts it to GeoJSON FeatureCollection.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Any, Dict, Iterable, List

import requests


OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

SCHEMA_DICTIONARY = {
    "temp": "temperature_2m",
    "humidity": "relative_humidity_2m",
    "wind_speed": "wind_speed_10m",
    "wind_dir": "wind_direction_10m",
}


def _frange(start: float, stop: float, step: float) -> Iterable[float]:
    """Generate a range of floats inclusive of stop if it lands on the step."""
    value = start
    while value <= stop + 1e-9:
        yield round(value, 4)
        value += step


def build_grid(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    step: float,
) -> List[Dict[str, float]]:
    """Create a grid of points over the given bounding box."""
    points = []
    for lat in _frange(lat_min, lat_max, step):
        for lon in _frange(lon_min, lon_max, step):
            points.append({"latitude": lat, "longitude": lon})
    return points


def _date_range(days: int) -> Dict[str, str]:
    """Return date range dict for last N days ending yesterday."""
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=days - 1)
    return {"start_date": start.isoformat(), "end_date": end.isoformat()}


def fetch_weather_data(
    latitude: float,
    longitude: float,
    variables: List[str],
) -> Dict[str, Any]:
    """Fetch current forecast and last 7 days of historical data.

    Returns a dict with keys: "current" and "historical".
    """
    params_forecast = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ",".join(variables),
        "timezone": "UTC",
    }

    params_historical = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": ",".join(variables),
        "timezone": "UTC",
        **_date_range(7),
    }

    forecast_resp = requests.get(OPEN_METEO_URL, params=params_forecast, timeout=30)
    forecast_resp.raise_for_status()

    historical_resp = requests.get(OPEN_METEO_URL, params=params_historical, timeout=30)
    historical_resp.raise_for_status()

    return {
        "current": forecast_resp.json(),
        "historical": historical_resp.json(),
    }


def _normalize_wind_direction(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(round(float(value))) % 360
    except (TypeError, ValueError):
        return None


def to_geojson_feature_collection(
    latitude: float,
    longitude: float,
    weather_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Convert Open-Meteo response to GeoJSON FeatureCollection."""
    current = weather_payload.get("current", {})
    historical = weather_payload.get("historical", {}).get("daily", {})

    properties = {}
    for prop_key, api_key in SCHEMA_DICTIONARY.items():
        value = current.get(api_key)
        if api_key == "wind_direction_10m":
            properties[prop_key] = _normalize_wind_direction(value)
        else:
            properties[prop_key] = value

    properties["historical_daily"] = historical

    feature = {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [longitude, latitude],
        },
        "properties": properties,
    }

    return {
        "type": "FeatureCollection",
        "features": [feature],
    }


def validate_geojson(feature_collection: Dict[str, Any]) -> None:
    """Validate minimal GeoJSON structure and required properties."""
    if feature_collection.get("type") != "FeatureCollection":
        raise ValueError("GeoJSON must be a FeatureCollection.")

    features = feature_collection.get("features")
    if not isinstance(features, list):
        raise ValueError("GeoJSON features must be a list.")

    required_keys = set(SCHEMA_DICTIONARY.keys())

    for feature in features:
        if feature.get("type") != "Feature":
            raise ValueError("Each feature must have type 'Feature'.")

        geometry = feature.get("geometry", {})
        if geometry.get("type") != "Point":
            raise ValueError("Each feature geometry must be a Point.")

        coordinates = geometry.get("coordinates")
        if (
            not isinstance(coordinates, list)
            or len(coordinates) != 2
            or not all(isinstance(value, (int, float)) for value in coordinates)
        ):
            raise ValueError("Point coordinates must be [lon, lat] numbers.")

        properties = feature.get("properties", {})
        if not isinstance(properties, dict):
            raise ValueError("Feature properties must be a dict.")

        missing = required_keys.difference(properties.keys())
        if missing:
            raise ValueError(f"Missing required properties: {sorted(missing)}")


def build_geojson_for_grid(
    lat_min: float = 36.0,
    lat_max: float = 38.5,
    lon_min: float = -121.0,
    lon_max: float = -119.0,
    step: float = 0.25,
) -> Dict[str, Any]:
    """Fetch weather for each grid point and return FeatureCollection."""
    variables = list(SCHEMA_DICTIONARY.values())

    features = []
    for point in build_grid(lat_min, lat_max, lon_min, lon_max, step):
        weather = fetch_weather_data(point["latitude"], point["longitude"], variables)
        fc = to_geojson_feature_collection(
            point["latitude"], point["longitude"], weather
        )
        features.extend(fc["features"])

    return {"type": "FeatureCollection", "features": features}


def save_geojson(feature_collection: Dict[str, Any], output_path: str) -> None:
    """Save FeatureCollection to a file."""
    validate_geojson(feature_collection)
    with open(output_path, "w", encoding="utf-8") as file_handle:
        json.dump(feature_collection, file_handle, ensure_ascii=False, indent=2)


def main() -> None:
    feature_collection = build_geojson_for_grid()
    save_geojson(feature_collection, "pestcast_weather.geojson")


if __name__ == "__main__":
    main()
