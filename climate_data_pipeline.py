"""Climate Data Pipeline for PestCast.

Fetches forecast and historical weather data from Open-Meteo for the
California Central Valley and converts it to GeoJSON FeatureCollection.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Tuple

import requests


OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

CENTRAL_VALLEY_BOUNDS = {
    "lat_min": 36.0,
    "lat_max": 38.5,
    "lon_min": -121.0,
    "lon_max": -119.0,
}

DEMO_HIGH_RISK_THRESHOLDS = {
    "humidity": 60.0,
    "temperature": 20.0,
}

DEMO_HOTSPOTS: List[Tuple[float, float]] = [
    (36.7783, -119.4179),  # Fresno
    (37.3382, -121.8863),  # San Jose
    (37.6391, -120.9969),  # Modesto
    (36.6067, -121.5545),  # Salinas
    (37.9577, -121.2908),  # Stockton
]

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
        "hourly": ",".join(variables),
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


def _is_demo_hotspot(latitude: float, longitude: float) -> bool:
    for hotspot_lat, hotspot_lon in DEMO_HOTSPOTS:
        if abs(latitude - hotspot_lat) <= 0.15 and abs(longitude - hotspot_lon) <= 0.15:
            return True
    return False


def _inject_demo_high_risk(
    latitude: float,
    longitude: float,
    properties: Dict[str, Any],
) -> Dict[str, Any]:
    if not _is_demo_hotspot(latitude, longitude):
        return properties

    temp_key = "temp"
    humidity_key = "humidity"

    temp_value = properties.get(temp_key)
    humidity_value = properties.get(humidity_key)

    if temp_value is None or temp_value < DEMO_HIGH_RISK_THRESHOLDS["temperature"]:
        properties[temp_key] = max(
            DEMO_HIGH_RISK_THRESHOLDS["temperature"],
            24.0,
        )

    if (
        humidity_value is None
        or humidity_value < DEMO_HIGH_RISK_THRESHOLDS["humidity"]
    ):
        properties[humidity_key] = max(
            DEMO_HIGH_RISK_THRESHOLDS["humidity"],
            70.0,
        )

    properties["demo_high_risk"] = True
    properties["demo_pest"] = "aphids_or_fall_armyworm"
    return properties


def to_geojson_feature_collection(
    latitude: float,
    longitude: float,
    weather_payload: Dict[str, Any],
    demo_mode: bool = False,
) -> Dict[str, Any]:
    """Convert Open-Meteo response to GeoJSON FeatureCollection."""
    current = weather_payload.get("current", {})
    historical = weather_payload.get("historical", {}).get("hourly", {})

    properties = {}
    for prop_key, api_key in SCHEMA_DICTIONARY.items():
        value = current.get(api_key)
        if api_key == "wind_direction_10m":
            properties[prop_key] = _normalize_wind_direction(value)
        else:
            properties[prop_key] = value

    properties["historical_hourly"] = historical

    if demo_mode:
        properties = _inject_demo_high_risk(latitude, longitude, properties)

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


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    raise ValueError("Timestamp must be an ISO-8601 string or datetime.")


def _bearing_degrees(
    start_lat: float, start_lon: float, end_lat: float, end_lon: float
) -> float:
    """Calculate bearing in degrees from start to end."""
    lat1 = math.radians(start_lat)
    lat2 = math.radians(end_lat)
    delta_lon = math.radians(end_lon - start_lon)

    x = math.sin(delta_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(
        delta_lon
    )
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360.0) % 360.0


def _destination_point(
    start_lat: float, start_lon: float, bearing_deg: float, distance_km: float
) -> Tuple[float, float]:
    """Return lat/lon reached by moving distance along bearing on a sphere."""
    radius_km = 6371.0
    bearing = math.radians(bearing_deg)
    lat1 = math.radians(start_lat)
    lon1 = math.radians(start_lon)
    angular_distance = distance_km / radius_km

    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular_distance)
        + math.cos(lat1) * math.sin(angular_distance) * math.cos(bearing)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(angular_distance) * math.cos(lat1),
        math.cos(angular_distance) - math.sin(lat1) * math.sin(lat2),
    )

    return (math.degrees(lat2), (math.degrees(lon2) + 540.0) % 360.0 - 180.0)


def _haversine_km(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    radius_km = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    )
    return 2 * radius_km * math.asin(math.sqrt(a))


def _unit_vector_from_bearing(bearing_deg: float) -> Tuple[float, float]:
    rad = math.radians(bearing_deg)
    return (math.sin(rad), math.cos(rad))


def _bearing_from_unit_vector(x: float, y: float) -> float:
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360.0) % 360.0


def _extract_climate_metrics(
    climate_geojson: Dict[str, Any],
    target_lat: float,
    target_lon: float,
) -> Tuple[float | None, int | None, float | None]:
    features = climate_geojson.get("features", [])
    if not isinstance(features, list) or not features:
        return (None, None, None, None)

    nearest = None
    nearest_dist = float("inf")
    local_temps: List[float] = []

    for feature in features:
        geometry = feature.get("geometry", {})
        coords = geometry.get("coordinates", [])
        if not isinstance(coords, list) or len(coords) != 2:
            continue
        lon, lat = coords
        distance = _haversine_km(target_lat, target_lon, lat, lon)
        if distance < nearest_dist:
            nearest_dist = distance
            nearest = feature

        if distance <= 25.0:
            temp_value = feature.get("properties", {}).get("temp")
            if isinstance(temp_value, (int, float)):
                local_temps.append(float(temp_value))

    if not local_temps:
        for feature in features:
            temp_value = feature.get("properties", {}).get("temp")
            if isinstance(temp_value, (int, float)):
                local_temps.append(float(temp_value))

    avg_temp = sum(local_temps) / len(local_temps) if local_temps else None

    wind_speed = None
    wind_dir = None
    if nearest:
        props = nearest.get("properties", {})
        wind_speed = props.get("wind_speed")
        wind_dir = props.get("wind_dir")
        if isinstance(wind_speed, (int, float)):
            wind_speed = float(wind_speed)
        else:
            wind_speed = None
        if isinstance(wind_dir, (int, float)):
            wind_dir = int(wind_dir) % 360
        else:
            wind_dir = None

    return (wind_speed, wind_dir, avg_temp)


def predict_pest_expansion(
    cluster_data: List[Tuple[float, float, Any]],
    climate_geojson: Dict[str, Any],
) -> Dict[str, Any]:
    """Predict expansion as a GeoJSON Polygon cone.

    Scientific rationale:
    - We blend movement direction with wind because lightweight pests drift with
      airflow; 30% weighting reflects a moderate, not dominant, wind influence.
    - We cap spread in cold conditions (<15°C) since metabolic rates and flight
      activity typically slow, reducing range.
    """
    if not cluster_data:
        raise ValueError("cluster_data must contain at least one sighting.")

    sightings = sorted(
        [(lat, lon, _parse_timestamp(ts)) for lat, lon, ts in cluster_data],
        key=lambda item: item[2],
    )

    latest_lat, latest_lon, _latest_ts = sightings[-1]

    if len(sightings) >= 2:
        prev_lat, prev_lon, _prev_ts = sightings[-2]
        movement_bearing = _bearing_degrees(prev_lat, prev_lon, latest_lat, latest_lon)
    else:
        movement_bearing = None

    wind_speed, wind_dir, avg_temp = _extract_climate_metrics(
        climate_geojson, latest_lat, latest_lon
    )

    if movement_bearing is None and wind_dir is None:
        movement_bearing = 0.0
    elif movement_bearing is None:
        movement_bearing = float(wind_dir)

    final_bearing = float(movement_bearing)

    if wind_speed is not None and wind_speed > 10.0 and wind_dir is not None:
        wind_weight = 0.30
        move_x, move_y = _unit_vector_from_bearing(final_bearing)
        wind_x, wind_y = _unit_vector_from_bearing(float(wind_dir))
        blended_x = (1.0 - wind_weight) * move_x + wind_weight * wind_x
        blended_y = (1.0 - wind_weight) * move_y + wind_weight * wind_y
        final_bearing = _bearing_from_unit_vector(blended_x, blended_y)

    base_radius_km = 5.0
    max_radius_km = 50.0
    if avg_temp is not None and avg_temp < 15.0:
        max_radius_km = min(max_radius_km, 25.0)

    cone_half_angle = 20.0
    left_bearing = (final_bearing - cone_half_angle) % 360
    right_bearing = (final_bearing + cone_half_angle) % 360

    def _angle_delta(start: float, end: float) -> float:
        """Smallest positive rotation from start to end."""
        return (end - start + 360.0) % 360.0

    arc_points: List[Tuple[float, float]] = []
    delta = _angle_delta(left_bearing, right_bearing)
    for step in range(0, 11):
        bearing = (left_bearing + delta * (step / 10)) % 360.0
        lat, lon = _destination_point(latest_lat, latest_lon, bearing, base_radius_km)
        arc_points.append((lon, lat))

    right_far_lat, right_far_lon = _destination_point(
        latest_lat, latest_lon, right_bearing, max_radius_km
    )
    tip_lat, tip_lon = _destination_point(
        latest_lat, latest_lon, final_bearing, max_radius_km
    )
    left_far_lat, left_far_lon = _destination_point(
        latest_lat, latest_lon, left_bearing, max_radius_km
    )

    polygon_coords = (
        arc_points
        + [(right_far_lon, right_far_lat), (tip_lon, tip_lat), (left_far_lon, left_far_lat)]
        + [arc_points[0]]
    )

    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [polygon_coords],
        },
        "properties": {
            "center": [latest_lon, latest_lat],
            "bearing": round(final_bearing, 2),
            "base_radius_km": base_radius_km,
            "max_radius_km": max_radius_km,
            "avg_temp_c": None if avg_temp is None else round(avg_temp, 2),
            "wind_speed_kmh": wind_speed,
            "wind_dir": wind_dir,
        },
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
    demo_mode: bool = False,
) -> Dict[str, Any]:
    """Fetch weather for each grid point and return FeatureCollection."""
    variables = list(SCHEMA_DICTIONARY.values())

    features = []
    for point in build_grid(lat_min, lat_max, lon_min, lon_max, step):
        weather = fetch_weather_data(point["latitude"], point["longitude"], variables)
        fc = to_geojson_feature_collection(
            point["latitude"], point["longitude"], weather, demo_mode=demo_mode
        )
        features.extend(fc["features"])

    return {"type": "FeatureCollection", "features": features}


def save_geojson(feature_collection: Dict[str, Any], output_path: str) -> None:
    """Save FeatureCollection to a file."""
    validate_geojson(feature_collection)
    with open(output_path, "w", encoding="utf-8") as file_handle:
        json.dump(feature_collection, file_handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="PestCast Climate Data Pipeline")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Inject high-risk climate markers at Central Valley hotspots.",
    )
    parser.add_argument(
        "--output",
        default="pestcast_weather.geojson",
        help="Output path for the GeoJSON FeatureCollection.",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.25,
        help="Grid step size in degrees for the Central Valley bounds.",
    )
    args = parser.parse_args()

    feature_collection = build_geojson_for_grid(
        lat_min=CENTRAL_VALLEY_BOUNDS["lat_min"],
        lat_max=CENTRAL_VALLEY_BOUNDS["lat_max"],
        lon_min=CENTRAL_VALLEY_BOUNDS["lon_min"],
        lon_max=CENTRAL_VALLEY_BOUNDS["lon_max"],
        step=args.step,
        demo_mode=args.demo,
    )
    save_geojson(feature_collection, args.output)


if __name__ == "__main__":
    main()
