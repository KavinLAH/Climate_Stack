# PestCast Integration Notes

## For Person 3 (Backend)

**POST /predict**
- Endpoint: `http://<host>:<port>/predict`
- Content-Type: `application/json`
- Body schema:
  - `sightings`: list of `[lat, lon, timestamp]`
  - `climate_data`: GeoJSON FeatureCollection from the climate pipeline

Example Node/Supabase request (pseudo-code):

```ts
const payload = {
  sightings: [
    [36.7783, -119.4179, "2026-02-13T18:04:00Z"],
    [36.7801, -119.4102, "2026-02-13T19:12:00Z"]
  ],
  climate_data
};

const res = await fetch("http://api-host/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(payload)
});

const polygon = await res.json();
```

**GET /health**
- Endpoint: `http://<host>:<port>/health`
- Returns: `{ "status": "ok" }`

## For Person 4 (Frontend)

The /predict response is a GeoJSON **Feature** with a Polygon geometry. Bind to these properties:

- `center`: `[lon, lat]`
- `bearing`: degrees (0-360)
- `base_radius_km`: number
- `max_radius_km`: number
- `avg_temp_c`: number or null
- `wind_speed_kmh`: number or null
- `wind_dir`: integer (0-360) or null

You can rotate Mapbox layers using `bearing` and adjust styling based on `avg_temp_c` and `wind_speed_kmh`.
