-- Farmers table to track farm locations
CREATE TABLE public.farmers (
  id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name       TEXT NOT NULL,
  latitude   FLOAT NOT NULL,
  longitude  FLOAT NOT NULL,
  device_id  TEXT UNIQUE NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_farmers_device_id ON public.farmers (device_id);
CREATE INDEX idx_farmers_location ON public.farmers (latitude, longitude);

-- RLS: open for hackathon demo
ALTER TABLE public.farmers ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Public read access" ON public.farmers FOR SELECT USING (true);
CREATE POLICY "Public insert access" ON public.farmers FOR INSERT WITH CHECK (true);
CREATE POLICY "Public update access" ON public.farmers FOR UPDATE USING (true);

-- RPC: get risk level for a farmer based on proximity to recent sightings
-- Returns nearby sightings within a radius (km), grouped by species with risk score
CREATE OR REPLACE FUNCTION get_farmer_risk(
  farm_lat FLOAT,
  farm_lng FLOAT,
  radius_km FLOAT DEFAULT 50,
  hours_back INT DEFAULT 168
)
RETURNS TABLE (
  species TEXT,
  sighting_count BIGINT,
  avg_severity FLOAT,
  nearest_km FLOAT,
  risk_score FLOAT
) AS $$
  WITH nearby AS (
    SELECT
      s.species,
      s.severity,
      -- Haversine distance in km
      (
        6371 * acos(
          LEAST(1.0,
            cos(radians(farm_lat)) * cos(radians(s.latitude))
            * cos(radians(s.longitude) - radians(farm_lng))
            + sin(radians(farm_lat)) * sin(radians(s.latitude))
          )
        )
      ) AS distance_km
    FROM public.sightings s
    WHERE s.created_at >= now() - (hours_back || ' hours')::interval
  )
  SELECT
    n.species,
    COUNT(*) AS sighting_count,
    AVG(CASE n.severity
      WHEN 'low' THEN 1
      WHEN 'medium' THEN 2
      WHEN 'high' THEN 3
      WHEN 'critical' THEN 4
    END) AS avg_severity,
    MIN(n.distance_km) AS nearest_km,
    -- Risk score: higher when more sightings, higher severity, closer distance
    LEAST(100,
      (COUNT(*)::FLOAT / 5.0)
      * AVG(CASE n.severity
          WHEN 'low' THEN 1 WHEN 'medium' THEN 2
          WHEN 'high' THEN 3 WHEN 'critical' THEN 4
        END)
      * (radius_km / GREATEST(MIN(n.distance_km), 0.1))
    ) AS risk_score
  FROM nearby n
  WHERE n.distance_km <= radius_km
  GROUP BY n.species
  ORDER BY risk_score DESC;
$$ LANGUAGE sql STABLE;
