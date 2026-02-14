-- Create severity enum
CREATE TYPE severity_level AS ENUM ('low', 'medium', 'high', 'critical');

-- Create sightings table
CREATE TABLE public.sightings (
  id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  species    TEXT NOT NULL,
  confidence FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
  latitude   FLOAT NOT NULL,
  longitude  FLOAT NOT NULL,
  severity   severity_level NOT NULL DEFAULT 'low',
  image_url  TEXT,
  device_id  TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  synced     BOOLEAN NOT NULL DEFAULT true
);

-- Indexes for common query patterns
CREATE INDEX idx_sightings_species ON public.sightings (species);
CREATE INDEX idx_sightings_created_at ON public.sightings (created_at DESC);
CREATE INDEX idx_sightings_species_created ON public.sightings (species, created_at);
CREATE INDEX idx_sightings_location ON public.sightings (latitude, longitude);
CREATE INDEX idx_sightings_severity ON public.sightings (severity);
CREATE INDEX idx_sightings_device_id ON public.sightings (device_id);

-- RPC function for distinct species with counts
CREATE OR REPLACE FUNCTION get_distinct_species()
RETURNS TABLE (species TEXT, count BIGINT) AS $$
  SELECT species, COUNT(*) AS count
  FROM public.sightings
  GROUP BY species
  ORDER BY count DESC;
$$ LANGUAGE sql STABLE;
