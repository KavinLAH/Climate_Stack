-- Enable Row Level Security
ALTER TABLE public.sightings ENABLE ROW LEVEL SECURITY;

-- Public read access (anyone can query sightings)
CREATE POLICY "Public read access"
  ON public.sightings
  FOR SELECT
  USING (true);

-- Public insert access (anyone can submit sightings)
CREATE POLICY "Public insert access"
  ON public.sightings
  FOR INSERT
  WITH CHECK (true);

-- Public update access (for syncing status updates)
CREATE POLICY "Public update access"
  ON public.sightings
  FOR UPDATE
  USING (true);
