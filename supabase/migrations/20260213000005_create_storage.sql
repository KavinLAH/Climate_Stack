-- Create storage bucket for pest sighting images
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
  'pest-images',
  'pest-images',
  true,
  5242880,  -- 5MB limit
  ARRAY['image/jpeg', 'image/png', 'image/webp', 'image/heic']
);

-- Allow public uploads to pest-images bucket
CREATE POLICY "Public upload pest images"
  ON storage.objects
  FOR INSERT
  WITH CHECK (bucket_id = 'pest-images');

-- Allow public read of pest images
CREATE POLICY "Public read pest images"
  ON storage.objects
  FOR SELECT
  USING (bucket_id = 'pest-images');
