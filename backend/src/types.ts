export type SeverityLevel = 'low' | 'medium' | 'high' | 'critical';

export interface Sighting {
  species: string;
  confidence: number;
  latitude: number;
  longitude: number;
  severity: SeverityLevel;
  image_url: string | null;
  device_id: string;
  created_at: string;
  synced: boolean;
}

export interface Farmer {
  name: string;
  latitude: number;
  longitude: number;
  device_id: string;
}
