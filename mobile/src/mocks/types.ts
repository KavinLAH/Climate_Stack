export type SeverityLevel = 'low' | 'medium' | 'high' | 'critical';

export interface Sighting {
  id: string;
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
  id: string;
  name: string;
  latitude: number;
  longitude: number;
  device_id: string;
  created_at: string;
}

export interface FarmerRisk {
  species: string;
  sighting_count: number;
  avg_severity: number;
  nearest_km: number;
  risk_score: number;
}

export interface SpeciesCount {
  species: string;
  count: number;
}
