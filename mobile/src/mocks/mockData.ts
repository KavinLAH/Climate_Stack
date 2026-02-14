import type { Sighting, Farmer, FarmerRisk, SpeciesCount } from './types';

// Current user — Garcia Family Farm in Fresno (high risk area)
export const CURRENT_FARMER: Farmer = {
  id: 'f1a2b3c4-d5e6-7890-abcd-ef1234567890',
  name: 'Garcia Family Farm',
  latitude: 36.7300,
  longitude: -119.7900,
  device_id: 'mobile-user-01',
  created_at: '2026-02-03T08:00:00.000Z',
};

export const ALL_FARMERS: Farmer[] = [
  CURRENT_FARMER,
  {
    id: 'a2b3c4d5-e6f7-8901-bcde-f12345678901',
    name: 'Johnson Ranch',
    latitude: 35.4200,
    longitude: -119.0300,
    device_id: 'mobile-user-02',
    created_at: '2026-02-01T10:00:00.000Z',
  },
  {
    id: 'b3c4d5e6-f7a8-9012-cdef-123456789012',
    name: 'Chen Orchards',
    latitude: 38.1000,
    longitude: -121.6800,
    device_id: 'mobile-user-03',
    created_at: '2026-02-02T09:00:00.000Z',
  },
  {
    id: 'c4d5e6f7-a8b9-0123-def0-234567890123',
    name: 'Patel Vineyards',
    latitude: 36.3300,
    longitude: -119.3000,
    device_id: 'field-scout-01',
    created_at: '2026-02-01T07:00:00.000Z',
  },
  {
    id: 'd5e6f7a8-b9c0-1234-ef01-345678901234',
    name: 'Williams Dairy & Crops',
    latitude: 37.6400,
    longitude: -121.0000,
    device_id: 'field-scout-02',
    created_at: '2026-02-03T06:00:00.000Z',
  },
  {
    id: 'e6f7a8b9-c0d1-2345-f012-456789012345',
    name: 'Nguyen Vegetable Farm',
    latitude: 38.5500,
    longitude: -121.7400,
    device_id: 'field-scout-03',
    created_at: '2026-02-02T11:00:00.000Z',
  },
  {
    id: 'f7a8b9c0-d1e2-3456-0123-567890123456',
    name: 'Rodriguez Almond Farm',
    latitude: 36.1000,
    longitude: -119.0200,
    device_id: 'drone-cam-01',
    created_at: '2026-02-01T14:00:00.000Z',
  },
  {
    id: 'a8b9c0d1-e2f3-4567-1234-678901234567',
    name: 'Kim Berry Farm',
    latitude: 37.3000,
    longitude: -120.4800,
    device_id: 'drone-cam-02',
    created_at: '2026-02-02T15:00:00.000Z',
  },
];

// Recent sightings — mix of species, severities, and distances from current farmer
export const MOCK_SIGHTINGS: Sighting[] = [
  // Fall Armyworm — near Fresno (close to current farmer = high urgency)
  {
    id: '11111111-1111-1111-1111-111111111101',
    species: 'Fall Armyworm',
    confidence: 0.94,
    latitude: 36.7520,
    longitude: -119.7650,
    severity: 'critical',
    image_url: null,
    device_id: 'field-scout-01',
    created_at: '2026-02-13T06:30:00.000Z',
    synced: true,
  },
  {
    id: '11111111-1111-1111-1111-111111111102',
    species: 'Fall Armyworm',
    confidence: 0.91,
    latitude: 36.7100,
    longitude: -119.8100,
    severity: 'critical',
    image_url: null,
    device_id: 'drone-cam-01',
    created_at: '2026-02-13T05:15:00.000Z',
    synced: true,
  },
  {
    id: '11111111-1111-1111-1111-111111111103',
    species: 'Fall Armyworm',
    confidence: 0.88,
    latitude: 36.6800,
    longitude: -119.7400,
    severity: 'high',
    image_url: null,
    device_id: 'field-scout-02',
    created_at: '2026-02-12T18:00:00.000Z',
    synced: true,
  },
  {
    id: '11111111-1111-1111-1111-111111111104',
    species: 'Fall Armyworm',
    confidence: 0.85,
    latitude: 36.6500,
    longitude: -119.5000,
    severity: 'high',
    image_url: null,
    device_id: 'mobile-user-02',
    created_at: '2026-02-12T14:30:00.000Z',
    synced: true,
  },
  {
    id: '11111111-1111-1111-1111-111111111105',
    species: 'Fall Armyworm',
    confidence: 0.82,
    latitude: 36.5800,
    longitude: -119.4200,
    severity: 'high',
    image_url: null,
    device_id: 'trap-sensor-01',
    created_at: '2026-02-12T10:00:00.000Z',
    synced: true,
  },
  // Fall Armyworm — further south along migration path
  {
    id: '11111111-1111-1111-1111-111111111106',
    species: 'Fall Armyworm',
    confidence: 0.79,
    latitude: 36.3300,
    longitude: -119.2900,
    severity: 'high',
    image_url: null,
    device_id: 'field-scout-01',
    created_at: '2026-02-11T16:00:00.000Z',
    synced: true,
  },
  {
    id: '11111111-1111-1111-1111-111111111107',
    species: 'Fall Armyworm',
    confidence: 0.75,
    latitude: 36.1200,
    longitude: -118.5000,
    severity: 'medium',
    image_url: null,
    device_id: 'drone-cam-02',
    created_at: '2026-02-10T12:00:00.000Z',
    synced: true,
  },
  {
    id: '11111111-1111-1111-1111-111111111108',
    species: 'Fall Armyworm',
    confidence: 0.72,
    latitude: 35.6800,
    longitude: -118.8000,
    severity: 'medium',
    image_url: null,
    device_id: 'trap-sensor-02',
    created_at: '2026-02-09T09:00:00.000Z',
    synced: true,
  },

  // Green Peach Aphid — Sacramento Delta cluster
  {
    id: '22222222-2222-2222-2222-222222222201',
    species: 'Green Peach Aphid',
    confidence: 0.96,
    latitude: 38.0800,
    longitude: -121.7000,
    severity: 'high',
    image_url: null,
    device_id: 'mobile-user-03',
    created_at: '2026-02-13T07:00:00.000Z',
    synced: true,
  },
  {
    id: '22222222-2222-2222-2222-222222222202',
    species: 'Green Peach Aphid',
    confidence: 0.93,
    latitude: 38.1200,
    longitude: -121.6500,
    severity: 'high',
    image_url: null,
    device_id: 'field-scout-03',
    created_at: '2026-02-13T04:30:00.000Z',
    synced: true,
  },
  {
    id: '22222222-2222-2222-2222-222222222203',
    species: 'Green Peach Aphid',
    confidence: 0.89,
    latitude: 38.0500,
    longitude: -121.7500,
    severity: 'medium',
    image_url: null,
    device_id: 'trap-sensor-01',
    created_at: '2026-02-12T20:00:00.000Z',
    synced: true,
  },
  {
    id: '22222222-2222-2222-2222-222222222204',
    species: 'Green Peach Aphid',
    confidence: 0.87,
    latitude: 38.2100,
    longitude: -121.5200,
    severity: 'medium',
    image_url: null,
    device_id: 'mobile-user-03',
    created_at: '2026-02-12T15:00:00.000Z',
    synced: true,
  },

  // Spotted Lanternfly — scattered, low confidence
  {
    id: '33333333-3333-3333-3333-333333333301',
    species: 'Spotted Lanternfly',
    confidence: 0.58,
    latitude: 37.9577,
    longitude: -121.2908,
    severity: 'medium',
    image_url: null,
    device_id: 'mobile-user-02',
    created_at: '2026-02-11T11:00:00.000Z',
    synced: true,
  },
  {
    id: '33333333-3333-3333-3333-333333333302',
    species: 'Spotted Lanternfly',
    confidence: 0.45,
    latitude: 38.5816,
    longitude: -121.4944,
    severity: 'low',
    image_url: null,
    device_id: 'field-scout-02',
    created_at: '2026-02-10T08:00:00.000Z',
    synced: true,
  },
  {
    id: '33333333-3333-3333-3333-333333333303',
    species: 'Spotted Lanternfly',
    confidence: 0.52,
    latitude: 36.9841,
    longitude: -120.0607,
    severity: 'low',
    image_url: null,
    device_id: 'drone-cam-01',
    created_at: '2026-02-08T14:00:00.000Z',
    synced: true,
  },

  // Background noise — Whitefly
  {
    id: '44444444-4444-4444-4444-444444444401',
    species: 'Whitefly',
    confidence: 0.71,
    latitude: 38.5382,
    longitude: -121.7617,
    severity: 'low',
    image_url: null,
    device_id: 'trap-sensor-02',
    created_at: '2026-02-12T09:00:00.000Z',
    synced: true,
  },
  {
    id: '44444444-4444-4444-4444-444444444402',
    species: 'Whitefly',
    confidence: 0.65,
    latitude: 36.6002,
    longitude: -119.3470,
    severity: 'low',
    image_url: null,
    device_id: 'field-scout-01',
    created_at: '2026-02-11T07:00:00.000Z',
    synced: true,
  },

  // Background noise — Japanese Beetle
  {
    id: '55555555-5555-5555-5555-555555555501',
    species: 'Japanese Beetle',
    confidence: 0.68,
    latitude: 37.7749,
    longitude: -121.2000,
    severity: 'low',
    image_url: null,
    device_id: 'mobile-user-01',
    created_at: '2026-02-10T16:00:00.000Z',
    synced: true,
  },
  {
    id: '55555555-5555-5555-5555-555555555502',
    species: 'Japanese Beetle',
    confidence: 0.62,
    latitude: 37.2000,
    longitude: -120.2500,
    severity: 'low',
    image_url: null,
    device_id: 'trap-sensor-01',
    created_at: '2026-02-09T12:00:00.000Z',
    synced: true,
  },
];

// Risk assessment for current farmer (Garcia Family Farm, Fresno)
export const CURRENT_FARMER_RISKS: FarmerRisk[] = [
  {
    species: 'Fall Armyworm',
    sighting_count: 8,
    avg_severity: 3.25, // between high and critical
    nearest_km: 3.4,
    risk_score: 100,
  },
  {
    species: 'Whitefly',
    sighting_count: 1,
    avg_severity: 1.0,
    nearest_km: 45.2,
    risk_score: 2.1,
  },
  {
    species: 'Spotted Lanternfly',
    sighting_count: 1,
    avg_severity: 1.0,
    nearest_km: 38.7,
    risk_score: 1.3,
  },
];

// Species list for filter dropdown
export const SPECIES_LIST: SpeciesCount[] = [
  { species: 'Fall Armyworm', count: 55 },
  { species: 'Green Peach Aphid', count: 35 },
  { species: 'Spotted Lanternfly', count: 20 },
  { species: 'Whitefly', count: 15 },
  { species: 'Japanese Beetle', count: 5 },
];

// Helper: get overall risk level for display
export function getOverallRiskLevel(risks: FarmerRisk[]): {
  level: 'low' | 'moderate' | 'high' | 'critical';
  label: string;
  color: string;
} {
  if (risks.length === 0) return { level: 'low', label: 'Low Risk', color: '#22c55e' };
  const topScore = risks[0].risk_score;
  if (topScore >= 80) return { level: 'critical', label: 'Critical Risk', color: '#ef4444' };
  if (topScore >= 40) return { level: 'high', label: 'High Risk', color: '#f97316' };
  if (topScore >= 15) return { level: 'moderate', label: 'Moderate Risk', color: '#eab308' };
  return { level: 'low', label: 'Low Risk', color: '#22c55e' };
}

// Helper: get sightings near a location (simple distance filter)
export function getNearbySightings(
  lat: number,
  lng: number,
  radiusKm: number = 50,
): Sighting[] {
  return MOCK_SIGHTINGS.filter((s) => {
    const dLat = (s.latitude - lat) * 111.32;
    const dLng = (s.longitude - lng) * 111.32 * Math.cos((lat * Math.PI) / 180);
    const dist = Math.sqrt(dLat * dLat + dLng * dLng);
    return dist <= radiusKm;
  });
}

// Helper: get sightings filtered by species
export function getSightingsBySpecies(species: string): Sighting[] {
  return MOCK_SIGHTINGS.filter((s) => s.species === species);
}

// Helper: format time ago for display
export function timeAgo(dateString: string): string {
  const now = Date.now();
  const then = new Date(dateString).getTime();
  const hours = Math.floor((now - then) / (1000 * 60 * 60));
  if (hours < 1) return 'Just now';
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}
