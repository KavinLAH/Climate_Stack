// Real California Central Valley farmland coordinates

// Fall Armyworm migration path: Bakersfield → Visalia → Fresno (SW to NE)
export const ARMYWORM_PATH: [number, number][] = [
  [35.3733, -119.0187],  // Bakersfield
  [35.4500, -118.9500],  // North Bakersfield
  [35.5600, -118.8800],  // McFarland area
  [35.6800, -118.8000],  // Delano
  [35.7900, -118.7200],  // Earlimart
  [35.9000, -118.6500],  // Pixley
  [36.0100, -118.5800],  // Tipton
  [36.1200, -118.5000],  // Tulare
  [36.2300, -118.4200],  // Near Visalia
  [36.3300, -119.2900],  // Visalia proper
  [36.4500, -119.3500],  // Goshen
  [36.5800, -119.4200],  // Kingsburg
  [36.6500, -119.5000],  // Selma
  [36.7477, -119.7724],  // Fresno
];

// Aphid cluster — Sacramento River Delta points
export const APHID_DELTA_POINTS: [number, number][] = [
  [38.0500, -121.7500],  // Rio Vista
  [38.0800, -121.7000],  // Birds Landing
  [38.1200, -121.6500],  // Isleton
  [38.1500, -121.6000],  // Walnut Grove
  [38.1800, -121.5500],  // Locke
  [38.2100, -121.5200],  // Courtland
  [38.2500, -121.5000],  // Hood
  [38.3000, -121.4800],  // Freeport
  [38.0200, -121.8000],  // Antioch (east)
  [38.0600, -121.6800],  // Bethel Island
];

// Scattered points across Central Valley (for Spotted Lanternfly)
export const VALLEY_WIDE_POINTS: [number, number][] = [
  [38.6785, -121.7733],  // Woodland/Davis
  [38.5816, -121.4944],  // Sacramento
  [37.9577, -121.2908],  // Stockton
  [37.6391, -120.9969],  // Modesto
  [37.3022, -120.4830],  // Merced
  [36.7477, -119.7724],  // Fresno
  [35.3733, -119.0187],  // Bakersfield
  [36.3302, -119.2921],  // Visalia
  [36.9841, -120.0607],  // Madera
  [37.4830, -120.8470],  // Turlock
];

// Background noise locations
export const NOISE_POINTS: [number, number][] = [
  [38.5382, -121.7617],  // Davis
  [37.7749, -121.2000],  // Tracy
  [36.6002, -119.3470],  // Hanford
  [35.8900, -119.0600],  // Wasco
  [37.2000, -120.2500],  // Chowchilla
  [38.3500, -121.9600],  // Vacaville
  [36.0988, -119.0156],  // Porterville
  [37.9500, -121.6800],  // Brentwood
];

export const SPECIES = {
  FALL_ARMYWORM: 'Fall Armyworm',
  APHID: 'Green Peach Aphid',
  SPOTTED_LANTERNFLY: 'Spotted Lanternfly',
  WHITEFLY: 'Whitefly',
  JAPANESE_BEETLE: 'Japanese Beetle',
} as const;

export const DEVICE_IDS = [
  'field-scout-01',
  'field-scout-02',
  'field-scout-03',
  'drone-cam-01',
  'drone-cam-02',
  'trap-sensor-01',
  'trap-sensor-02',
  'mobile-user-01',
  'mobile-user-02',
  'mobile-user-03',
];
