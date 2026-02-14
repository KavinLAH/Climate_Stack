import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import type { Sighting, Farmer, SeverityLevel } from './types.js';
import {
  ARMYWORM_PATH,
  APHID_DELTA_POINTS,
  VALLEY_WIDE_POINTS,
  NOISE_POINTS,
  SPECIES,
  DEVICE_IDS,
} from './constants.js';

const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!supabaseUrl || !supabaseKey) {
  console.error('Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in .env');
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseKey);

// Helpers
function randomInRange(min: number, max: number): number {
  return Math.random() * (max - min) + min;
}

function jitter(coord: number, range = 0.02): number {
  return coord + randomInRange(-range, range);
}

function randomItem<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

function hoursAgo(hours: number): string {
  return new Date(Date.now() - hours * 60 * 60 * 1000).toISOString();
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

// Narrative 1: Fall Armyworm migration (~55 sightings)
function generateArmywormSightings(): Sighting[] {
  const sightings: Sighting[] = [];
  const totalHours = 240;
  const pathLen = ARMYWORM_PATH.length;

  for (let i = 0; i < 55; i++) {
    const progress = Math.min(1, (i / 55) + randomInRange(-0.05, 0.05));
    const pathIndex = Math.min(pathLen - 2, Math.floor(progress * (pathLen - 1)));
    const t = (progress * (pathLen - 1)) - pathIndex;

    const baseLat = lerp(ARMYWORM_PATH[pathIndex][0], ARMYWORM_PATH[pathIndex + 1][0], t);
    const baseLng = lerp(ARMYWORM_PATH[pathIndex][1], ARMYWORM_PATH[pathIndex + 1][1], t);

    const hoursBack = totalHours * (1 - progress) + randomInRange(-6, 6);

    let severity: SeverityLevel;
    if (progress < 0.25) severity = 'medium';
    else if (progress < 0.55) severity = 'high';
    else severity = 'critical';

    const confidence = Math.min(1, randomInRange(0.7, 0.95) + progress * 0.05);

    sightings.push({
      species: SPECIES.FALL_ARMYWORM,
      confidence: parseFloat(confidence.toFixed(2)),
      latitude: parseFloat(jitter(baseLat, 0.03).toFixed(6)),
      longitude: parseFloat(jitter(baseLng, 0.03).toFixed(6)),
      severity,
      image_url: null,
      device_id: randomItem(DEVICE_IDS),
      created_at: hoursAgo(Math.max(0.5, hoursBack)),
      synced: true,
    });
  }

  return sightings;
}

// Narrative 2: Aphid cluster along Sacramento River delta (~35 sightings)
function generateAphidSightings(): Sighting[] {
  const sightings: Sighting[] = [];

  for (let i = 0; i < 35; i++) {
    const point = randomItem(APHID_DELTA_POINTS);
    const hoursBack = randomInRange(1, 120);

    sightings.push({
      species: SPECIES.APHID,
      confidence: parseFloat(randomInRange(0.75, 0.98).toFixed(2)),
      latitude: parseFloat(jitter(point[0], 0.015).toFixed(6)),
      longitude: parseFloat(jitter(point[1], 0.015).toFixed(6)),
      severity: randomItem(['medium', 'high'] as SeverityLevel[]),
      image_url: null,
      device_id: randomItem(DEVICE_IDS),
      created_at: hoursAgo(hoursBack),
      synced: true,
    });
  }

  return sightings;
}

// Narrative 3: Spotted Lanternfly — scattered, low confidence (~20 sightings)
function generateLanternflySightings(): Sighting[] {
  const sightings: Sighting[] = [];

  for (let i = 0; i < 20; i++) {
    const point = randomItem(VALLEY_WIDE_POINTS);

    sightings.push({
      species: SPECIES.SPOTTED_LANTERNFLY,
      confidence: parseFloat(randomInRange(0.40, 0.75).toFixed(2)),
      latitude: parseFloat(jitter(point[0], 0.05).toFixed(6)),
      longitude: parseFloat(jitter(point[1], 0.05).toFixed(6)),
      severity: randomItem(['low', 'medium'] as SeverityLevel[]),
      image_url: null,
      device_id: randomItem(DEVICE_IDS),
      created_at: hoursAgo(randomInRange(1, 336)),
      synced: true,
    });
  }

  return sightings;
}

// Narrative 4: Background noise — Whitefly + Japanese Beetle (~20 sightings)
function generateNoiseSightings(): Sighting[] {
  const sightings: Sighting[] = [];
  const noiseSpecies = [SPECIES.WHITEFLY, SPECIES.JAPANESE_BEETLE];

  for (let i = 0; i < 20; i++) {
    const point = randomItem(NOISE_POINTS);

    sightings.push({
      species: randomItem(noiseSpecies),
      confidence: parseFloat(randomInRange(0.50, 0.85).toFixed(2)),
      latitude: parseFloat(jitter(point[0], 0.04).toFixed(6)),
      longitude: parseFloat(jitter(point[1], 0.04).toFixed(6)),
      severity: 'low',
      image_url: null,
      device_id: randomItem(DEVICE_IDS),
      created_at: hoursAgo(randomInRange(1, 168)),
      synced: true,
    });
  }

  return sightings;
}

// Demo farmers across Central Valley
const DEMO_FARMERS: Farmer[] = [
  { name: 'Garcia Family Farm',     latitude: 36.7300, longitude: -119.7900, device_id: 'mobile-user-01' },
  { name: 'Johnson Ranch',          latitude: 35.4200, longitude: -119.0300, device_id: 'mobile-user-02' },
  { name: 'Chen Orchards',          latitude: 38.1000, longitude: -121.6800, device_id: 'mobile-user-03' },
  { name: 'Patel Vineyards',        latitude: 36.3300, longitude: -119.3000, device_id: 'field-scout-01' },
  { name: 'Williams Dairy & Crops', latitude: 37.6400, longitude: -121.0000, device_id: 'field-scout-02' },
  { name: 'Nguyen Vegetable Farm',  latitude: 38.5500, longitude: -121.7400, device_id: 'field-scout-03' },
  { name: 'Rodriguez Almond Farm',  latitude: 36.1000, longitude: -119.0200, device_id: 'drone-cam-01' },
  { name: 'Kim Berry Farm',         latitude: 37.3000, longitude: -120.4800, device_id: 'drone-cam-02' },
];

async function seed() {
  console.log('Generating seed data...');

  const allSightings = [
    ...generateArmywormSightings(),
    ...generateAphidSightings(),
    ...generateLanternflySightings(),
    ...generateNoiseSightings(),
  ];

  console.log(`Generated ${allSightings.length} sightings`);

  // Clear existing data
  const { error: deleteFarmersError } = await supabase
    .from('farmers')
    .delete()
    .neq('id', '00000000-0000-0000-0000-000000000000');

  if (deleteFarmersError) {
    console.error('Error clearing farmers:', deleteFarmersError.message);
  }

  const { error: deleteError } = await supabase
    .from('sightings')
    .delete()
    .neq('id', '00000000-0000-0000-0000-000000000000');

  if (deleteError) {
    console.error('Error clearing existing data:', deleteError.message);
  } else {
    console.log('Cleared existing data');
  }

  // Insert farmers
  const { error: farmersError } = await supabase.from('farmers').insert(DEMO_FARMERS);
  if (farmersError) {
    console.error('Error inserting farmers:', farmersError.message);
  } else {
    console.log(`Inserted ${DEMO_FARMERS.length} demo farmers`);
  }

  // Insert sightings in batches of 50
  const batchSize = 50;
  let inserted = 0;

  for (let i = 0; i < allSightings.length; i += batchSize) {
    const batch = allSightings.slice(i, i + batchSize);
    const { error } = await supabase.from('sightings').insert(batch);

    if (error) {
      console.error(`Error inserting batch ${i / batchSize + 1}:`, error.message);
    } else {
      inserted += batch.length;
      console.log(`Inserted batch ${Math.floor(i / batchSize) + 1} (${inserted}/${allSightings.length})`);
    }
  }

  // Verify
  const { count, error: countError } = await supabase
    .from('sightings')
    .select('*', { count: 'exact', head: true });

  if (countError) {
    console.error('Error verifying:', countError.message);
  } else {
    console.log(`\nSeed complete! ${count} sightings in database.`);
  }

  // Show species breakdown
  const { data: species, error: speciesError } = await supabase
    .rpc('get_distinct_species');

  if (!speciesError && species) {
    console.log('\nSpecies breakdown:');
    for (const s of species) {
      console.log(`  ${s.species}: ${s.count}`);
    }
  }

  // Show farmer risk for demo
  console.log('\nFarmer risk assessment (demo):');
  for (const farmer of DEMO_FARMERS) {
    const { data: risk } = await supabase.rpc('get_farmer_risk', {
      farm_lat: farmer.latitude,
      farm_lng: farmer.longitude,
    });
    if (risk && risk.length > 0) {
      const topRisk = risk[0];
      console.log(`  ${farmer.name}: ${topRisk.species} (score: ${topRisk.risk_score.toFixed(1)}, nearest: ${topRisk.nearest_km.toFixed(1)}km)`);
    } else {
      console.log(`  ${farmer.name}: No nearby threats`);
    }
  }
}

seed().catch(console.error);
