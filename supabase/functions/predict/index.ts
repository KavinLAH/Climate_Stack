import { serve } from 'https://deno.land/std@0.208.0/http/server.ts';
import { corsHeaders } from '../_shared/cors.ts';
import { getSupabaseClient } from '../_shared/supabase-client.ts';

const EARTH_RADIUS_KM = 6371;
const DEG_TO_RAD = Math.PI / 180;
const RAD_TO_DEG = 180 / Math.PI;
const CLUSTER_DISTANCE_KM = 10;
const CLUSTER_TIME_HOURS = 48;
const MIN_CLUSTER_SIZE = 3;
const MAX_PROJECTION_KM = 200;

interface SightingRow {
  id: string;
  species: string;
  confidence: number;
  latitude: number;
  longitude: number;
  severity: string;
  created_at: string;
}

interface Cluster {
  sightings: SightingRow[];
  centroidLat: number;
  centroidLng: number;
}

function haversine(lat1: number, lng1: number, lat2: number, lng2: number): number {
  const dLat = (lat2 - lat1) * DEG_TO_RAD;
  const dLng = (lng2 - lng1) * DEG_TO_RAD;
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(lat1 * DEG_TO_RAD) * Math.cos(lat2 * DEG_TO_RAD) * Math.sin(dLng / 2) ** 2;
  return 2 * EARTH_RADIUS_KM * Math.asin(Math.sqrt(a));
}

function bearing(lat1: number, lng1: number, lat2: number, lng2: number): number {
  const dLng = (lng2 - lng1) * DEG_TO_RAD;
  const y = Math.sin(dLng) * Math.cos(lat2 * DEG_TO_RAD);
  const x =
    Math.cos(lat1 * DEG_TO_RAD) * Math.sin(lat2 * DEG_TO_RAD) -
    Math.sin(lat1 * DEG_TO_RAD) * Math.cos(lat2 * DEG_TO_RAD) * Math.cos(dLng);
  return ((Math.atan2(y, x) * RAD_TO_DEG) + 360) % 360;
}

function projectPoint(
  lat: number,
  lng: number,
  distanceKm: number,
  bearingDeg: number,
): [number, number] {
  const d = distanceKm / EARTH_RADIUS_KM;
  const brng = bearingDeg * DEG_TO_RAD;
  const lat1 = lat * DEG_TO_RAD;
  const lng1 = lng * DEG_TO_RAD;

  const lat2 = Math.asin(
    Math.sin(lat1) * Math.cos(d) + Math.cos(lat1) * Math.sin(d) * Math.cos(brng),
  );
  const lng2 =
    lng1 +
    Math.atan2(Math.sin(brng) * Math.sin(d) * Math.cos(lat1), Math.cos(d) - Math.sin(lat1) * Math.sin(lat2));

  return [lat2 * RAD_TO_DEG, lng2 * RAD_TO_DEG];
}

function centroid(points: { latitude: number; longitude: number }[]): [number, number] {
  const lat = points.reduce((s, p) => s + p.latitude, 0) / points.length;
  const lng = points.reduce((s, p) => s + p.longitude, 0) / points.length;
  return [lat, lng];
}

function clusterSightings(sightings: SightingRow[]): Cluster[] {
  const assigned = new Set<string>();
  const clusters: Cluster[] = [];

  for (const s of sightings) {
    if (assigned.has(s.id)) continue;

    const group: SightingRow[] = [s];
    assigned.add(s.id);
    const sTime = new Date(s.created_at).getTime();

    for (const other of sightings) {
      if (assigned.has(other.id)) continue;
      const dist = haversine(s.latitude, s.longitude, other.latitude, other.longitude);
      const timeDiff = Math.abs(new Date(other.created_at).getTime() - sTime) / (1000 * 60 * 60);

      if (dist <= CLUSTER_DISTANCE_KM && timeDiff <= CLUSTER_TIME_HOURS) {
        group.push(other);
        assigned.add(other.id);
      }
    }

    if (group.length >= MIN_CLUSTER_SIZE) {
      const [cLat, cLng] = centroid(group);
      clusters.push({ sightings: group, centroidLat: cLat, centroidLng: cLng });
    }
  }

  return clusters;
}

async function fetchWind(lat: number, lng: number): Promise<{ speed: number; direction: number }> {
  try {
    const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lng}&current=wind_speed_10m,wind_direction_10m`;
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`Open-Meteo status ${resp.status}`);
    const data = await resp.json();
    return {
      speed: data.current.wind_speed_10m ?? 10,
      direction: data.current.wind_direction_10m ?? 315,
    };
  } catch {
    return { speed: 10, direction: 315 };
  }
}

function generateEllipse(
  originLat: number,
  originLng: number,
  endLat: number,
  endLng: number,
  distanceKm: number,
): number[][] {
  const midLat = (originLat + endLat) / 2;
  const midLng = (originLng + endLng) / 2;
  const dir = bearing(originLat, originLng, endLat, endLng);

  const narrowWidth = Math.max(3, distanceKm * 0.15);
  const wideWidth = Math.max(5, distanceKm * 0.35);
  const halfLen = distanceKm / 2;

  const points: number[][] = [];
  const steps = 24;

  for (let i = 0; i <= steps; i++) {
    const angle = (i / steps) * 2 * Math.PI;
    const along = Math.cos(angle) * halfLen;
    const across = Math.sin(angle) * (angle > Math.PI ? narrowWidth : wideWidth);

    const rotatedAlong = along * Math.cos(dir * DEG_TO_RAD) - across * Math.sin(dir * DEG_TO_RAD);
    const rotatedAcross = along * Math.sin(dir * DEG_TO_RAD) + across * Math.cos(dir * DEG_TO_RAD);

    const dLat = rotatedAlong / 111.32;
    const dLng = rotatedAcross / (111.32 * Math.cos(midLat * DEG_TO_RAD));

    points.push([midLng + dLng, midLat + dLat]);
  }

  points.push(points[0]);
  return points;
}

function assessRisk(cluster: Cluster, speed: number): string {
  const avgSeverityScore = cluster.sightings.reduce((s, si) => {
    const scores = { low: 1, medium: 2, high: 3, critical: 4 };
    return s + (scores[si.severity as keyof typeof scores] || 1);
  }, 0) / cluster.sightings.length;

  if (avgSeverityScore >= 3 && speed > 5) return 'critical';
  if (avgSeverityScore >= 2.5 || speed > 10) return 'high';
  if (avgSeverityScore >= 1.5) return 'medium';
  return 'low';
}

serve(async (req: Request) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { status: 204, headers: corsHeaders });
  }

  try {
    let species: string | null = null;
    let hoursBack = 48;
    let hoursForward = 72;

    if (req.method === 'GET') {
      const url = new URL(req.url);
      species = url.searchParams.get('species');
      hoursBack = parseInt(url.searchParams.get('hours_back') || '48');
      hoursForward = parseInt(url.searchParams.get('hours_forward') || '72');
    } else if (req.method === 'POST') {
      const body = await req.json();
      species = body.species || null;
      hoursBack = body.hours_back || 48;
      hoursForward = body.hours_forward || 72;
    }

    if (!species) {
      return new Response(
        JSON.stringify({ error: 'species parameter is required' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } },
      );
    }

    const supabase = getSupabaseClient();
    const since = new Date(Date.now() - hoursBack * 60 * 60 * 1000).toISOString();

    const { data: sightings, error } = await supabase
      .from('sightings')
      .select('id, species, confidence, latitude, longitude, severity, created_at')
      .eq('species', species)
      .gte('created_at', since)
      .order('created_at', { ascending: true });

    if (error) {
      return new Response(
        JSON.stringify({ error: error.message }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } },
      );
    }

    if (!sightings || sightings.length < MIN_CLUSTER_SIZE) {
      return new Response(
        JSON.stringify({
          type: 'FeatureCollection',
          features: [],
          metadata: { species, sightings_found: sightings?.length ?? 0, message: 'Insufficient data for prediction' },
        }),
        { status: 200, headers: { ...corsHeaders, 'Content-Type': 'application/json' } },
      );
    }

    const clusters = clusterSightings(sightings);

    if (clusters.length === 0) {
      return new Response(
        JSON.stringify({
          type: 'FeatureCollection',
          features: [],
          metadata: { species, sightings_found: sightings.length, clusters_found: 0, message: 'No dense clusters found' },
        }),
        { status: 200, headers: { ...corsHeaders, 'Content-Type': 'application/json' } },
      );
    }

    const features = [];

    for (const cluster of clusters) {
      const sorted = [...cluster.sightings].sort(
        (a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
      );

      const thirdLen = Math.max(1, Math.floor(sorted.length / 3));
      const firstThird = sorted.slice(0, thirdLen);
      const lastThird = sorted.slice(-thirdLen);

      const [earlyLat, earlyLng] = centroid(firstThird);
      const [lateLat, lateLng] = centroid(lastThird);

      const sightingBearing = bearing(earlyLat, earlyLng, lateLat, lateLng);
      const sightingDist = haversine(earlyLat, earlyLng, lateLat, lateLng);

      const timeSpanHours = Math.max(
        1,
        (new Date(sorted[sorted.length - 1].created_at).getTime() - new Date(sorted[0].created_at).getTime()) /
          (1000 * 60 * 60),
      );
      const sightingSpeed = sightingDist / timeSpanHours;

      const wind = await fetchWind(cluster.centroidLat, cluster.centroidLng);

      const windPushDirection = (wind.direction + 180) % 360;
      const projectedBearing = sightingBearing * 0.6 + windPushDirection * 0.4;

      const windSpeedKmh = wind.speed;
      const projectedSpeed = sightingSpeed * 0.6 + windSpeedKmh * 0.4;

      let projectedDistKm = projectedSpeed * hoursForward;
      projectedDistKm = Math.min(projectedDistKm, MAX_PROJECTION_KM);

      const [endLat, endLng] = projectPoint(
        cluster.centroidLat,
        cluster.centroidLng,
        projectedDistKm,
        projectedBearing,
      );

      const polygon = generateEllipse(
        cluster.centroidLat,
        cluster.centroidLng,
        endLat,
        endLng,
        projectedDistKm,
      );

      const avgConfidence =
        cluster.sightings.reduce((s, si) => s + si.confidence, 0) / cluster.sightings.length;
      const clusterConfidence = Math.min(1, avgConfidence * (cluster.sightings.length / 10));

      const risk = assessRisk(cluster, projectedSpeed);

      features.push({
        type: 'Feature',
        geometry: {
          type: 'Polygon',
          coordinates: [polygon],
        },
        properties: {
          species,
          cluster_size: cluster.sightings.length,
          centroid: [cluster.centroidLng, cluster.centroidLat],
          projected_end: [endLng, endLat],
          bearing: parseFloat(projectedBearing.toFixed(1)),
          speed_kmh: parseFloat(projectedSpeed.toFixed(2)),
          projected_distance_km: parseFloat(projectedDistKm.toFixed(1)),
          hours_forward: hoursForward,
          wind_speed_kmh: wind.speed,
          wind_direction: wind.direction,
          risk_level: risk,
          confidence: parseFloat(clusterConfidence.toFixed(2)),
        },
      });
    }

    const geojson = {
      type: 'FeatureCollection',
      features,
      metadata: {
        species,
        sightings_found: sightings.length,
        clusters_found: clusters.length,
        hours_back: hoursBack,
        hours_forward: hoursForward,
        generated_at: new Date().toISOString(),
      },
    };

    return new Response(JSON.stringify(geojson), {
      status: 200,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  } catch (err) {
    return new Response(
      JSON.stringify({ error: err instanceof Error ? err.message : 'Internal server error' }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } },
    );
  }
});
