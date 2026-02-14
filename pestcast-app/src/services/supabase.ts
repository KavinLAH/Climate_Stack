import { createClient } from '@supabase/supabase-js';
import { SUPABASE_URL, SUPABASE_ANON_KEY } from '../config';

export const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

/** Shape of a row in the sightings table. */
export interface SightingRow {
    id: string;
    species: string;
    confidence: number;
    latitude: number;
    longitude: number;
    severity: 'low' | 'medium' | 'high' | 'critical';
    image_url?: string;
    device_id?: string;
    created_at: string;
}

/** Convert a sighting row to a GeoJSON Feature. */
export function sightingToFeature(row: SightingRow): GeoJSON.Feature {
    return {
        type: 'Feature',
        geometry: {
            type: 'Point',
            coordinates: [row.longitude, row.latitude],
        },
        properties: {
            id: row.id,
            species: row.species,
            confidence: row.confidence,
            severity: row.severity,
            image_url: row.image_url,
            device_id: row.device_id,
            created_at: row.created_at,
        },
    };
}

/** Fetch all sightings as a GeoJSON FeatureCollection. */
export async function fetchSightings(): Promise<GeoJSON.FeatureCollection> {
    const { data, error } = await supabase
        .from('sightings')
        .select('*')
        .order('created_at', { ascending: false });

    if (error) {
        console.warn('Supabase fetch error:', error.message);
        return { type: 'FeatureCollection', features: [] };
    }

    const features = (data || []).map((row: SightingRow) => sightingToFeature(row));
    return { type: 'FeatureCollection', features };
}

/**
 * Subscribe to real-time INSERT events on the sightings table.
 * Returns an unsubscribe function.
 */
export function subscribeSightings(
    onInsert: (feature: GeoJSON.Feature) => void
): () => void {
    const channel = supabase
        .channel('sightings-realtime')
        .on(
            'postgres_changes',
            {
                event: 'INSERT',
                schema: 'public',
                table: 'sightings',
            },
            (payload) => {
                const feature = sightingToFeature(payload.new as SightingRow);
                onInsert(feature);
            }
        )
        .subscribe();

    return () => {
        supabase.removeChannel(channel);
    };
}
