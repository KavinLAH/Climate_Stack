/** A single prediction from the TFLite classifier. */
export interface ClassificationResult {
    label: string;
    confidence: number;
}

/** Severity levels for a pest sighting. */
export type Severity = 'low' | 'medium' | 'high' | 'critical';

/** A pest sighting record, mirroring the Supabase `sightings` table. */
export interface Sighting {
    id: string;
    image_url: string;
    pest_type: string;
    confidence: number;
    lat: number;
    lng: number;
    severity: Severity;
    device_id: string;
    created_at: string;
    synced: boolean;
}

/** GPS coordinates. */
export interface Coordinates {
    lat: number;
    lng: number;
}

/** Application screen states. */
export type AppScreen = 'camera' | 'classifying' | 'results' | 'syncing' | 'done';
