import * as FileSystem from 'expo-file-system/legacy';
import { supabase } from './supabase';
import { enqueue } from './offlineQueue';
import type { Sighting } from '../types';

// Supabase Storage bucket name for sighting images.
const STORAGE_BUCKET = 'sighting-images';

// ---------------------------------------------------------------------------
// Image upload
// ---------------------------------------------------------------------------

/**
 * Upload a local image file to Supabase Storage and return its public URL.
 * The image is read from the local cache URI produced by the camera.
 */
export async function uploadImage(localUri: string): Promise<string> {
    const filename = `${Date.now()}_${localUri.split('/').pop()}`;

    // Read the file as base64.
    const base64 = await FileSystem.readAsStringAsync(localUri, {
        encoding: FileSystem.EncodingType.Base64,
    });

    // Convert base64 → ArrayBuffer for Supabase upload.
    const binaryStr = atob(base64);
    const bytes = new Uint8Array(binaryStr.length);
    for (let i = 0; i < binaryStr.length; i++) {
        bytes[i] = binaryStr.charCodeAt(i);
    }

    const { error: uploadError } = await supabase.storage
        .from(STORAGE_BUCKET)
        .upload(filename, bytes.buffer, {
            contentType: 'image/jpeg',
            upsert: false,
        });

    if (uploadError) throw uploadError;

    const { data } = supabase.storage.from(STORAGE_BUCKET).getPublicUrl(filename);
    return data.publicUrl;
}

// ---------------------------------------------------------------------------
// Sighting insert
// ---------------------------------------------------------------------------

/**
 * Insert a sighting row into the Supabase `sightings` table.
 */
export async function insertSighting(sighting: Omit<Sighting, 'synced'>) {
    const { error } = await supabase.from('sightings').insert({
        id: sighting.id,
        species: sighting.pest_type,
        confidence: sighting.confidence,
        latitude: sighting.lat,
        longitude: sighting.lng,
        severity: sighting.severity,
        image_url: sighting.image_url,
        device_id: sighting.device_id,
        created_at: sighting.created_at,
        timestamp: new Date().toISOString(), // Added as per Task 3 instruction
        synced: true,
    });

    if (error) throw error;
}

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

/**
 * High-level helper: upload image + insert sighting row.
 * If anything fails (e.g. no connectivity), the sighting is pushed onto
 * the offline queue for later retry.
 */
export async function syncSighting(sighting: Sighting): Promise<boolean> {
    try {
        // 1. Upload the photo and get its public URL.
        const publicUrl = await uploadImage(sighting.image_url);

        // 2. Insert the record (overwriting local cache URI with the cloud URL).
        await insertSighting({ ...sighting, image_url: publicUrl });

        return true;
    } catch (err) {
        console.warn('[syncService] Sync failed, queuing for retry:', err);
        await enqueue(sighting);
        return false;
    }
}
