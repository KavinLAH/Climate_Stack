// Helper to find env vars with or without Expo prefix
const getEnv = (key: string) => {
    return process.env[`EXPO_PUBLIC_${key}`] || process.env[key] || '';
};

export const MAPBOX_TOKEN = getEnv('MAPBOX_TOKEN');
export const SUPABASE_URL = getEnv('SUPABASE_URL');
export const SUPABASE_ANON_KEY = getEnv('SUPABASE_ANON_KEY');

console.log('[Config] Environment Check:');
console.log(' - Supabase URL:', SUPABASE_URL ? 'Loaded' : 'MISSING');
console.log(' - Mapbox Token:', MAPBOX_TOKEN ? 'Loaded' : 'MISSING');

// Davis, CA center coordinates
export const MAP_CENTER: [number, number] = [-121.7405, 38.5449];
export const MAP_ZOOM = 9;
