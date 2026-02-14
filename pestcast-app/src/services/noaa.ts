const NOAA_BASE = 'https://api.weather.gov';
const CACHE_TTL = 30 * 60 * 1000; // 30 minutes

let cachedData: GeoJSON.FeatureCollection | null = null;
let cachedAt = 0;

/** Generate a grid of lat/lng sample points around a center. */
function generateGrid(
    centerLat: number,
    centerLng: number,
    steps = 4,
    spacing = 0.08
): { lat: number; lng: number }[] {
    const points: { lat: number; lng: number }[] = [];
    const half = (steps - 1) / 2;
    for (let i = 0; i < steps; i++) {
        for (let j = 0; j < steps; j++) {
            points.push({
                lat: +(centerLat + (i - half) * spacing).toFixed(4),
                lng: +(centerLng + (j - half) * spacing).toFixed(4),
            });
        }
    }
    return points;
}

/** Fetch gridpoint forecast URL from NOAA for a lat/lng. */
async function getGridpointUrl(lat: number, lng: number): Promise<string | null> {
    try {
        const res = await fetch(`${NOAA_BASE}/points/${lat},${lng}`, {
            headers: { 'User-Agent': 'PestCast App (pestcast@demo.app)' },
        });
        if (!res.ok) return null;
        const json = await res.json();
        return json.properties?.forecastGridData || null;
    } catch {
        return null;
    }
}

/** Fetch current temperature from a gridpoint URL. */
async function getTemperatureFromGridpoint(url: string): Promise<number | null> {
    try {
        const res = await fetch(url, {
            headers: { 'User-Agent': 'PestCast App (pestcast@demo.app)' },
        });
        if (!res.ok) return null;
        const json = await res.json();
        const tempValues = json.properties?.temperature?.values;
        if (!tempValues || tempValues.length === 0) return null;

        const now = new Date();
        for (const entry of tempValues) {
            const [start, duration] = entry.validTime.split('/');
            const startTime = new Date(start);
            const hours = parseInt(duration.match(/(\d+)H/)?.[1] || '1', 10);
            const endTime = new Date(startTime.getTime() + hours * 3600000);
            if (now >= startTime && now <= endTime) {
                return entry.value as number;
            }
        }
        return tempValues[0].value as number;
    } catch {
        return null;
    }
}

/**
 * Fetch a grid of temperatures around a center point.
 * Returns a GeoJSON FeatureCollection with temperature properties.
 * Results are cached for 30 minutes.
 */
export async function fetchTemperatureGrid(
    centerLat = 38.5449,
    centerLng = -121.7405
): Promise<GeoJSON.FeatureCollection> {
    if (cachedData && Date.now() - cachedAt < CACHE_TTL) {
        return cachedData;
    }

    const grid = generateGrid(centerLat, centerLng, 4, 0.08);
    const features: GeoJSON.Feature[] = [];

    for (const point of grid) {
        try {
            await new Promise((r) => setTimeout(r, 120)); // rate limit
            const gridUrl = await getGridpointUrl(point.lat, point.lng);
            if (!gridUrl) continue;

            await new Promise((r) => setTimeout(r, 120));
            const temp = await getTemperatureFromGridpoint(gridUrl);
            if (temp === null) continue;

            features.push({
                type: 'Feature',
                geometry: { type: 'Point', coordinates: [point.lng, point.lat] },
                properties: { temperature: temp },
            });
        } catch (err: any) {
            console.warn(`NOAA fetch failed for ${point.lat},${point.lng}:`, err?.message);
        }
    }

    const result: GeoJSON.FeatureCollection = { type: 'FeatureCollection', features };
    cachedData = result;
    cachedAt = Date.now();
    return result;
}
