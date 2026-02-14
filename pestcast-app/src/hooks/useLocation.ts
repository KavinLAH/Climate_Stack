import { useEffect, useState } from 'react';
import * as Location from 'expo-location';
import type { Coordinates } from '../types';

/**
 * Hook that requests foreground location permission on mount and exposes a
 * helper to grab the device's current GPS coordinates on demand.
 */
export function useLocation() {
    const [permissionGranted, setPermissionGranted] = useState(false);

    useEffect(() => {
        (async () => {
            const { status } = await Location.requestForegroundPermissionsAsync();
            setPermissionGranted(status === 'granted');
        })();
    }, []);

    /** Returns the current lat/lng or `null` if permission was denied. */
    async function getCurrentPosition(): Promise<Coordinates | null> {
        if (!permissionGranted) {
            const { status } = await Location.requestForegroundPermissionsAsync();
            if (status !== 'granted') return null;
            setPermissionGranted(true);
        }

        const loc = await Location.getCurrentPositionAsync({
            accuracy: Location.Accuracy.High,
        });

        return { lat: loc.coords.latitude, lng: loc.coords.longitude };
    }

    return { permissionGranted, getCurrentPosition };
}
