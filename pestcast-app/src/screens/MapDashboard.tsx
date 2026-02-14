import React, { useEffect, useState, useCallback, useRef } from 'react';
import {
    View,
    Text,
    TouchableOpacity,
    StyleSheet,
    ActivityIndicator,
    Platform,
} from 'react-native';
import MapView, { Marker, Polygon, PROVIDER_DEFAULT } from 'react-native-maps';
import { MAP_CENTER, MAP_ZOOM } from '../config';
import { fetchSightings, subscribeSightings } from '../services/supabase';
import AtRiskCrops from '../components/AtRiskCrops';

const mockWeatherGrid = () => {
    const polygons: any[] = [];
    const startLat = 38.56;
    const startLng = -121.76;
    const step = 0.015; // Slightly larger for react-native-maps visibility

    for (let i = 0; i < 4; i++) {
        for (let j = 0; j < 4; j++) {
            const lat = startLat - i * step;
            const lng = startLng + j * step;
            const temp = (i + j) % 2 === 0 ? 85 : 55; // Alternating grid

            polygons.push({
                id: `weather-${i}-${j}`,
                temp,
                coordinates: [
                    { latitude: lat, longitude: lng },
                    { latitude: lat, longitude: lng + step },
                    { latitude: lat - step, longitude: lng + step },
                    { latitude: lat - step, longitude: lng },
                ],
                color: temp > 80 ? 'rgba(255, 0, 0, 0.4)' : 'rgba(0, 0, 255, 0.2)',
            });
        }
    }
    return polygons;
};

export default function MapDashboard() {
    const [sightings, setSightings] = useState<any[]>([]);
    const [weatherPolygons, setWeatherPolygons] = useState<any[]>([]);
    const [weatherVisible, setWeatherVisible] = useState(false);
    const [weatherLoading, setWeatherLoading] = useState(false);
    const [loading, setLoading] = useState(true);
    const weatherLoadedRef = useRef(false);

    /* ── Load initial sightings + subscribe ── */
    useEffect(() => {
        let unsub: (() => void) | null = null;

        (async () => {
            const geo = await fetchSightings();
            setSightings(geo.features);
            setLoading(false);

            unsub = subscribeSightings((feature) => {
                setSightings((prev) => [...prev, feature]);
            });
        })();

        return () => {
            if (unsub) unsub();
        };
    }, []);

    /* ── Toggle weather overlay ── */
    const toggleWeather = useCallback(async () => {
        const next = !weatherVisible;
        setWeatherVisible(next);

        if (next && !weatherLoadedRef.current) {
            weatherLoadedRef.current = true;
            setWeatherLoading(true);
            setTimeout(() => {
                setWeatherPolygons(mockWeatherGrid());
                setWeatherLoading(false);
            }, 600);
        }
    }, [weatherVisible]);

    return (
        <View style={styles.container}>
            <MapView
                style={styles.map}
                provider={PROVIDER_DEFAULT}
                userInterfaceStyle="dark"
                initialRegion={{
                    latitude: MAP_CENTER[1],
                    longitude: MAP_CENTER[0],
                    latitudeDelta: 0.1,
                    longitudeDelta: 0.1,
                }}
            >
                {/* ── Sightings markers ── */}
                {sightings.map((f: any) => (
                    <Marker
                        key={f.properties.id}
                        coordinate={{
                            latitude: f.geometry.coordinates[1],
                            longitude: f.geometry.coordinates[0],
                        }}
                        title={f.properties.species}
                        description={`Severity: ${f.properties.severity}`}
                        pinColor={
                            f.properties.severity === 'critical' ? '#ef4444' :
                                f.properties.severity === 'high' ? '#f59e0b' :
                                    '#3b82f6'
                        }
                    />
                ))}

                {/* ── Weather overlay ── */}
                {weatherVisible && weatherPolygons.map((p) => (
                    <Polygon
                        key={p.id}
                        coordinates={p.coordinates}
                        fillColor={p.color}
                        strokeColor="rgba(255,255,255,0.3)"
                        strokeWidth={1}
                    />
                ))}
            </MapView>

            {/* ── Weather toggle FAB ── */}
            <TouchableOpacity
                style={[styles.weatherFab, weatherVisible && styles.weatherFabActive]}
                onPress={toggleWeather}
                activeOpacity={0.8}
            >
                {weatherLoading ? (
                    <ActivityIndicator size="small" color="#fff" />
                ) : (
                    <>
                        <Text style={styles.fabIcon}>🌡️</Text>
                        <Text style={styles.fabLabel}>
                            Weather {weatherVisible ? 'On' : 'Off'}
                        </Text>
                    </>
                )}
            </TouchableOpacity>

            {/* ── Connection status ── */}
            <View style={styles.statusBar}>
                <View style={styles.statusDot} />
                <Text style={styles.statusText}>
                    {loading ? 'Connecting…' : 'Live · Davis, CA'}
                </Text>
            </View>

            {/* ── At-risk crops bottom sheet ── */}
            <AtRiskCrops sightings={sightings} />
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#0a0e17',
    },
    map: {
        flex: 1,
    },

    /* Weather FAB */
    weatherFab: {
        position: 'absolute',
        top: Platform.OS === 'ios' ? 60 : 40,
        right: 16,
        flexDirection: 'row',
        alignItems: 'center',
        gap: 6,
        backgroundColor: 'rgba(17, 24, 39, 0.8)',
        paddingHorizontal: 14,
        paddingVertical: 10,
        borderRadius: 10,
        borderWidth: 1,
        borderColor: 'rgba(99, 102, 241, 0.2)',
    },
    weatherFabActive: {
        backgroundColor: 'rgba(99, 102, 241, 0.25)',
        borderColor: '#6366f1',
    },
    fabIcon: { fontSize: 16 },
    fabLabel: {
        color: '#f1f5f9',
        fontSize: 13,
        fontWeight: '600',
    },

    /* Status bar */
    statusBar: {
        position: 'absolute',
        top: Platform.OS === 'ios' ? 60 : 40,
        left: 16,
        flexDirection: 'row',
        alignItems: 'center',
        gap: 6,
        backgroundColor: 'rgba(17, 24, 39, 0.8)',
        paddingHorizontal: 12,
        paddingVertical: 8,
        borderRadius: 10,
        borderWidth: 1,
        borderColor: 'rgba(99, 102, 241, 0.2)',
    },
    statusDot: {
        width: 8,
        height: 8,
        borderRadius: 4,
        backgroundColor: '#22c55e',
    },
    statusText: {
        color: '#94a3b8',
        fontSize: 12,
        fontWeight: '500',
    },
});
