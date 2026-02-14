import React, { useEffect, useState, useCallback, useRef } from 'react';
import {
    View,
    Text,
    TouchableOpacity,
    StyleSheet,
    ActivityIndicator,
    Platform,
} from 'react-native';
import MapboxGL from '@rnmapbox/maps';
import { MAPBOX_TOKEN, MAP_CENTER, MAP_ZOOM } from '../config';
import { fetchSightings, subscribeSightings } from '../services/supabase';
import { fetchTemperatureGrid } from '../services/noaa';
import AtRiskCrops from '../components/AtRiskCrops';

MapboxGL.setAccessToken(MAPBOX_TOKEN);

/* ── Heatmap layer style ── */
const heatmapStyle: MapboxGL.HeatmapLayerStyle = {
    heatmapWeight: [
        'interpolate',
        ['linear'],
        [
            'match',
            ['get', 'severity'],
            'low', 0.25,
            'medium', 0.5,
            'high', 0.75,
            'critical', 1.0,
            0.5,
        ],
        0, 0,
        1, 1,
    ] as any,
    heatmapIntensity: [
        'interpolate', ['linear'], ['zoom'],
        0, 1,
        9, 3,
    ] as any,
    heatmapRadius: [
        'interpolate', ['linear'], ['zoom'],
        0, 4,
        9, 30,
        14, 50,
    ] as any,
    heatmapColor: [
        'interpolate', ['linear'], ['heatmap-density'],
        0, 'rgba(0,0,0,0)',
        0.1, '#0d1b2a',
        0.25, '#1b4965',
        0.4, '#2a9d8f',
        0.55, '#e9c46a',
        0.7, '#f4a261',
        0.85, '#e76f51',
        1, '#d62828',
    ] as any,
    heatmapOpacity: [
        'interpolate', ['linear'], ['zoom'],
        7, 1,
        15, 0.4,
    ] as any,
};

/* ── Individual sighting dots at high zoom ── */
const circleStyle: MapboxGL.CircleLayerStyle = {
    circleRadius: [
        'interpolate', ['linear'], ['zoom'],
        13, 3,
        18, 8,
    ] as any,
    circleColor: [
        'match', ['get', 'severity'],
        'critical', '#ef4444',
        'high', '#f59e0b',
        'medium', '#3b82f6',
        'low', '#22c55e',
        '#94a3b8',
    ] as any,
    circleStrokeWidth: 1,
    circleStrokeColor: 'rgba(255,255,255,0.3)',
    circleOpacity: 0.85,
};

/* ── Weather temperature overlay (FillLayer) ── */
const weatherFillStyle: MapboxGL.FillLayerStyle = {
    fillColor: [
        'case',
        ['>', ['get', 'temp'], 80], 'rgba(255, 0, 0, 0.4)',
        ['<', ['get', 'temp'], 60], 'rgba(0, 0, 255, 0.2)',
        'rgba(255, 255, 255, 0.1)',
    ] as any,
    fillOutlineColor: 'rgba(255, 255, 255, 0.2)',
};

const mockWeatherGrid = (): GeoJSON.FeatureCollection => {
    const features: GeoJSON.Feature[] = [];
    const startLat = 38.56;
    const startLng = -121.76;
    const step = 0.01;

    for (let i = 0; i < 4; i++) {
        for (let j = 0; j < 4; j++) {
            const lat = startLat - i * step;
            const lng = startLng + j * step;
            const temp = i % 2 === 0 ? 85 : 55; // Alternating hot/cold for demo

            features.push({
                type: 'Feature',
                properties: { temp },
                geometry: {
                    type: 'Polygon',
                    coordinates: [[
                        [lng, lat],
                        [lng + step, lat],
                        [lng + step, lat - step],
                        [lng, lat - step],
                        [lng, lat],
                    ]],
                },
            });
        }
    }
    return { type: 'FeatureCollection', features };
};

const EMPTY_GEO: GeoJSON.FeatureCollection = {
    type: 'FeatureCollection',
    features: [],
};

export default function MapDashboard() {
    const [sightings, setSightings] = useState<GeoJSON.Feature[]>([]);
    const [geoJson, setGeoJson] = useState<GeoJSON.FeatureCollection>(EMPTY_GEO);
    const [weatherGeo, setWeatherGeo] = useState<GeoJSON.FeatureCollection>(EMPTY_GEO);
    const [weatherVisible, setWeatherVisible] = useState(false);
    const [weatherLoading, setWeatherLoading] = useState(false);
    const [loading, setLoading] = useState(true);
    const weatherLoadedRef = useRef(false);

    /* ── Load initial sightings + subscribe ── */
    useEffect(() => {
        let unsub: (() => void) | null = null;

        (async () => {
            const geo = await fetchSightings();
            setGeoJson(geo);
            setSightings(geo.features);
            setLoading(false);

            unsub = subscribeSightings((feature) => {
                setGeoJson((prev) => ({
                    ...prev,
                    features: [...prev.features, feature],
                }));
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
            // Simulating a fetch delay but using our mock grid
            setTimeout(() => {
                setWeatherGeo(mockWeatherGrid());
                setWeatherLoading(false);
            }, 500);
        }
    }, [weatherVisible]);

    return (
        <View style={styles.container}>
            <MapboxGL.MapView
                style={styles.map}
                styleURL="mapbox://styles/mapbox/dark-v11"
                logoEnabled={false}
                attributionEnabled={false}
            >
                <MapboxGL.Camera
                    defaultSettings={{
                        centerCoordinate: MAP_CENTER,
                        zoomLevel: MAP_ZOOM,
                    }}
                />

                {/* ── Sightings heatmap ── */}
                <MapboxGL.ShapeSource id="sightings-source" shape={geoJson}>
                    <MapboxGL.HeatmapLayer
                        id="sightings-heat"
                        maxZoomLevel={16}
                        style={heatmapStyle}
                    />
                    <MapboxGL.CircleLayer
                        id="sightings-points"
                        minZoomLevel={13}
                        style={circleStyle}
                    />
                </MapboxGL.ShapeSource>

                {/* ── Weather overlay ── */}
                {weatherVisible && (
                    <MapboxGL.ShapeSource id="weather-source" shape={weatherGeo}>
                        <MapboxGL.FillLayer
                            id="weather-layer"
                            style={weatherFillStyle}
                        />
                    </MapboxGL.ShapeSource>
                )}
            </MapboxGL.MapView>

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
