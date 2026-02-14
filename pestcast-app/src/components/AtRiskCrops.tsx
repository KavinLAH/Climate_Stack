import React, { useEffect, useMemo, useCallback, useRef } from 'react';
import {
    View,
    Text,
    StyleSheet,
    Dimensions,
} from 'react-native';
import BottomSheet, { BottomSheetScrollView } from '@gorhom/bottom-sheet';

/* ── Agricultural zones near Davis / Central Valley ── */
const AG_ZONES = [
    { name: 'Rice', icon: '🌾', lat: 38.56, lng: -121.72, radiusKm: 15 },
    { name: 'Tomatoes', icon: '🍅', lat: 38.50, lng: -121.80, radiusKm: 12 },
    { name: 'Almonds', icon: '🌳', lat: 38.60, lng: -121.65, radiusKm: 18 },
    { name: 'Grapes', icon: '🍇', lat: 38.45, lng: -121.75, radiusKm: 10 },
    { name: 'Corn', icon: '🌽', lat: 38.55, lng: -121.68, radiusKm: 14 },
];

const SEVERITY_MULT: Record<string, number> = {
    low: 0.5,
    medium: 1,
    high: 2,
    critical: 4,
};

/** Haversine distance in km. */
function haversine(lat1: number, lng1: number, lat2: number, lng2: number): number {
    const R = 6371;
    const dLat = ((lat2 - lat1) * Math.PI) / 180;
    const dLng = ((lng2 - lng1) * Math.PI) / 180;
    const a =
        Math.sin(dLat / 2) ** 2 +
        Math.cos((lat1 * Math.PI) / 180) *
        Math.cos((lat2 * Math.PI) / 180) *
        Math.sin(dLng / 2) ** 2;
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function threatLevel(score: number): string {
    if (score >= 5) return 'critical';
    if (score >= 2) return 'high';
    if (score >= 0.5) return 'medium';
    return 'low';
}

const LEVEL_COLORS: Record<string, string> = {
    critical: '#ef4444',
    high: '#f59e0b',
    medium: '#3b82f6',
    low: '#22c55e',
};

interface Props {
    sightings: GeoJSON.Feature[];
}

export default function AtRiskCrops({ sightings }: Props) {
    const bottomSheetRef = useRef<BottomSheet>(null);
    const snapPoints = useMemo(() => ['12%', '45%', '75%'], []);

    const ranked = useMemo(() => {
        return AG_ZONES.map((zone) => {
            let threat = 0;
            let nearestDist = Infinity;
            let nearby = 0;

            for (const f of sightings) {
                const coords = (f.geometry as any)?.coordinates;
                if (!coords) continue;
                const [lng, lat] = coords;
                const dist = haversine(zone.lat, zone.lng, lat, lng);
                if (dist > 50) continue;

                const sev = SEVERITY_MULT[f.properties?.severity] || 1;
                threat += sev / Math.max(dist * dist, 0.1);
                if (dist < nearestDist) nearestDist = dist;
                nearby++;
            }

            return {
                ...zone,
                threat,
                level: threatLevel(threat),
                nearestDist: nearestDist === Infinity ? null : nearestDist,
                nearby,
            };
        })
            .sort((a, b) => b.threat - a.threat)
            .slice(0, 3);
    }, [sightings]);

    const totalSightings = sightings.length;
    const uniqueSpecies = new Set(
        sightings.map((f) => f.properties?.species).filter(Boolean)
    ).size;
    const criticalCount = sightings.filter(
        (f) => f.properties?.severity === 'critical'
    ).length;

    return (
        <BottomSheet
            ref={bottomSheetRef}
            index={0}
            snapPoints={snapPoints}
            backgroundStyle={styles.sheetBg}
            handleIndicatorStyle={styles.handleIndicator}
            enablePanDownToClose={false}
        >
            {/* Collapsed peek header */}
            <View style={styles.peekHeader}>
                <Text style={styles.peekIcon}>🎯</Text>
                <Text style={styles.peekTitle}>Top 3 At-Risk Crops</Text>
                <View style={styles.peekStats}>
                    <Text style={styles.peekStatValue}>{totalSightings}</Text>
                    <Text style={styles.peekStatLabel}> sightings</Text>
                </View>
            </View>

            <BottomSheetScrollView contentContainerStyle={styles.scrollContent}>
                {/* Crop cards */}
                {ranked.map((crop, i) => (
                    <View key={crop.name} style={styles.card}>
                        <View
                            style={[
                                styles.cardEdge,
                                { backgroundColor: LEVEL_COLORS[crop.level] },
                            ]}
                        />
                        <View style={styles.cardBody}>
                            <View style={styles.cardHeader}>
                                <View style={styles.cropName}>
                                    <Text style={styles.cropIcon}>{crop.icon}</Text>
                                    <Text style={styles.cropLabel}>{crop.name}</Text>
                                </View>
                                <View style={styles.rank}>
                                    <Text style={styles.rankText}>{i + 1}</Text>
                                </View>
                            </View>

                            <View
                                style={[
                                    styles.badge,
                                    { backgroundColor: LEVEL_COLORS[crop.level] + '25' },
                                ]}
                            >
                                <View
                                    style={[
                                        styles.badgeDot,
                                        { backgroundColor: LEVEL_COLORS[crop.level] },
                                    ]}
                                />
                                <Text
                                    style={[styles.badgeText, { color: LEVEL_COLORS[crop.level] }]}
                                >
                                    {crop.level.toUpperCase()} THREAT
                                </Text>
                            </View>

                            <View style={styles.statsRow}>
                                <Text style={styles.statText}>
                                    Nearest:{' '}
                                    <Text style={styles.statValue}>
                                        {crop.nearestDist
                                            ? `${crop.nearestDist.toFixed(1)} km`
                                            : '—'}
                                    </Text>
                                </Text>
                                <Text style={styles.statText}>
                                    Sightings:{' '}
                                    <Text style={styles.statValue}>{crop.nearby}</Text>
                                </Text>
                            </View>
                        </View>
                    </View>
                ))}

                {ranked.length === 0 && (
                    <Text style={styles.emptyText}>
                        No sighting data yet. Sightings will appear here in real time.
                    </Text>
                )}

                {/* Summary stats */}
                <View style={styles.summaryRow}>
                    <View style={styles.summaryItem}>
                        <Text style={styles.summaryValue}>{totalSightings}</Text>
                        <Text style={styles.summaryLabel}>SIGHTINGS</Text>
                    </View>
                    <View style={styles.summaryItem}>
                        <Text style={styles.summaryValue}>{uniqueSpecies}</Text>
                        <Text style={styles.summaryLabel}>SPECIES</Text>
                    </View>
                    <View style={styles.summaryItem}>
                        <Text style={styles.summaryValue}>{criticalCount}</Text>
                        <Text style={styles.summaryLabel}>CRITICAL</Text>
                    </View>
                </View>
            </BottomSheetScrollView>
        </BottomSheet>
    );
}

const styles = StyleSheet.create({
    sheetBg: {
        backgroundColor: 'rgba(17, 24, 39, 0.95)',
        borderTopLeftRadius: 20,
        borderTopRightRadius: 20,
    },
    handleIndicator: {
        backgroundColor: '#4b5563',
        width: 36,
    },
    peekHeader: {
        flexDirection: 'row',
        alignItems: 'center',
        paddingHorizontal: 20,
        paddingBottom: 8,
        gap: 8,
    },
    peekIcon: { fontSize: 16 },
    peekTitle: {
        color: '#f1f5f9',
        fontSize: 15,
        fontWeight: '700',
        flex: 1,
    },
    peekStats: { flexDirection: 'row', alignItems: 'baseline' },
    peekStatValue: {
        color: '#818cf8',
        fontSize: 16,
        fontWeight: '800',
    },
    peekStatLabel: { color: '#64748b', fontSize: 11 },

    scrollContent: {
        paddingHorizontal: 16,
        paddingBottom: 40,
    },
    card: {
        backgroundColor: 'rgba(30, 41, 59, 0.7)',
        borderRadius: 14,
        flexDirection: 'row',
        marginBottom: 10,
        overflow: 'hidden',
    },
    cardEdge: {
        width: 4,
    },
    cardBody: {
        flex: 1,
        padding: 14,
    },
    cardHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 8,
    },
    cropName: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 8,
    },
    cropIcon: { fontSize: 22 },
    cropLabel: {
        color: '#f1f5f9',
        fontSize: 16,
        fontWeight: '700',
    },
    rank: {
        width: 26,
        height: 26,
        borderRadius: 13,
        backgroundColor: '#6366f1',
        alignItems: 'center',
        justifyContent: 'center',
    },
    rankText: {
        color: '#fff',
        fontSize: 12,
        fontWeight: '800',
    },

    badge: {
        flexDirection: 'row',
        alignItems: 'center',
        alignSelf: 'flex-start',
        borderRadius: 20,
        paddingHorizontal: 10,
        paddingVertical: 3,
        gap: 5,
        marginBottom: 8,
    },
    badgeDot: { width: 6, height: 6, borderRadius: 3 },
    badgeText: { fontSize: 10, fontWeight: '700', letterSpacing: 0.8 },

    statsRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
    },
    statText: { color: '#94a3b8', fontSize: 12 },
    statValue: { color: '#f1f5f9', fontWeight: '600' },

    emptyText: {
        color: '#64748b',
        fontSize: 13,
        textAlign: 'center',
        paddingVertical: 20,
    },

    summaryRow: {
        flexDirection: 'row',
        justifyContent: 'space-around',
        marginTop: 16,
        paddingTop: 16,
        borderTopWidth: 1,
        borderTopColor: 'rgba(99, 102, 241, 0.18)',
    },
    summaryItem: { alignItems: 'center' },
    summaryValue: {
        fontSize: 20,
        fontWeight: '800',
        color: '#818cf8',
    },
    summaryLabel: {
        fontSize: 10,
        color: '#64748b',
        letterSpacing: 0.6,
        marginTop: 2,
    },
});
